"""Furrow ML layer: per-pixel predictions aggregated to H3 cells whose centre lies inside the bbox."""

import argparse
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import h3
import pandas as pd
from pydantic import BaseModel, Field

from fs_ml.config import (
    BaselineConfig,
    CropsConfig,
    FurrowExport,
    MLConfig,
    baseline_fingerprint,
    load_crops_config,
    load_ml_config,
)
from fs_ml.inference import low_obs_months, season_months
from fs_ml.models import decision_rule_text

_UNPREDICTED = "unpredicted"
_NODATA = "nodata"
LOW_CONFIDENCE = "(низкая достоверность)"
# Share of a cell's pixels (with observations in the month) that must be bare for the month to be bare in the cell.
_CELL_BARE_SHARE = 0.5


class YearEntry(BaseModel):
    """One season of a cell. All fields optional so older readers and partial data stay valid."""

    crop: str | None = Field(default=None, description="состав по долям пикселей ячейки, other не выводится")
    source: str | None = Field(default=None, description="USDA CDL (факт) или model (предсказание, год со звёздочкой)")
    bare_obs_dates: int | None = Field(default=None, description="медиана по пикселям: чистые даты с NDVI ниже порога")
    bare_months: list[int] | None = Field(default=None, description="месяцы, голые у не менее половины пикселей")
    low_obs_months: list[int] | None = Field(
        default=None, description="месяцы с одной чистой датой и меньше: мало наблюдений, в bare_months не входят"
    )


class FurrowCell(BaseModel):
    """crop_mode: dominant."""

    crop: str = Field(description="ключ культуры Furrow (pipeline/crops.py)")
    crop_pixel_share: float = Field(
        ge=0,
        le=1,
        description="доля пикселей ячейки внутри bbox, предсказанных как crop; не вероятность модели",
    )
    years: dict[str, YearEntry] | None = None


class CompositionCell(BaseModel):
    """crop_mode: composition."""

    crop: str = Field(description="состав ячейки по предсказанию: группа и доля пикселей, other не выводится")
    crop_p: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="доля пикселей самой большой группы с precision выше порога; не вероятность модели",
    )
    crop_top: str | None = Field(
        default=None, description="ключ самой большой группы ячейки без other (для раскраски карты)"
    )
    crop_top_share: float | None = Field(default=None, ge=0, le=1, description="доля пикселей crop_top")
    low_confidence: bool | None = Field(
        default=None, description="true, если precision группы crop_top по bbox не выше порога"
    )
    years: dict[str, YearEntry] | None = None


class DetailLayer(BaseModel):
    h3_resolution: int
    cells: dict[str, FurrowCell | CompositionCell]


class FurrowLayer(BaseModel):
    model: str
    version: str
    validation: str
    cells: dict[str, FurrowCell | CompositionCell]
    detail: DetailLayer | None = None


@dataclass
class ExportReport:
    written: dict[str, dict[str, Any]] = field(default_factory=dict)
    centre_outside_bbox: list[str] = field(default_factory=list)
    skipped_reasons: dict[str, str] = field(default_factory=dict)
    groups_without_furrow_key: list[str] = field(default_factory=list)
    groups_below_precision: dict[str, dict[str, float]] = field(default_factory=dict)


def check_baseline_matches(summary: dict[str, Any], ml: MLConfig, crops: CropsConfig) -> None:
    """Refuse to export from reports made with other settings than the current configs."""
    expected = {
        "config_sha256": baseline_fingerprint(ml, crops),
        "decision_rule": decision_rule_text(ml.baseline),
        "min_proba": dict(ml.baseline.min_proba),
    }
    diffs = [
        f"{key}: в отчёте {summary.get(key)!r}, в конфиге {value!r}"
        for key, value in expected.items()
        if summary.get(key) != value
    ]
    if diffs:
        raise RuntimeError(
            "отчёты baseline сделаны с другим конфигом, перезапусти baseline "
            "(python -m fs_ml.baseline <config> <crops>): " + "; ".join(diffs)
        )


def blocked_groups(class_metrics: pd.DataFrame, groups: list[str], min_precision: float) -> dict[str, dict[str, float]]:
    """Groups whose bbox precision is not above min_precision (precision <= threshold): low confidence."""
    blocked: dict[str, dict[str, float]] = {}
    for group in groups:
        precision = float(class_metrics.loc[group, "precision_bbox"])
        if precision <= min_precision:
            blocked[group] = {"precision_bbox": round(precision, 4), "min_precision": min_precision}
    return blocked


def _inside(lon: float, lat: float, bbox: tuple[float, float, float, float]) -> bool:
    west, south, east, north = bbox
    return west <= lon <= east and south <= lat <= north


def split_cells(bbox: tuple[float, float, float, float], resolution: int) -> tuple[list[str], list[str]]:
    """H3 cells overlapping the bbox, split into (centre inside bbox, centre outside bbox)."""
    west, south, east, north = bbox
    polygon = h3.LatLngPoly([(south, west), (south, east), (north, east), (north, west)])
    overlapping: list[str] = sorted(h3.h3shape_to_cells_experimental(polygon, resolution, contain="overlap"))
    inside: list[str] = []
    outside: list[str] = []
    for cell in overlapping:
        lat, lon = h3.cell_to_latlng(cell)
        (inside if _inside(lon, lat, bbox) else outside).append(cell)
    return inside, outside


def _cell_groups(
    pixels: pd.DataFrame, bbox: tuple[float, float, float, float], resolution: int, report: ExportReport
) -> dict[str, pd.DataFrame]:
    """In-bbox predicted pixels grouped by centre-inside cell (cells without pixels are reported)."""
    inside_cells, report.centre_outside_bbox = split_cells(bbox, resolution)
    keep = pixels["in_bbox"].to_numpy() & (pixels["dominant_crop"].to_numpy() != _UNPREDICTED)
    chosen = pixels.loc[keep]
    keys = [h3.latlng_to_cell(y, x, resolution) for x, y in zip(chosen["lon"], chosen["lat"], strict=True)]
    chosen = chosen.assign(cell=keys)
    inside = set(inside_cells)
    groups = {str(cell): frame for cell, frame in chosen.groupby("cell") if cell in inside}
    for cell in inside_cells:
        if cell not in groups:
            report.skipped_reasons[cell] = "нет пикселей внутри bbox"
    return {cell: groups[cell] for cell in inside_cells if cell in groups}


def _cell_counts(
    pixels: pd.DataFrame, bbox: tuple[float, float, float, float], resolution: int, report: ExportReport
) -> dict[str, pd.Series]:
    """Predicted-group counts of in-bbox pixels for every centre-inside cell (cells without pixels are reported)."""
    return {
        cell: frame["dominant_crop"].value_counts()
        for cell, frame in _cell_groups(pixels, bbox, resolution, report).items()
    }


def history_years(pixels: pd.DataFrame) -> list[int]:
    """Previous-year CDL columns present in the pixel table (cdl_group_YYYY)."""
    return sorted(int(c.removeprefix("cdl_group_")) for c in pixels.columns if c.startswith("cdl_group_"))


def bare_month_columns(pixels: pd.DataFrame) -> list[tuple[int, str]]:
    return sorted((int(c.removeprefix("bare_status_")), c) for c in pixels.columns if c.startswith("bare_status_"))


def cell_bare_months(frame: pd.DataFrame, skip: set[int] | None = None) -> list[int]:
    """Months bare for at least half of the cell's pixels that have observations in that month (status 1/0/-1).

    Months in `skip` (low observations) are never reported as bare.
    """
    out: list[int] = []
    for month, column in bare_month_columns(frame):
        if month in (skip or set()):
            continue
        status = frame[column].to_numpy()
        known = status >= 0
        if known.any() and float((status[known] == 1).mean()) >= _CELL_BARE_SHARE:
            out.append(month)
    return out


def cell_years(
    frame: pd.DataFrame,
    prediction_year: int,
    prediction_crop: str,
    furrow_keys: dict[str, str],
    other_group: str,
    low_obs: set[int] | None = None,
) -> dict[str, YearEntry]:
    """years for one cell: previous years from CDL (fact), the prediction year from the model."""
    years: dict[str, YearEntry] = {}
    for year in history_years(frame):
        groups = frame[f"cdl_group_{year}"]
        groups = groups[groups != _NODATA]
        if groups.empty:
            years[str(year)] = YearEntry(source="USDA CDL")
            continue
        shares = {str(k): float(v) for k, v in groups.value_counts(normalize=True).items()}
        text, _ = composition_text(shares, furrow_keys, other_group, set())
        years[str(year)] = YearEntry(crop=text or None, source="USDA CDL")
    entry = YearEntry(crop=prediction_crop or None, source="model")
    if "bare_obs_dates" in frame.columns:
        entry.bare_obs_dates = round(float(frame["bare_obs_dates"].median()))
    if bare_month_columns(frame):
        entry.bare_months = cell_bare_months(frame, low_obs)
        if low_obs:
            entry.low_obs_months = sorted(low_obs)
    years[str(prediction_year)] = entry
    return years


def _written_info(cell: str, counts: pd.Series) -> dict[str, Any]:
    return {
        "n_pixels_in_bbox": int(counts.sum()),
        "cell_area_km2": round(float(h3.cell_area(cell, unit="km^2")), 1),
        "group_shares": {str(k): round(float(v / counts.sum()), 3) for k, v in counts.items()},
    }


def aggregate(
    pixels: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    resolution: int,
    furrow_keys: dict[str, str],
    groups: list[str],
    blocked: dict[str, dict[str, float]],
) -> tuple[dict[str, FurrowCell], ExportReport]:
    """crop_mode dominant: majority predicted group per centre-inside cell over in-bbox pixels.

    A cell is not written when its dominant group has no Furrow key or is in `blocked` (precision too low).
    """
    report = ExportReport(
        groups_without_furrow_key=[g for g in groups if g not in furrow_keys], groups_below_precision=blocked
    )
    cells: dict[str, FurrowCell] = {}
    for cell, counts in _cell_counts(pixels, bbox, resolution, report).items():
        group = str(counts.index[0])
        share = round(float(counts.iloc[0] / counts.sum()), 4)
        if group not in furrow_keys:
            report.skipped_reasons[cell] = f"доминирующая группа {group} ({share:.3f}) без ключа Furrow, ячейка не записана"
            continue
        if group in blocked:
            report.skipped_reasons[cell] = (
                f"доминирующая группа {group} ({share:.3f}): precision по bbox {blocked[group]['precision_bbox']:.3f} "
                f"не выше порога {blocked[group]['min_precision']}, ячейка не записана"
            )
            continue
        cells[cell] = FurrowCell(crop=furrow_keys[group], crop_pixel_share=share)
        report.written[cell] = {"dominant_group": group, **_written_info(cell, counts)}
    return cells, report


def composition_text(
    shares: dict[str, float], furrow_keys: dict[str, str], other_group: str, low: set[str]
) -> tuple[str, float | None]:
    """'alfalfa 20%, corn 19%, fallow 32% (низкая достоверность)' and crop_p.

    Confident groups first, then low-confidence ones, each by share descending; other and groups that round to
    0% are left out. Labels are Furrow keys where they exist, else our group names. crop_p is the share of the
    largest confident group (None if there is none).
    """
    shown = {g: s for g, s in shares.items() if g != other_group and round(s * 100) >= 1}
    confident = sorted((g for g in shown if g not in low), key=lambda g: -shown[g])
    weak = sorted((g for g in shown if g in low), key=lambda g: -shown[g])
    parts = [f"{furrow_keys.get(g, g)} {round(shown[g] * 100)}%" for g in confident]
    parts += [f"{furrow_keys.get(g, g)} {round(shown[g] * 100)}% {LOW_CONFIDENCE}" for g in weak]
    crop_p = round(shown[confident[0]], 4) if confident else None
    return ", ".join(parts), crop_p


def top_group(shares: dict[str, float], other_group: str) -> tuple[str, float] | None:
    """Largest group of the cell other than other_group, with its share (None if the cell is only other)."""
    candidates = {g: s for g, s in shares.items() if g != other_group}
    if not candidates:
        return None
    group = max(candidates, key=lambda g: (candidates[g], g))
    return group, candidates[group]


def compose(
    pixels: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    resolution: int,
    furrow_keys: dict[str, str],
    other_group: str,
    blocked: dict[str, dict[str, float]],
    prediction_year: int | None = None,
    low_obs: set[int] | None = None,
) -> tuple[dict[str, CompositionCell], ExportReport]:
    """crop_mode composition: share of every predicted group per centre-inside cell (other not shown).

    With prediction_year set, each cell also gets `years` (previous-year CDL columns and bare-month columns are
    used when the pixel table has them).
    """
    report = ExportReport(groups_below_precision=blocked)
    cells: dict[str, CompositionCell] = {}
    for cell, frame in _cell_groups(pixels, bbox, resolution, report).items():
        counts = frame["dominant_crop"].value_counts()
        shares = {str(k): float(v / counts.sum()) for k, v in counts.items()}
        text, crop_p = composition_text(shares, furrow_keys, other_group, set(blocked))
        if not text:
            report.skipped_reasons[cell] = f"в ячейке только {other_group}, ячейка не записана"
            continue
        years = (
            cell_years(frame, prediction_year, text, furrow_keys, other_group, low_obs)
            if prediction_year is not None
            else None
        )
        top = top_group(shares, other_group)
        cells[cell] = CompositionCell(
            crop=text,
            crop_p=crop_p,
            crop_top=furrow_keys.get(top[0], top[0]) if top else None,
            crop_top_share=round(top[1], 4) if top else None,
            low_confidence=(top[0] in blocked) if top else None,
            years=years,
        )
        report.written[cell] = _written_info(cell, counts)
    return cells, report


def low_confidence_notes(
    crops: CropsConfig, class_metrics: pd.DataFrame, blocked: dict[str, dict[str, float]]
) -> list[str]:
    """One note per low-confidence group that has a Furrow key: either its precision or why it is not exported."""
    notes: list[str] = []
    for group in crops.low_confidence_groups:
        if group not in crops.furrow_keys:
            continue
        key = crops.furrow_keys[group]
        precision = float(class_metrics.loc[group, "precision_bbox"])
        recall = float(class_metrics.loc[group, "recall_bbox"])
        if group in blocked:
            threshold = blocked[group]["min_precision"]
            notes.append(f"{key} не экспортируется: precision по bbox {precision:.2f} не выше порога {threshold}")
        else:
            notes.append(f"{key} — низкая достоверность, см. precision ({precision:.2f}, recall {recall:.2f} по bbox)")
    return notes


def composition_notes(
    crops: CropsConfig, blocked: dict[str, dict[str, float]], min_precision: float
) -> list[str]:
    """Validation notes for crop_mode composition."""
    weak = ", ".join(f"{crops.furrow_keys.get(g, g)} {v['precision_bbox']:.2f}" for g, v in blocked.items())
    notes = [
        f"crop — состав ячейки по долям пикселей внутри bbox, {crops.other_group} не выводится",
        f"{LOW_CONFIDENCE} — precision по bbox не выше {min_precision}" + (f": {weak}" if weak else ""),
        f"crop_p — доля самой большой группы с precision выше {min_precision}, не вероятность модели",
    ]
    return notes


def years_notes(
    history: list[int], prediction_year: int, bare_ndvi: float, low_obs: dict[int, list[str]]
) -> list[str]:
    """Validation notes for cells[h3].years: sources and aggregation rules, plus months with few observations."""
    hist = "–".join(str(y) for y in (history[:1] + history[-1:] if len(history) > 1 else history))
    notes = []
    if history:
        notes.append(f"years: {hist} — история USDA CDL (факт), доли пикселей ячейки по группам")
    notes += [
        f"{prediction_year}* — предсказание модели (source: model)",
        f"bare_obs_dates — медиана по пикселям ячейки числа чистых дат с NDVI < {bare_ndvi}",
        (
            f"bare_months — месяцы, голые у не менее {_CELL_BARE_SHARE:.0%} пикселей ячейки с наблюдениями; "
            f"месяц пикселя голый, если NDVI < {bare_ndvi} на половине его чистых дат или больше, "
            "без чистых дат — неизвестно"
        ),
    ]
    for month, days in sorted(low_obs.items()):
        seen = ", ".join(days) if days else "нет дат"
        notes.append(
            f"месяц {month:02d}: мало наблюдений ({len(days)} чистая дата по bbox: {seen}), "
            "в bare_months не входит, см. low_obs_months"
        )
    return notes


def validation_text(
    summary: dict[str, Any],
    bcfg: BaselineConfig,
    crops: CropsConfig,
    notes: list[str],
    bbox: tuple[float, float, float, float],
    year: int,
    sensor: str,
    checks: list[str] | None = None,
) -> str:
    """Human-readable validation line; every number and setting comes from the summary or the configs.

    `checks` (export.furrow.checks) replaces the closing "not checked" note when other checks were run.
    """
    n_agri = len(crops.agricultural_groups)
    excluded = bcfg.buffer_rule_exclude_groups
    rule_part = f"без {', '.join(excluded)} {summary['macro_f1_rule_agri_truth']:.2f}; " if excluded else ""
    blocks = bcfg.spatial_blocks
    parts = [
        (
            f"macro F1 {summary['macro_f1_agri_truth']:.2f} по {n_agri} сельхоз-группам CDL {year} "
            f"на сельхоз-пикселях ({rule_part}на всём bbox {summary['macro_f1_agri_bbox']:.2f})"
        ),
        (
            f"accuracy bbox {summary['accuracy_bbox']:.2f} против {summary['majority_baseline_bbox']:.2f} "
            "у самого частого класса"
        ),
        f"пространственная блочная CV {blocks}x{blocks}, буфер {summary['buffer_px']} px",
        f"{sensor} {year}, {summary['dates_used']} из {summary['dates']} дат",
        f"признаки: {summary['features_text']}",
        f"только bbox {list(bbox)}",
        f"правило решения: {summary['decision_rule']}",
        *notes,
        *(checks or ["другие регионы и годы не проверялись"]),
    ]
    return "; ".join(parts)


def build_layer(
    ml: MLConfig,
    crops: CropsConfig,
    furrow: FurrowExport,
    summary: dict[str, Any],
    class_metrics: pd.DataFrame,
    pixels: pd.DataFrame,
    low_obs: dict[int, list[str]] | None = None,
) -> tuple[FurrowLayer, ExportReport]:
    groups = list(crops.groups) + [crops.other_group]
    blocked = blocked_groups(class_metrics, list(crops.groups), furrow.min_precision)
    year = min(ml.ingest.dates).year
    cells: dict[str, FurrowCell | CompositionCell]
    detail: DetailLayer | None = None
    if furrow.crop_mode == "composition":
        skip = set(low_obs or {})
        composed, report = compose(
            pixels, ml.ingest.bbox, furrow.h3_resolution, crops.furrow_keys, crops.other_group, blocked, year, skip
        )
        cells = dict(composed)
        notes = composition_notes(crops, blocked, furrow.min_precision)
        notes += years_notes(history_years(pixels), year, ml.thresholds.bare_soil_ndvi, low_obs or {})
        if furrow.detail_resolution is not None:
            fine, _ = compose(
                pixels,
                ml.ingest.bbox,
                furrow.detail_resolution,
                crops.furrow_keys,
                crops.other_group,
                blocked,
                year,
                skip,
            )
            detail = DetailLayer(h3_resolution=furrow.detail_resolution, cells=dict(fine))
            notes.append(f"detail — те же поля на ячейках H3 res {furrow.detail_resolution} с центром внутри bbox")
    else:
        dominant, report = aggregate(pixels, ml.ingest.bbox, furrow.h3_resolution, crops.furrow_keys, groups, blocked)
        cells = dict(dominant)
        notes = [
            *low_confidence_notes(crops, class_metrics, blocked),
            "crop_pixel_share — доля пикселей, не вероятность модели",
        ]
    layer = FurrowLayer(
        model=furrow.model,
        version=furrow.version,
        validation=validation_text(
            summary, ml.baseline, crops, notes, ml.ingest.bbox, year, ml.ingest.short_name, furrow.checks
        ),
        cells=cells,
        detail=detail,
    )
    return layer, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the Furrow ML layer for H3 cells centred inside the bbox.")
    parser.add_argument("config", type=Path, help="путь к configs/ml_bboxB.yaml")
    parser.add_argument("crops", type=Path, help="путь к configs/crops.yaml")
    args = parser.parse_args()
    ml = load_ml_config(args.config)
    crops = load_crops_config(args.crops)
    if ml.export is None or ml.export.furrow is None:
        raise ValueError(f"{args.config}: нет секции export.furrow")
    furrow = ml.export.furrow
    bcfg = ml.baseline

    summary: dict[str, Any] = json.loads((bcfg.reports_dir / "summary_2021.json").read_text(encoding="utf-8"))
    check_baseline_matches(summary, ml, crops)
    class_metrics = pd.read_csv(bcfg.reports_dir / "class_metrics_2021.csv").set_index("group")
    pixels = pd.read_parquet(bcfg.output_dir / "pixel_summary_2021.parquet")
    kept = [date.fromisoformat(d) for d in summary.get("date_list", [])]
    low_obs = low_obs_months(kept, season_months(ml.ingest.temporal)) if kept else {}
    layer, report = build_layer(ml, crops, furrow, summary, class_metrics, pixels, low_obs)
    furrow.path.parent.mkdir(parents=True, exist_ok=True)
    furrow.path.write_text(layer.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")

    year = min(ml.ingest.dates).year
    export_report = {
        "path": furrow.path.as_posix(),
        "crop_mode": furrow.crop_mode,
        "h3_resolution": furrow.h3_resolution,
        "config_sha256": summary["config_sha256"],
        "cells_written": report.written,
        "cells_centre_outside_bbox": report.centre_outside_bbox,
        "skipped_reasons": report.skipped_reasons,
        "groups_without_furrow_key": report.groups_without_furrow_key,
        "groups_below_precision": report.groups_below_precision,
        "furrow_keys": crops.furrow_keys,
    }
    report_path = bcfg.reports_dir / f"furrow_export_{year}.json"
    report_path.write_text(json.dumps(export_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"written {furrow.path}: {len(layer.cells)} cells ({furrow.crop_mode}); report {report_path}")
    print(json.dumps(export_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
