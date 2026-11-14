"""Furrow ML layer: per-pixel predictions aggregated to H3 cells whose centre lies inside the bbox."""

import argparse
import json
from dataclasses import dataclass, field
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
from fs_ml.models import decision_rule_text

_UNPREDICTED = "unpredicted"
LOW_CONFIDENCE = "(низкая достоверность)"


class FurrowCell(BaseModel):
    """crop_mode: dominant."""

    crop: str = Field(description="ключ культуры Furrow (pipeline/crops.py)")
    crop_pixel_share: float = Field(
        ge=0,
        le=1,
        description="доля пикселей ячейки внутри bbox, предсказанных как crop; не вероятность модели",
    )


class CompositionCell(BaseModel):
    """crop_mode: composition."""

    crop: str = Field(description="состав ячейки по предсказанию: группа и доля пикселей, other не выводится")
    crop_p: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="доля пикселей самой большой группы с precision выше порога; не вероятность модели",
    )


class FurrowLayer(BaseModel):
    model: str
    version: str
    validation: str
    cells: dict[str, FurrowCell | CompositionCell]


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


def _cell_counts(
    pixels: pd.DataFrame, bbox: tuple[float, float, float, float], resolution: int, report: ExportReport
) -> dict[str, pd.Series]:
    """Predicted-group counts of in-bbox pixels for every centre-inside cell (cells without pixels are reported)."""
    inside_cells, report.centre_outside_bbox = split_cells(bbox, resolution)
    keep = pixels["in_bbox"].to_numpy() & (pixels["dominant_crop"].to_numpy() != _UNPREDICTED)
    chosen = pixels.loc[keep, ["lon", "lat", "dominant_crop"]]
    keys = [h3.latlng_to_cell(y, x, resolution) for x, y in zip(chosen["lon"], chosen["lat"], strict=True)]
    chosen = chosen.assign(cell=keys)
    out: dict[str, pd.Series] = {}
    for cell in inside_cells:
        counts = chosen.loc[chosen["cell"] == cell, "dominant_crop"].value_counts()
        if counts.empty:
            report.skipped_reasons[cell] = "нет пикселей внутри bbox"
            continue
        out[cell] = counts
    return out


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


def compose(
    pixels: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    resolution: int,
    furrow_keys: dict[str, str],
    other_group: str,
    blocked: dict[str, dict[str, float]],
) -> tuple[dict[str, CompositionCell], ExportReport]:
    """crop_mode composition: share of every predicted group per centre-inside cell (other not shown)."""
    report = ExportReport(groups_below_precision=blocked)
    cells: dict[str, CompositionCell] = {}
    for cell, counts in _cell_counts(pixels, bbox, resolution, report).items():
        shares = {str(k): float(v / counts.sum()) for k, v in counts.items()}
        text, crop_p = composition_text(shares, furrow_keys, other_group, set(blocked))
        if not text:
            report.skipped_reasons[cell] = f"в ячейке только {other_group}, ячейка не записана"
            continue
        cells[cell] = CompositionCell(crop=text, crop_p=crop_p)
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


def validation_text(
    summary: dict[str, Any],
    bcfg: BaselineConfig,
    crops: CropsConfig,
    notes: list[str],
    bbox: tuple[float, float, float, float],
    year: int,
    sensor: str,
) -> str:
    """Human-readable validation line; every number and setting comes from the summary or the configs."""
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
        "другие регионы и годы не проверялись",
    ]
    return "; ".join(parts)


def build_layer(
    ml: MLConfig, crops: CropsConfig, furrow: FurrowExport, summary: dict[str, Any], class_metrics: pd.DataFrame,
    pixels: pd.DataFrame,
) -> tuple[FurrowLayer, ExportReport]:
    groups = list(crops.groups) + [crops.other_group]
    blocked = blocked_groups(class_metrics, list(crops.groups), furrow.min_precision)
    cells: dict[str, FurrowCell | CompositionCell]
    if furrow.crop_mode == "composition":
        composed, report = compose(
            pixels, ml.ingest.bbox, furrow.h3_resolution, crops.furrow_keys, crops.other_group, blocked
        )
        cells = dict(composed)
        notes = composition_notes(crops, blocked, furrow.min_precision)
    else:
        dominant, report = aggregate(pixels, ml.ingest.bbox, furrow.h3_resolution, crops.furrow_keys, groups, blocked)
        cells = dict(dominant)
        notes = [
            *low_confidence_notes(crops, class_metrics, blocked),
            "crop_pixel_share — доля пикселей, не вероятность модели",
        ]
    year = min(ml.ingest.dates).year
    layer = FurrowLayer(
        model=furrow.model,
        version=furrow.version,
        validation=validation_text(summary, ml.baseline, crops, notes, ml.ingest.bbox, year, ml.ingest.short_name),
        cells=cells,
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
    layer, report = build_layer(ml, crops, furrow, summary, class_metrics, pixels)
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
