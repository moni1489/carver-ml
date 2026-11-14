"""Label-free layer for a region without CDL (Furrow vko): bare months and black fallow on cropland, per year.

No crop classes are predicted: there are no labels to train or check them. For every cropland pixel (ESA WorldCover
2021, class 40) and season month, the month is bare when NDVI < bare_ndvi on at least half of the pixel's clear HLS
dates in that month (the same rule as the Arizona export). A pixel-year is black fallow when it was bare in every
month of the peak season (`--fallow-months`, default June-July: every crop is green then) that had clear dates,
with at least `min_months` such months.

    python -m fs_ml.region_bare --years 2019 2020 2021 --hls D:/carver-data/vko/hls \
        --worldcover D:/carver-data/vko/worldcover/ESA_WorldCover_10m_2021_v200_N48E081_Map.tif --out data/ml/vko.json
"""

import argparse
import json
import re
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

import h3
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_origin, xy
from rasterio.warp import Resampling, reproject, transform, transform_bounds

from fs_ml.features import cloud_mask
from fs_ml.furrow import CompositionCell, DetailLayer, FurrowLayer, YearEntry
from fs_ml.soil_moisture import SM_MAX, SM_MIN, Downscaled, downscale, ndmi

_NAME = re.compile(r"HLS\.L30\.(T\w{5})\.(\d{4})(\d{3})T\d{6}\.v2\.0\.(B04|B05|B06|Fmask)\.tif$")
CROPLAND = 40  # ESA WorldCover class "Cropland"
FALLOW, CROPPED = "fallow", "cropped"
FALLOW_PART = "fallow_part"  # hotspot cell whose worst-year black-fallow share is 20-50% of its cropland
BARE, NOT_BARE, UNKNOWN = 1, 0, -1


def scenes(folder: Path, months: list[int]) -> dict[date, dict[str, list[Path]]]:
    """Files by acquisition date and band (several tiles per date)."""
    out: dict[date, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(folder.glob("*.tif")):
        m = _NAME.search(path.name)
        if not m:
            continue
        day = datetime.strptime(m.group(2) + m.group(3), "%Y%j").date()
        if day.month in months:
            out[day][m.group(4)].append(path)
    return {d: dict(b) for d, b in sorted(out.items()) if all(k in b for k in ("B04", "B05", "Fmask"))}


def accumulate_month(
    n_obs: np.ndarray, n_bare: np.ndarray, red: np.ndarray, nir: np.ndarray, fmask: np.ndarray,
    bare_ndvi: float, cloud_bits: list[int],
) -> None:
    """Add one date to the month counters: clear = Fmask cloud bits unset and both bands > 0 (as in the RF)."""
    ok = (red > 0) & (nir > 0) & ~cloud_mask(fmask, cloud_bits)
    r, n = red.astype(np.float32), nir.astype(np.float32)
    ndvi = np.zeros(red.shape, np.float32)
    ndvi[ok] = (n[ok] - r[ok]) / (n[ok] + r[ok])
    n_obs += ok
    n_bare += ok & (ndvi < bare_ndvi)


def clear_ndmi(nir: np.ndarray, swir1: np.ndarray, fmask: np.ndarray, cloud_bits: list[int]) -> np.ndarray:
    """NDMI on clear pixels (Fmask cloud bits unset and both bands > 0); NaN elsewhere."""
    masked = np.where(cloud_mask(fmask, cloud_bits), np.nan, 1.0).astype(np.float32)
    return ndmi(nir.astype(np.float32) * masked, swir1.astype(np.float32) * masked)


def month_status(n_obs: np.ndarray, n_bare: np.ndarray) -> np.ndarray:
    """1 bare (bare on >= half of clear dates), 0 not bare, -1 no clear date."""
    status = np.full(n_obs.shape, UNKNOWN, np.int8)
    known = n_obs > 0
    status[known] = np.where(2 * n_bare[known] >= n_obs[known], BARE, NOT_BARE)
    return status


def black_fallow(status: np.ndarray, skip: list[int], min_months: int) -> np.ndarray:
    """(M, ...) month status -> bare in every observed month, at least min_months observed (skip months ignored)."""
    use = np.array([i not in skip for i in range(status.shape[0])])
    s = status[use]
    observed = (s != UNKNOWN).sum(axis=0)
    green = (s == NOT_BARE).any(axis=0)
    out: np.ndarray = (observed >= min_months) & ~green
    return out


def build_grid(bbox_cells: tuple[float, float, float, float], crs: str) -> tuple[Any, tuple[int, int]]:
    """30 m grid in the HLS UTM zone covering the cells extent, snapped to multiples of 30 m like the HLS tiles."""
    south, west, north, east = bbox_cells
    left, bottom, right, top = transform_bounds("EPSG:4326", crs, west, south, east, north)
    left, bottom = np.floor(left / 30) * 30, np.floor(bottom / 30) * 30
    right, top = np.ceil(right / 30) * 30, np.ceil(top / 30) * 30
    width, height = int((right - left) / 30), int((top - bottom) / 30)
    return from_origin(left, top, 30, 30), (height, width)


def mosaic(paths: list[Path], affine: Any, shape: tuple[int, int], nodata: int) -> np.ndarray:
    height, width = shape
    bounds = (affine.c, affine.f - 30 * height, affine.c + 30 * width, affine.f)
    sources = [rasterio.open(p) for p in paths]
    try:
        arr, _ = merge(sources, bounds=bounds, res=30, nodata=nodata, method="first")
    finally:
        for s in sources:
            s.close()
    out: np.ndarray = arr[0][:height, :width]
    return out


def cropland_on_grid(path: Path, affine: Any, shape: tuple[int, int], crs: str) -> np.ndarray:
    """WorldCover 10 m -> 30 m grid by mode; True where the dominant class is cropland."""
    dst = np.zeros(shape, np.uint8)
    with rasterio.open(path) as src:
        reproject(rasterio.band(src, 1), dst, dst_transform=affine, dst_crs=crs, resampling=Resampling.mode)
    out: np.ndarray = dst == CROPLAND
    return out


def composition(share_fallow: float) -> tuple[str, str, float]:
    """'fallow 23%, cropped 77%' (larger first), top key and its share."""
    parts = sorted([(FALLOW, share_fallow), (CROPPED, 1 - share_fallow)], key=lambda x: -x[1])
    text = ", ".join(f"{k} {round(v * 100)}%" for k, v in parts if round(v * 100) >= 1)
    return text, parts[0][0], round(parts[0][1], 4)


def aggregate(
    keys: np.ndarray, years: list[int], fallow: dict[int, np.ndarray], status: dict[int, np.ndarray],
    months: list[int], low_obs: dict[int, list[int]], min_pixels: int, min_fallow: float = 0.0,
) -> dict[str, CompositionCell]:
    """One cell per H3 key with >= min_pixels cropland pixels; latest year on top, every year in `years`.

    min_fallow > 0 keeps only cells whose black-fallow share reaches it in at least one year (bare-land hotspots).
    """
    order = np.argsort(keys, kind="stable")
    sk = keys[order]
    starts = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1]])
    ends = np.r_[starts[1:], sk.size]
    cells: dict[str, CompositionCell] = {}
    for a, b in zip(starts, ends, strict=True):
        if b - a < min_pixels:
            continue
        idx = order[a:b]
        if min_fallow > 0 and max(float(fallow[y][idx].mean()) for y in years) < min_fallow:
            continue
        entries: dict[str, YearEntry] = {}
        for y in years:
            st = status[y][:, idx]
            bare = [m for i, m in enumerate(months)
                    if m not in low_obs[y] and (st[i] >= 0).any() and float((st[i][st[i] >= 0] == BARE).mean()) >= 0.5]
            text, _, _ = composition(float(fallow[y][idx].mean()))
            entries[str(y)] = YearEntry(crop=text, source="satellite", bare_months=bare, low_obs_months=low_obs[y] or None)
        text, top, share = composition(float(fallow[years[-1]][idx].mean()))
        if min_fallow > 0:  # hotspot map: colour by the worst year, so fallow in any year shows
            worst = max(float(fallow[y][idx].mean()) for y in years)
            top, share = (FALLOW if worst >= 0.5 else FALLOW_PART), round(worst, 4)
        cells[str(sk[a])] = CompositionCell(crop=text, crop_top=top, crop_top_share=share, low_confidence=None,
                                            years=entries)
    return cells


def add_soil_moisture(
    detail: dict[str, CompositionCell],
    ndmi_days: list[np.ndarray],
    hls_days: list[date],
    coarse: np.ndarray,
    fine: np.ndarray,
    smap_path: Path,
    year: int,
    months: list[int],
    max_days: int,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Disaggregate SMAP L4 root-zone moisture over detail cells by NDMI and format honest notes and report."""
    raw = json.loads(smap_path.read_text(encoding="utf-8"))
    smap_dates = [date.fromisoformat(d[:10]) for d in raw["dates"]]
    index = np.stack(ndmi_days)
    ndmi_days.clear()
    d: Downscaled = downscale(
        index,
        coarse,
        fine,
        hls_days,
        smap_dates,
        raw["cells"],
        max_days=max_days,
        months=(months[0], months[-1]),
        fine_keep=set(detail),
    )
    for key, cell in detail.items():
        if key in d.cells:
            med, p10, p90, _ = d.cells[key]
            cell.sm_rootzone, cell.sm_rootzone_pi = med, [p10, p90]
    medians = [c.sm_rootzone for c in detail.values() if c.sm_rootzone is not None]
    lo, hi = (min(medians), max(medians)) if medians else (0.0, 0.0)
    rs = [float(v["r"]) for v in d.checks.values() if v["r"] is not None]
    ns = [int(v["n"]) for v in d.checks.values() if v["r"] is not None]
    r_med = round(float(np.median(rs)), 2) if rs else 0.0
    r_lo, r_hi = (round(min(rs), 2), round(max(rs), 2)) if rs else (0.0, 0.0)
    n_txt = (str(ns[0]) if min(ns) == max(ns) else f"{min(ns)}–{max(ns)}") if ns else "0"
    span_m = f"{months[0]:02d}–{months[-1]:02d} {year}"
    ru = [
        (f"sm_rootzone — влага корневой зоны SMAP L4 (9 км, ячейки H3 res 5), распределённая по пикселям пашни по "
         f"спутниковому индексу влажности NDMI (HLS {year}, {d.dates_used} дат): sm = SMAP × (1 + k·(NDMI − среднее "
         f"NDMI ячейки res 5)), k = {d.k} — наибольший, при котором все пиксели в {SM_MIN}–{SM_MAX} м³/м³; среднее по "
         f"ячейке res 5 равно SMAP; облачные пиксели не заполнялись; ближайшая дата SMAP не дальше {max_days} дней"),
        (f"sm_rootzone — медиана по датам {span_m}, sm_rootzone_pi — 10–90 перцентиль по датам; ячеек detail с "
         f"влагой {len(medians)}, медианы {lo:.3f}–{hi:.3f} м³/м³"),
        (f"проверка правдоподобия по {len(rs)} ячейкам H3 res 5 (n = {n_txt} дат на ячейку): корреляция среднего NDMI "
         f"ячейки со SMAP — медиана r = {r_med:.2f} (разброс {r_lo:.2f}..{r_hi:.2f})"),
        *(["в среднем по региону связь NDMI и SMAP во времени слабая: распределение по NDMI этим не подтверждено"]
          if rs and abs(r_med) < 0.3 else []),
        "это распределение, а не предсказание модели; наземной проверки влаги нет",
    ]
    en = [
        (f"root-zone moisture: SMAP L4 (9 km, H3 res 5 cells) spread over cropland pixels by the satellite moisture "
         f"index NDMI (HLS {year}, {d.dates_used} dates), sm = SMAP × (1 + k·(NDMI − cell mean NDMI)), k = {d.k}, "
         f"the largest that keeps every pixel within {SM_MIN}–{SM_MAX} m³/m³; the res 5 cell mean equals SMAP; cloudy "
         f"pixels are left empty; nearest SMAP date within {max_days} days"),
        (f"value = median over {span_m} dates, interval = 10th–90th percentile over dates; {len(medians)} detail cells "
         f"with moisture, medians {lo:.3f}–{hi:.3f} m³/m³"),
        (f"plausibility check across {len(rs)} H3 res 5 cells (n = {n_txt} dates per cell): correlation of cell mean "
         f"NDMI with SMAP, median r = {r_med:.2f} (range {r_lo:.2f}..{r_hi:.2f})"),
        *([("across the region NDMI and SMAP barely move together over time, so this check does not support the NDMI "
            "split")] if rs and abs(r_med) < 0.3 else []),
        "a disaggregation, not a model prediction; no ground check of moisture",
    ]
    summary: dict[str, Any] = {
        "year": year,
        "k": d.k,
        "dates_used": d.dates_used,
        "pixel_range": list(d.pixel_range),
        "detail_cells_with_moisture": len(medians),
        "median_range": [round(lo, 3), round(hi, 3)],
        "r_median": r_med if rs else None,
        "r_range": [r_lo, r_hi] if rs else None,
        "checks": d.checks,
    }
    return ru, en, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Bare months and black fallow on cropland for a region without labels.")
    ap.add_argument("--years", type=int, nargs="+", required=True)
    ap.add_argument("--hls", type=Path, required=True, help="folder with <year>/ subfolders of HLS L30 B04, B05, Fmask")
    ap.add_argument("--worldcover", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--grid", type=Path, required=True,
                    help="Furrow region file (web/public/data/<region>.json): its cells are the grid, so keys always match")
    ap.add_argument("--months", type=int, nargs="+", default=[5, 6, 7, 8, 9])
    ap.add_argument("--bare-ndvi", type=float, default=0.2)
    ap.add_argument("--min-months", type=int, default=2)
    ap.add_argument("--fallow-months", type=int, nargs="+", default=[6, 7], help="peak season: fallow = bare in all")
    ap.add_argument("--detail-resolution", type=int, default=8)
    ap.add_argument("--detail-min-fallow", type=float, default=0.0,
                    help="detail layer keeps only cells with at least this black-fallow share in some year")
    ap.add_argument("--version", default=date.today().isoformat())
    ap.add_argument("--note", action="append", default=[], help="extra validation note (Russian), e.g. threshold calibration")
    ap.add_argument("--note-en", action="append", default=[], help="the same note in English")
    ap.add_argument("--smap", type=Path, default=None,
                    help="Furrow data/raw/<region>.smap.json: disaggregate root-zone moisture by NDMI in the latest year")
    ap.add_argument("--smap-max-days", type=int, default=5)
    args = ap.parse_args()

    grid = [c["h3"] for c in json.loads(args.grid.read_text(encoding="utf-8"))["cells"]]
    ring = [p for c in grid for p in h3.cell_to_boundary(c)]
    extent = (min(p[0] for p in ring), min(p[1] for p in ring), max(p[0] for p in ring), max(p[1] for p in ring))
    first = next((args.hls / str(args.years[0])).glob("*.Fmask.tif"))
    with rasterio.open(first) as ds:
        crs = ds.crs.to_string()
    affine, shape = build_grid(extent, crs)
    print(f"grid {shape} in {crs}, {len(grid)} region cells", flush=True)

    rows = np.repeat(np.arange(shape[0], dtype=np.int32), shape[1])
    cols = np.tile(np.arange(shape[1], dtype=np.int32), shape[0])
    xs, ys = xy(affine, rows, cols, offset="center")
    lon, lat = transform(crs, "EPSG:4326", xs, ys)
    del rows, cols, xs, ys
    crop = cropland_on_grid(args.worldcover, affine, shape, crs).ravel()
    sel = np.flatnonzero(crop)
    coarse = np.array([h3.latlng_to_cell(lat[i], lon[i], 5) for i in sel])
    inside = np.isin(coarse, grid)
    sel, coarse = sel[inside], coarse[inside]
    fine = np.array([h3.latlng_to_cell(lat[i], lon[i], args.detail_resolution) for i in sel])
    print(f"cropland pixels in the grid: {sel.size} of {crop.size}", flush=True)

    status: dict[int, np.ndarray] = {}
    fallow: dict[int, np.ndarray] = {}
    low_obs: dict[int, list[int]] = {}
    dates_used: dict[int, int] = {}
    hls_days: list[date] = []
    ndmi_days: list[np.ndarray] = []
    for y in args.years:
        found = scenes(args.hls / str(y), args.months)
        dates_used[y] = len(found)
        n_obs = np.zeros((len(args.months), sel.size), np.int16)
        n_bare = np.zeros_like(n_obs)
        for day, bands in found.items():
            i = args.months.index(day.month)
            red = mosaic(bands["B04"], affine, shape, -9999).ravel()[sel]
            nir = mosaic(bands["B05"], affine, shape, -9999).ravel()[sel]
            fm = mosaic(bands["Fmask"], affine, shape, 255).ravel()[sel]
            accumulate_month(n_obs[i], n_bare[i], red, nir, fm, args.bare_ndvi, [1, 2, 3])
            if args.smap is not None and y == args.years[-1] and "B06" in bands:
                swir1 = mosaic(bands["B06"], affine, shape, -9999).ravel()[sel]
                hls_days.append(day)
                ndmi_days.append(clear_ndmi(nir, swir1, fm, [1, 2, 3]))
        status[y] = month_status(n_obs, n_bare)
        # a month counts as low-observation when most cropland pixels have no clear date in it
        low_obs[y] = [m for i, m in enumerate(args.months) if float((n_obs[i] > 0).mean()) < 0.5]
        rows_peak = [args.months.index(m) for m in args.fallow_months]
        skip = [k for k, m in enumerate(args.fallow_months) if m in low_obs[y]]
        fallow[y] = black_fallow(status[y][rows_peak], skip, args.min_months)
        print(f"{y}: {len(found)} dates, low-obs months {low_obs[y]}, black fallow {fallow[y].mean():.3f}", flush=True)

    years = list(args.years)
    cells = aggregate(coarse, years, fallow, status, args.months, low_obs, 1000)
    detail = aggregate(fine, years, fallow, status, args.months, low_obs, 200, args.detail_min_fallow)
    grid_set = set(grid)  # a fine cell's H3 parent can fall outside the grid at its edge; Furrow would drop it
    detail = {k: v for k, v in detail.items() if h3.cell_to_parent(k, 5) in grid_set}
    sm_ru: list[str] = []
    sm_en: list[str] = []
    sm_report: dict[str, Any] | None = None
    if args.smap is not None:
        if not ndmi_days:
            raise ValueError(f"{years[-1]}: no HLS scenes with B06 for --smap")
        sm_ru, sm_en, sm_report = add_soil_moisture(
            detail, ndmi_days, hls_days, coarse, fine, args.smap, years[-1], args.months, args.smap_max_days,
        )
    n_fallow_years = np.stack([fallow[y] for y in years]).sum(axis=0)
    hist = {k: round(float((n_fallow_years == k).mean()), 4) for k in range(len(years) + 1)}

    span = f"{years[0]}–{years[-1]}"
    months_txt = ", ".join(f"{m:02d}" for m in args.fallow_months)
    ru = "; ".join([
        "культуры не классифицировались: для этого региона нет меток (CDL есть только в США)",
        "пашня — ESA WorldCover 2021, класс Cropland, 10 м → сетка 30 м по преобладающему классу",
        f"HLS L30 {span}, месяцы {args.months[0]:02d}–{args.months[-1]:02d}, даты по годам: "
        + ", ".join(f"{y} — {n}" for y, n in dates_used.items()),
        f"месяц голый, если NDVI < {args.bare_ndvi} на половине чистых дат месяца или больше (Fmask биты 1–3 — облака)",
        (f"fallow — чёрный пар: пиксель пашни голый во все месяцы пика сезона ({months_txt}) с наблюдениями, "
        f"их не меньше {args.min_months}; в пик сезона любая культура зелёная"),
        "cropped — пашня, зелёная в пик сезона; какая культура — неизвестно",
        f"доли пашни по числу лет чёрного пара за {span}: " + ", ".join(f"{k} лет — {v:.0%}" for k, v in hist.items()),
        *([f"detail — только очаги голой земли: ячейки H3 res {args.detail_resolution}, где чёрный пар хотя бы в один год занимал не меньше {args.detail_min_fallow:.0%} пашни; цвет: fallow — пар ≥ 50% пашни ячейки хотя бы в один год, fallow_part — 20–50% (доля — максимум по годам)"] if args.detail_min_fallow > 0 else []),
        *args.note,
        "проверки по наземным данным нет: это правило по снимкам, не обученная модель",
        *sm_ru,
    ])
    en = "; ".join([
        "crops were not classified: no labels exist for this region (CDL covers the US only)",
        "cropland from ESA WorldCover 2021 (class Cropland, 10 m, mode to 30 m)",
        f"HLS L30 {span}, months {args.months[0]:02d}–{args.months[-1]:02d}",
        f"a month counts as bare when NDVI < {args.bare_ndvi} on at least half of its clear images",
        (f"fallow = black fallow: a cropland pixel bare in every observed peak-season month ({months_txt}, at least "
        f"{args.min_months}); every crop is green then"),
        "cropped = cropland green in the peak season; which crop is unknown",
        f"share of cropland by years of black fallow in {span}: " + ", ".join(f"{k}: {v:.0%}" for k, v in hist.items()),
        *([f"detail shows bare-land hotspots only: H3 res {args.detail_resolution} cells where black fallow covered at least {args.detail_min_fallow:.0%} of cropland in some year; colour: fallow = black fallow ≥ 50% of the cell's cropland in at least one year, fallow_part = 20–50% (share = the maximum over years)"] if args.detail_min_fallow > 0 else []),
        *args.note_en,
        "no ground check: a rule on satellite images, not a trained model",
        *sm_en,
    ])
    layer = FurrowLayer(model="fs_ml bare-season rule (HLS L30 NDVI, ESA WorldCover cropland)", version=args.version,
                        validation=ru, validation_en=en, cells=dict(cells),
                        detail=DetailLayer(h3_resolution=args.detail_resolution, cells=dict(detail)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(layer.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")
    report: dict[str, Any] = {
        "grid_shape": list(shape), "crs": crs, "cropland_pixels": int(sel.size), "dates_used": dates_used,
        "low_obs_months": low_obs, "black_fallow_share": {y: round(float(fallow[y].mean()), 4) for y in years},
        "fallow_years_hist": hist, "cells": len(cells), "detail_cells": len(detail),
    }
    if sm_report is not None:
        report["soil_moisture"] = sm_report
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
