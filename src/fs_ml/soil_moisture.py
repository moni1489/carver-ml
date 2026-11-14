"""SMAP root-zone moisture spread over the fields by NDMI (disaggregation, not a prediction).

SMAP L4 gives one root-zone value per ~9 km H3 res 5 cell. On every HLS date we take the nearest SMAP date
(within `max_days`) and spread the cell's value over its 30 m pixels by their NDMI anomaly:

    sm_px = SMAP_cell * (1 + k * (NDMI_px - mean NDMI_cell))

so the cell mean stays exactly SMAP. k is one number for the whole season: the largest that keeps every pixel in
the physical range [SM_MIN, SM_MAX]. Cloudy pixels stay NaN. Per detail cell we report the median over dates of
its mean moisture and the 10-90th percentile over dates. Nothing here is trained and there is no ground truth.
"""

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import h3
import numpy as np
from pyproj import Transformer
from rasterio.transform import Affine, xy

SM_MIN, SM_MAX = 0.02, 0.5  # m3/m3


def ndmi(nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """(NIR - SWIR1) / (NIR + SWIR1); NaN where a band is NaN or not positive (cloud, fill, bad reflectance)."""
    ok = np.isfinite(nir) & np.isfinite(swir1) & (nir > 0) & (swir1 > 0)
    out = np.full(np.broadcast(nir, swir1).shape, np.nan, np.float32)
    out[ok] = (nir[ok] - swir1[ok]) / (nir[ok] + swir1[ok])
    return out


def nearest_date(target: date, dates: list[date], max_days: int) -> int | None:
    """Index of the date closest to target, None when it is more than max_days away."""
    if not dates:
        return None
    i = min(range(len(dates)), key=lambda j: abs((dates[j] - target).days))
    return i if abs((dates[i] - target).days) <= max_days else None


def k_limit(smap: float, index: np.ndarray, lo: float = SM_MIN, hi: float = SM_MAX) -> float:
    """Largest k >= 0 keeping smap * (1 + k * (index - mean)) inside [lo, hi] for every finite pixel."""
    if not lo <= smap <= hi:
        raise ValueError(f"SMAP {smap} outside [{lo}, {hi}]")
    dev = index - np.nanmean(index)
    k = np.inf
    dmax, dmin = float(np.nanmax(dev)), float(np.nanmin(dev))
    if dmax > 0:
        k = min(k, (hi / smap - 1) / dmax)
    if dmin < 0:
        k = min(k, (1 - lo / smap) / -dmin)
    return float(k)


def disaggregate(smap: float, index: np.ndarray, k: float) -> np.ndarray:
    """Spread one coarse value over pixels by their index anomaly; mean over finite pixels == smap, NaN stays NaN."""
    if not np.isfinite(index).any():
        return np.full(index.shape, np.nan, np.float64)
    out: np.ndarray = smap * (1 + k * (index.astype(np.float64) - np.nanmean(index)))
    return out


@dataclass
class Downscaled:
    k: float
    cells: dict[str, tuple[float, float, float, int]]  # fine cell -> (median, p10, p90, n dates)
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)  # coarse cell -> r, n, pixels
    dates_used: int = 0
    pixel_range: tuple[float, float] = (np.nan, np.nan)


def downscale(
    index: np.ndarray,
    coarse: np.ndarray,
    fine: np.ndarray,
    hls_dates: list[date],
    smap_dates: list[date],
    smap: dict[str, list[float | None]],
    max_days: int = 5,
    min_clear: float = 0.5,
    min_dates: int = 5,
    months: tuple[int, int] = (4, 10),
    fine_keep: set[str] | None = None,
) -> Downscaled:
    """index (T, N) per pixel and date; coarse / fine (N,) H3 keys of each pixel.

    A (date, coarse cell) is used when SMAP has a value within max_days and at least min_clear of its pixels are
    clear; a fine cell's date counts when at least min_clear of its pixels are clear. If `fine_keep` is set, only
    those fine cell keys are aggregated.
    """
    use_t = [t for t, d in enumerate(hls_dates) if months[0] <= d.month <= months[1]]
    groups = {str(c): np.flatnonzero(coarse == c) for c in np.unique(coarse) if str(c) in smap}
    pairs: list[tuple[int, str, float]] = []
    for t in use_t:
        j = nearest_date(hls_dates[t], smap_dates, max_days)
        if j is None:
            continue
        for c, idx in groups.items():
            value = smap[c][j]
            if value is None or np.isfinite(index[t, idx]).mean() < min_clear:
                continue
            pairs.append((t, c, float(value)))
    if not pairs:
        raise ValueError("no HLS date has a SMAP value and enough clear pixels")
    k = min(k_limit(v, index[t, groups[c]]) for t, c, v in pairs)

    sm = np.full(index.shape, np.nan, np.float32)
    for t, c, v in pairs:
        sm[t, groups[c]] = disaggregate(v, index[t, groups[c]], k)

    out = Downscaled(k=round(k, 4), cells={}, dates_used=len({t for t, _, _ in pairs}))
    finite = sm[np.isfinite(sm)]
    out.pixel_range = (round(float(finite.min()), 4), round(float(finite.max()), 4))
    for c, idx in groups.items():
        sel = [(v, float(np.nanmean(index[t, idx]))) for t, cc, v in pairs if cc == c]
        r = float(np.corrcoef([a for a, _ in sel], [b for _, b in sel])[0, 1]) if len(sel) > 2 else None
        out.checks[c] = {"r": None if r is None else round(r, 2), "n": len(sel), "pixels": int(idx.size)}
    used = sorted({t for t, _, _ in pairs})
    order = np.argsort(fine, kind="stable")
    sf = fine[order]
    starts = np.flatnonzero(np.r_[True, sf[1:] != sf[:-1]])
    ends = np.r_[starts[1:], sf.size]
    for a, b in zip(starts, ends, strict=True):
        f = str(sf[a])
        if fine_keep is not None and f not in fine_keep:
            continue
        idx = order[a:b]
        daily = [float(np.nanmean(sm[t, idx])) for t in used if np.isfinite(sm[t, idx]).mean() >= min_clear]
        if len(daily) < min_dates:
            continue
        med, p10, p90 = np.percentile(daily, [50, 10, 90])
        out.cells[f] = (round(float(med), 3), round(float(p10), 3), round(float(p90), 3), len(daily))
    return out


def load_inputs(
    cube: Path, meta: Path, smap_path: Path, in_bbox: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[date], list[date], dict[str, list[float | None]]]:
    """HLS cube (npz x: T, C, H, W with B05 at 3, B06 at 4) -> NDMI (T, N) of in-bbox pixels and their lon/lat."""
    m = json.loads(meta.read_text(encoding="utf-8"))
    z = np.load(cube)
    channels = list(m["channels"])
    x = z["x"]
    mask = z["in_bbox"] if in_bbox else np.ones(x.shape[2:], bool)
    rows, cols = np.nonzero(mask)
    index = ndmi(x[:, channels.index("B05")][:, rows, cols], x[:, channels.index("B06")][:, rows, cols])
    xs, ys = xy(Affine(*m["transform"][:6]), rows, cols, offset="center")
    lon, lat = Transformer.from_crs(m["crs"], "EPSG:4326", always_xy=True).transform(xs, ys)
    s = json.loads(smap_path.read_text(encoding="utf-8"))
    smap_dates = [date.fromisoformat(d[:10]) for d in s["dates"]]
    hls_dates = [date.fromisoformat(d) for d in m["dates"]]
    return index, np.asarray(lon), np.asarray(lat), hls_dates, smap_dates, s["cells"]


def run(cube: Path, meta: Path, smap_path: Path, fine_resolution: int, max_days: int = 5) -> Downscaled:
    index, lon, lat, hls_dates, smap_dates, smap = load_inputs(cube, meta, smap_path)
    coarse = np.array([h3.latlng_to_cell(a, b, 5) for a, b in zip(lat, lon, strict=True)])
    fine = np.array([h3.latlng_to_cell(a, b, fine_resolution) for a, b in zip(lat, lon, strict=True)])
    return downscale(index, coarse, fine, hls_dates, smap_dates, smap, max_days=max_days)
