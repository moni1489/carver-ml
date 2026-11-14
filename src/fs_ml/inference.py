"""Per-pixel summary for the season: predicted crop, bare-soil observations and months, previous-year CDL groups."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from rasterio.transform import xy
from rasterio.warp import transform as warp_transform

from fs_ml.config import HistoryCdl
from fs_ml.features import Grid, LabelSpace, PixelStack, cdl_on_grid

NODATA_GROUP = "nodata"
# bare_status_MM values
BARE, NOT_BARE, UNKNOWN = 1, 0, -1


def _export_names(space: LabelSpace, export_merge: dict[str, list[str]]) -> np.ndarray:
    """Map each label (plus 'unpredicted') to its export name; members of a merge group share one name."""
    target = {member: group for group, members in export_merge.items() for member in members}
    names = space.names + ["unpredicted"]
    return np.array([target.get(name, name) for name in names])


def season_months(temporal: tuple[str, str]) -> list[int]:
    """Calendar months covered by ingest.temporal (one season within a year), e.g. 4..10."""
    start, end = date.fromisoformat(temporal[0]), date.fromisoformat(temporal[1])
    if start.year != end.year or start > end:
        raise ValueError(f"temporal должен быть одним сезоном внутри года: {temporal}")
    return list(range(start.month, end.month + 1))


def bare_month_status(
    ndvi: np.ndarray, dates: list[date], months: list[int], bare_ndvi: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per pixel and month: status (1 bare, 0 not bare, -1 unknown) and number of clear dates (n_obs).

    A month is bare if NDVI < bare_ndvi on at least half of the pixel's clear dates in that month; a month with
    no clear date is unknown (neither bare nor green).
    """
    n = ndvi.shape[0]
    status = np.full((n, len(months)), UNKNOWN, dtype=np.int8)
    n_obs = np.zeros((n, len(months)), dtype=np.int16)
    clear = ~np.isnan(ndvi)
    bare = clear & (np.nan_to_num(ndvi, nan=np.inf) < bare_ndvi)
    date_months = np.array([d.month for d in dates])
    for j, month in enumerate(months):
        cols = date_months == month
        obs = clear[:, cols].sum(axis=1)
        n_bare = bare[:, cols].sum(axis=1)
        n_obs[:, j] = obs
        known = obs > 0
        status[known, j] = np.where(2 * n_bare[known] >= obs[known], BARE, NOT_BARE)
    return status, n_obs


def longest_bare_run(status: np.ndarray) -> np.ndarray:
    """Longest run of consecutive bare months per pixel; an unknown month breaks the run."""
    best = np.zeros(status.shape[0], dtype=np.int16)
    run = np.zeros(status.shape[0], dtype=np.int16)
    for j in range(status.shape[1]):
        run = np.where(status[:, j] == BARE, run + 1, 0).astype(np.int16)
        best = np.maximum(best, run)
    return best


def bare_month_lists(status: np.ndarray, months: list[int]) -> list[list[int]]:
    month_arr = np.array(months)
    return [month_arr[row == BARE].tolist() for row in status]


def low_obs_months(dates: list[date], months: list[int]) -> dict[int, list[str]]:
    """Season months with at most one kept date (too few observations to call the month bare or green)."""
    out: dict[int, list[str]] = {}
    for month in months:
        days = [d.isoformat() for d in dates if d.month == month]
        if len(days) <= 1:
            out[month] = days
    return out


def history_groups(grid: Grid, history: list[HistoryCdl], space: LabelSpace) -> dict[int, np.ndarray]:
    """Group name per pixel for each previous-year CDL (same LUT as the labels; code 0 -> 'nodata')."""
    names = np.array(space.names + [NODATA_GROUP])
    out: dict[int, np.ndarray] = {}
    for item in sorted(history, key=lambda h: h.year):
        index = space.lut[cdl_on_grid(item.path, grid).ravel()]
        out[item.year] = names[np.where(index >= 0, index, len(space.names))]
    return out


def pixel_summary(
    stack: PixelStack,
    oof: np.ndarray,
    space: LabelSpace,
    bare_ndvi: float,
    export_merge: dict[str, list[str]],
    low_confidence_groups: list[str],
    months: list[int] | None = None,
    previous: dict[int, np.ndarray] | None = None,
    low_obs: list[int] | None = None,
) -> pd.DataFrame:
    """Per-pixel table. Months in `low_obs` (at most one kept date) keep their n_obs_MM / bare_status_MM columns
    but are not counted as full months: they are left out of bare_months, and longest_bare_run skips them
    (neither counts nor breaks a run)."""
    height, width = stack.grid.shape
    rows = np.repeat(np.arange(height, dtype=np.int32), width)
    cols = np.tile(np.arange(width, dtype=np.int32), height)
    xs, ys = xy(stack.grid.transform, rows, cols, offset="center")
    lon, lat = warp_transform(stack.grid.crs, "EPSG:4326", xs, ys)

    valid = ~np.isnan(stack.ndvi)
    bare_obs = np.sum((stack.ndvi < bare_ndvi) & valid, axis=1).astype(np.int16)
    valid_obs = np.sum(valid, axis=1).astype(np.int16)
    index = np.where(oof >= 0, oof, len(space.names))
    crop = np.array(space.names + ["unpredicted"])[index]
    crop_export = _export_names(space, export_merge)[index]
    low_conf = np.isin(crop, low_confidence_groups)
    columns: dict[str, object] = {
        "row": rows,
        "col": cols,
        "lon": np.asarray(lon, dtype=np.float64),
        "lat": np.asarray(lat, dtype=np.float64),
        "in_bbox": stack.in_bbox,
        "dominant_crop": crop,
        "dominant_crop_export": crop_export,
        "low_confidence": low_conf,
        "bare_obs_dates": bare_obs,
        "valid_obs_dates": valid_obs,
    }
    if months:
        status, n_obs = bare_month_status(stack.ndvi, stack.dates, months, bare_ndvi)
        full = [j for j, m in enumerate(months) if m not in set(low_obs or [])]
        columns["bare_months"] = bare_month_lists(status[:, full], [months[j] for j in full])
        columns["longest_bare_run"] = longest_bare_run(status[:, full])
        for j, month in enumerate(months):
            columns[f"n_obs_{month:02d}"] = n_obs[:, j]
            columns[f"bare_status_{month:02d}"] = status[:, j]
    for year, groups in sorted((previous or {}).items()):
        columns[f"cdl_group_{year}"] = groups
    return pd.DataFrame(columns)


def write_pixel_summary(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
