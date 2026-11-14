"""Months when a field lagged behind the same crop nearby: a relative NDVI signal, not a stress diagnosis.

For every crop group and kept date, NDVI of the pixels predicted as that crop forms a reference distribution. A
pixel lags on that date when its NDVI is below the crop's `quantile` there and at least `min_deficit` below its
median. Dates are only judged while the crop is in its green phase (median NDVI >= min_green, at least min_pixels clear pixels), so bare seasons are not "lag".
A month lags for a pixel when it lagged on at least half of its judged dates in that month.
"""

from datetime import date

import numpy as np
import pandas as pd

LAG, OK, UNKNOWN = 1, 0, -1


def lag_status(
    ndvi: np.ndarray,
    crop: np.ndarray,
    dates: list[date],
    groups: list[str],
    quantile: float = 0.10,
    min_deficit: float = 0.15,
    min_green: float = 0.4,
    min_pixels: int = 500,
) -> dict[int, np.ndarray]:
    """{month: (N,) int8} with 1 lag, 0 not lagging, -1 not judged (other crops, no green-phase clear date)."""
    n = ndvi.shape[0]
    lag_count: dict[int, np.ndarray] = {}
    judged: dict[int, np.ndarray] = {}
    for t, day in enumerate(dates):
        lag_count.setdefault(day.month, np.zeros(n, np.int16))
        judged.setdefault(day.month, np.zeros(n, np.int16))
        col = ndvi[:, t]
        for g in groups:
            mask = (crop == g) & ~np.isnan(col)
            if mask.sum() < min_pixels:
                continue
            vals = col[mask]
            if float(np.median(vals)) < min_green:
                continue
            med = float(np.median(vals))
            thr = min(float(np.quantile(vals, quantile)), med - min_deficit)
            judged[day.month][mask] += 1
            lag_count[day.month][mask] += vals < thr
    out: dict[int, np.ndarray] = {}
    for month in sorted(judged):
        status = np.full(n, UNKNOWN, np.int8)
        seen = judged[month] > 0
        status[seen] = np.where(2 * lag_count[month][seen] >= judged[month][seen], LAG, OK)
        out[month] = status
    return out


def add_lag_columns(pixels: pd.DataFrame, table: pd.DataFrame, groups: list[str], **kwargs: float) -> pd.DataFrame:
    """Join NDVI per kept date (pixels_2021.parquet: row, col, ndvi_YYYY-MM-DD) to the pixel summary and add
    lag_status_MM columns judged on the predicted crop (dominant_crop)."""
    cols = sorted(c for c in table.columns if c.startswith("ndvi_"))
    dates = [date.fromisoformat(c.removeprefix("ndvi_")) for c in cols]
    merged = pixels[["row", "col", "dominant_crop"]].merge(table[["row", "col", *cols]], on=["row", "col"], how="left")
    status = lag_status(merged[cols].to_numpy(np.float32), merged["dominant_crop"].to_numpy(), dates, groups,
                        **kwargs)  # type: ignore[arg-type]
    out = pixels.copy()
    for month, s in status.items():
        out[f"lag_status_{month:02d}"] = s
    return out
