"""Lag months: NDVI below the same crop's quantile, green phase only (fs_ml.lag)."""

from datetime import date

import numpy as np

from fs_ml.lag import lag_status


def test_lag_status_flags_low_pixels_in_green_phase_only() -> None:
    rng = np.random.default_rng(0)
    n = 1000
    crop = np.array(["corn"] * n)
    green = rng.uniform(0.6, 0.8, n).astype(np.float32)
    green[:50] = 0.3                                   # 50 lagging corn pixels
    bare = rng.uniform(0.1, 0.2, n).astype(np.float32)  # May: corn not yet green -> not judged
    ndvi = np.stack([bare, green, green], axis=1)
    dates = [date(2021, 5, 1), date(2021, 7, 1), date(2021, 7, 15)]
    st = lag_status(ndvi, crop, dates, ["corn"], min_pixels=100)
    assert (st[5] == -1).all()
    assert (st[7][:50] == 1).all() and (st[7][50:] == 1).mean() < 0.3


def test_lag_status_ignores_other_groups_and_small_samples() -> None:
    ndvi = np.full((20, 1), 0.7, np.float32)
    crop = np.array(["fallow_idle"] * 10 + ["corn"] * 10)
    st = lag_status(ndvi, crop, [date(2021, 7, 1)], ["corn"], min_pixels=50)
    assert (st[7] == -1).all()                          # 10 corn pixels < min_pixels
