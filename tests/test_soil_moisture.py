"""SMAP root-zone moisture spread by NDMI (fs_ml.soil_moisture)."""

from datetime import date

import numpy as np
import pytest

from fs_ml.soil_moisture import (
    SM_MAX,
    SM_MIN,
    disaggregate,
    downscale,
    k_limit,
    ndmi,
    nearest_date,
)


def test_disaggregate_keeps_mean_and_nan() -> None:
    index = np.array([0.1, np.nan, 0.3, -0.2, np.nan, 0.5])
    sm = disaggregate(0.18, index, 0.8)
    assert np.isnan(sm[[1, 4]]).all()                        # clouds are not filled
    assert np.isfinite(sm[[0, 2, 3, 5]]).all()
    assert np.nanmean(sm) == pytest.approx(0.18)              # cell mean == SMAP
    assert sm[5] > sm[2] > sm[0] > sm[3]                      # wetter index -> more moisture
    assert np.isnan(disaggregate(0.18, np.full(3, np.nan), 0.8)).all()


def test_k_limit_keeps_range() -> None:
    index = np.array([-0.4, 0.0, 0.1, 0.9])
    k = k_limit(0.2, index)
    sm = disaggregate(0.2, index, k)
    assert sm.min() >= SM_MIN - 1e-9 and sm.max() <= SM_MAX + 1e-9
    assert sm.min() == pytest.approx(SM_MIN) or sm.max() == pytest.approx(SM_MAX)  # the largest such k
    with pytest.raises(ValueError):
        k_limit(0.6, index)


def test_ndmi_and_nearest_date() -> None:
    out = ndmi(np.array([3000.0, np.nan, 2000.0, -5.0]), np.array([1000.0, 1000.0, 2000.0, 100.0]))
    assert out[0] == pytest.approx(0.5) and out[2] == 0 and np.isnan(out[[1, 3]]).all()
    days = [date(2021, 4, 1), date(2021, 4, 11)]
    assert nearest_date(date(2021, 4, 9), days, 5) == 1
    assert nearest_date(date(2021, 4, 20), days, 5) is None


def test_downscale_cells_and_check() -> None:
    rng = np.random.default_rng(0)
    t, n = 6, 40
    hls = [date(2021, 5, 1 + 5 * i) for i in range(t)]
    smap_dates = hls
    index = rng.uniform(-0.1, 0.4, (t, n))
    index[0, :5] = np.nan
    coarse = np.array(["c"] * n)
    fine = np.array(["a"] * 20 + ["b"] * 20)
    vals: list[float | None] = [0.15, 0.16, 0.2, 0.18, 0.17, 0.16]
    d = downscale(index, coarse, fine, hls, smap_dates, {"c": vals}, min_dates=3)
    assert d.dates_used == t and d.checks["c"]["n"] == t and d.k > 0
    assert set(d.cells) == {"a", "b"}
    med, p10, p90, nd = d.cells["a"]
    assert p10 <= med <= p90 and nd == t
    assert SM_MIN <= d.pixel_range[0] and d.pixel_range[1] <= SM_MAX
    d_sub = downscale(index, coarse, fine, hls, smap_dates, {"c": vals}, min_dates=3, fine_keep={"a"})
    assert set(d_sub.cells) == {"a"} and d_sub.cells["a"] == d.cells["a"]
