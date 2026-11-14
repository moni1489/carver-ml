"""Label-free bare-season rule (fs_ml.region_bare)."""

import numpy as np
import pytest

from fs_ml.region_bare import (
    accumulate_month,
    aggregate,
    black_fallow,
    clear_ndmi,
    composition,
    month_status,
)


def test_month_status_and_black_fallow() -> None:
    n_obs = np.array([[2, 0, 1], [1, 1, 2], [2, 2, 0], [1, 1, 1]], np.int16)  # 4 months x 3 pixels
    n_bare = np.array([[1, 0, 1], [1, 0, 2], [2, 0, 0], [1, 0, 1]], np.int16)
    st = month_status(n_obs, n_bare)
    assert st[:, 0].tolist() == [1, 1, 1, 1]          # half bare counts as bare
    assert st[:, 1].tolist() == [-1, 0, 0, 0]
    bf = black_fallow(st, skip=[], min_months=3)
    assert bf.tolist() == [True, False, True]          # pixel 2: 3 observed months, all bare
    assert black_fallow(st, skip=[0], min_months=3).tolist() == [True, False, False]


def test_accumulate_month_masks_clouds_and_fill() -> None:
    n_obs = np.zeros(4, np.int16)
    n_bare = np.zeros(4, np.int16)
    red = np.array([1000, 1000, 1000, -9999], np.int16)
    nir = np.array([1100, 3000, 3000, 3000], np.int16)
    fmask = np.array([0, 0, 0b10, 0], np.uint8)       # pixel 2 cloudy (bit 1)
    accumulate_month(n_obs, n_bare, red, nir, fmask, 0.2, [1, 2, 3])
    assert n_obs.tolist() == [1, 1, 0, 0] and n_bare.tolist() == [1, 0, 0, 0]


def test_composition() -> None:
    assert composition(0.23) == ("cropped 77%, fallow 23%", "cropped", 0.77)


def test_aggregate_hotspot_colour() -> None:
    keys = np.array(["a"] * 4 + ["b"] * 4)
    f19 = np.array([1, 1, 1, 0, 1, 0, 0, 0], bool)    # a: 75% fallow, b: 25%
    f20 = np.zeros(8, bool)
    st = np.zeros((1, 8), np.int8)
    cells = aggregate(keys, [2019, 2020], {2019: f19, 2020: f20}, {2019: st, 2020: st}, [6],
                      {2019: [], 2020: []}, 1, min_fallow=0.2)
    assert (cells["a"].crop_top, cells["a"].crop_top_share) == ("fallow", 0.75)
    assert (cells["b"].crop_top, cells["b"].crop_top_share) == ("fallow_part", 0.25)


def test_clear_ndmi_masks_clouds_and_fill() -> None:
    nir = np.array([3000, 3000, -9999, 1000], np.int16)
    swir1 = np.array([1000, 1000, 1000, 0], np.int16)
    fmask = np.array([0, 0b100, 0, 0], np.uint8)       # pixel 1 cloud-adjacent (bit 2)
    out = clear_ndmi(nir, swir1, fmask, [1, 2, 3])
    assert out[0] == pytest.approx(0.5)
    assert np.isnan(out[1:]).all()
