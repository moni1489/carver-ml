"""Bare-month rule, low-observation months, previous-year CDL groups and cell `years` (synthetic data, no D:)."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fs_ml.config import CdlSource, HistoryCdl
from fs_ml.features import Grid, PixelStack
from fs_ml.furrow import cell_bare_months, cell_years, compose
from fs_ml.inference import (
    BARE,
    NOT_BARE,
    UNKNOWN,
    bare_month_status,
    history_groups,
    longest_bare_run,
    low_obs_months,
    pixel_summary,
    season_months,
)
from tests.test_pipeline import (
    NAMES,
    UTM12,
    _cell_and_bbox,
    _grid,
    _pixels,
    _space,
    _write_tif,
)

nan = np.nan
DATES = [date(2021, 4, 5), date(2021, 4, 14), date(2021, 5, 7), date(2021, 7, 26)]
MONTHS = [4, 5, 6, 7]


def test_bare_month_rule_half_of_clear_dates_and_unknown_without_dates() -> None:
    ndvi = np.array(
        [
            [0.10, 0.50, 0.10, nan],  # Apr 1 of 2 bare -> bare (half counts); May bare; Jul no clear date
            [0.50, nan, nan, 0.05],  # Apr 0 of 1 -> not bare; May unknown; Jul bare
            [0.20, 0.20, 0.19, 0.30],  # 0.2 is not below 0.2 -> not bare; May 0.19 bare
        ]
    )
    status, n_obs = bare_month_status(ndvi, DATES, MONTHS, 0.2)
    assert status[0].tolist() == [BARE, BARE, UNKNOWN, UNKNOWN]  # June has no dates at all -> unknown
    assert status[1].tolist() == [NOT_BARE, UNKNOWN, UNKNOWN, BARE]
    assert status[2].tolist() == [NOT_BARE, BARE, UNKNOWN, NOT_BARE]
    assert n_obs.tolist() == [[2, 1, 0, 0], [1, 0, 0, 1], [2, 1, 0, 1]]


def test_longest_bare_run_unknown_breaks_run() -> None:
    status = np.array([[1, 1, -1, 1, 1, 1], [1, 0, 1, 1, 0, 0], [-1, -1, -1, -1, -1, -1]], dtype=np.int8)
    assert longest_bare_run(status).tolist() == [3, 2, 0]


def test_low_obs_months_and_season_months() -> None:
    assert low_obs_months(DATES, MONTHS) == {5: ["2021-05-07"], 6: [], 7: ["2021-07-26"]}
    assert season_months(("2021-04-01", "2021-10-31")) == [4, 5, 6, 7, 8, 9, 10]
    with pytest.raises(ValueError):
        season_months(("2021-10-01", "2022-03-31"))


def _stack(ndvi: np.ndarray) -> PixelStack:
    n = ndvi.shape[0]
    return PixelStack(
        grid=_grid((1, n)),
        dates=DATES,
        ndvi=ndvi.astype(np.float32),
        n_clear=(~np.isnan(ndvi)).sum(axis=1).astype(np.int16),
        in_bbox=np.ones(n, dtype=bool),
        date_clear_share=dict.fromkeys(DATES, 1.0),
        dropped_dates=[],
        cdl=np.zeros(n, dtype=np.uint8),
        labels=np.zeros(n, dtype=np.int16),
    )


def test_pixel_summary_low_obs_month_kept_in_data_but_not_in_bare_months() -> None:
    # Apr bare, May bare, Jun unknown, Jul bare on its only date
    stack = _stack(np.array([[0.1, 0.1, 0.1, 0.1]]))
    table = pixel_summary(
        stack, np.zeros(1, dtype=np.int16), _space(), 0.2, {}, [], months=MONTHS, low_obs=[7]
    )
    assert list(table.loc[0, "bare_months"]) == [4, 5]
    assert int(table.loc[0, "bare_status_07"]) == BARE and int(table.loc[0, "n_obs_07"]) == 1
    assert int(table.loc[0, "longest_bare_run"]) == 2  # June unknown breaks; July skipped, not counted
    full = pixel_summary(stack, np.zeros(1, dtype=np.int16), _space(), 0.2, {}, [], months=MONTHS)
    assert list(full.loc[0, "bare_months"]) == [4, 5, 7]


def test_history_groups_use_label_lut_and_nodata(tmp_path: Path) -> None:
    grid: Grid = _grid((1, 3))
    path = tmp_path / "2019.tif"
    _write_tif(path, np.array([[36, 0, 152]], dtype=np.uint8), UTM12, grid.transform)
    space = _space()
    space.lut[36] = NAMES.index("alfalfa")
    space.lut[152] = NAMES.index("other")
    space.lut[0] = -1
    item = HistoryCdl(
        year=2019, path=path, source=CdlSource(zip_path=Path("x.zip"), member="x.tif", crs="EPSG:5070", margin_px=0)
    )
    groups = history_groups(grid, [item], space)
    assert groups[2019].tolist() == ["alfalfa", "nodata", "other"]


def test_cell_bare_months_half_of_known_pixels_and_skip() -> None:
    frame = pd.DataFrame(
        {
            "bare_status_04": [1, 1, 0, -1],  # 2 of 3 known -> bare
            "bare_status_05": [1, 0, 0, 0],  # 1 of 4 -> not bare
            "bare_status_06": [-1, -1, -1, -1],  # nothing known -> not reported
            "bare_status_07": [1, 1, 1, 1],  # bare but low observations -> skipped
        }
    )
    assert cell_bare_months(frame) == [4, 7]
    assert cell_bare_months(frame, {7}) == [4]


def test_cell_years_sources_history_composition_and_prediction_fields() -> None:
    frame = pd.DataFrame(
        {
            "cdl_group_2019": ["alfalfa", "alfalfa", "fallow_idle", "nodata"],
            "cdl_group_2020": ["nodata"] * 4,
            "bare_obs_dates": [2, 4, 6, 9],
            "bare_status_04": [1, 1, 1, 0],
            "bare_status_07": [1, 1, 1, 1],
        }
    )
    keys = {"fallow_idle": "fallow"}
    years = cell_years(frame, 2021, "alfalfa 50%", keys, "other", {7})
    assert years["2019"].crop == "alfalfa 67%, fallow 33%" and years["2019"].source == "USDA CDL"
    assert years["2020"].crop is None and years["2020"].source == "USDA CDL"  # no data that year
    assert years["2021"].source == "model" and years["2021"].crop == "alfalfa 50%"
    assert years["2021"].bare_obs_dates == 5
    assert years["2021"].bare_months == [4] and years["2021"].low_obs_months == [7]


def test_compose_adds_years_only_with_prediction_year() -> None:
    cell, bbox = _cell_and_bbox()
    pixels = _pixels(["cotton", "cotton", "alfalfa"], bbox).assign(
        cdl_group_2019=["cotton", "corn", "alfalfa"], bare_obs_dates=[1, 2, 3], bare_status_04=[1, 1, 0]
    )
    keys = {"cotton": "cotton"}
    plain, _ = compose(pixels, bbox, 5, keys, "other", {})
    assert plain[cell].years is None
    with_years, _ = compose(pixels, bbox, 5, keys, "other", {}, 2021)
    years = with_years[cell].years
    assert years is not None and set(years) == {"2019", "2021"}
    assert years["2021"].bare_months == [4]
