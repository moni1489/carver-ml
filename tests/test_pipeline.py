"""Unit tests on small synthetic arrays: Fmask, date selection, bbox, decision rule, H3 export, CDL grid and clip.

Nothing here reads D:; rasters are written to pytest's tmp_path.
"""

import zipfile
from datetime import date
from pathlib import Path
from typing import Any

import h3
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform
from rasterio.windows import Window

from fs_ml.cdl import clip_to_grid
from fs_ml.config import (
    BaselineConfig,
    CdlSource,
    CropsConfig,
    MLConfig,
    baseline_fingerprint,
)
from fs_ml.features import (
    Grid,
    LabelSpace,
    PixelStack,
    cdl_on_grid,
    feature_matrix,
    grid_mismatch,
    inside_bbox,
    min_clear_dates,
    ndvi_masked,
    select_dates,
    swir_masked,
)
from fs_ml.furrow import (
    aggregate,
    blocked_groups,
    check_baseline_matches,
    compose,
    composition_text,
    low_confidence_notes,
    validation_text,
)
from fs_ml.models import decide, decision_rule_text, usable_mask

NAMES = ["alfalfa", "cotton", "small_grains", "corn", "fallow_idle", "other"]
FALLOW = NAMES.index("fallow_idle")
UTM12 = CRS.from_epsg(32612)


def _baseline(**overrides: Any) -> BaselineConfig:
    raw: dict[str, Any] = {
        "cdl_path": "cdl.tif",
        "hls_dir": "hls",
        "output_dir": "out",
        "reports_dir": "reports",
        "nir_band": "B05",
        "red_band": "B04",
        "fmask_band": "Fmask",
        "cloud_bits": [1, 2, 3],
        "min_date_clear_share": 0.5,
        "min_clear_fraction": 0.5,
        "spatial_blocks": 3,
        "buffer_px": 0,
        "max_train_per_class": 10,
        "random_state": 0,
        "random_forest": {"n_estimators": 1, "n_jobs": 1, "class_weight": None},
    }
    raw.update(overrides)
    return BaselineConfig.model_validate(raw)


def _ml(**baseline_overrides: Any) -> MLConfig:
    return MLConfig.model_validate(
        {
            "ingest": {
                "short_name": "HLSL30",
                "version": "2.0",
                "bbox": [-112.0, 32.9, -111.9, 33.0],
                "temporal": ["2021-04-01", "2021-10-31"],
                "count": 10,
                "dates": ["2021-04-05", "2021-04-14"],
                "max_cloud_pct": 20,
                "max_download_mb": 100,
                "local_path": "hls",
            },
            "thresholds": {"max_cloud_pct": 20, "bare_soil_ndvi": 0.2},
            "baseline": _baseline(**baseline_overrides).model_dump(mode="json"),
        }
    )


def _crops() -> CropsConfig:
    return CropsConfig.model_validate(
        {
            "other_group": "other",
            "groups": {"alfalfa": [36], "cotton": [2], "small_grains": [21, 22], "corn": [1], "fallow_idle": [61]},
            "agricultural_groups": NAMES[:-1],
            "low_confidence_groups": ["fallow_idle"],
            "furrow_keys": {"cotton": "cotton", "fallow_idle": "fallow"},
        }
    )


def _space() -> LabelSpace:
    return LabelSpace(names=NAMES, agricultural=list(range(5)), lut=np.zeros(256, dtype=np.int16))


def _grid(shape: tuple[int, int] = (1, 1), x0: float = 400000.0, y0: float = 3650000.0) -> Grid:
    return Grid(
        window=Window(0, 0, shape[1], shape[0]),
        transform=from_origin(x0, y0, 30.0, 30.0),
        crs=UTM12,
        shape=shape,
        clipped_to_raster=False,
    )


def _stack(ndvi: np.ndarray, labels: np.ndarray, in_bbox: np.ndarray) -> PixelStack:
    n, t = ndvi.shape
    dates = [date(2021, 4, 1 + i) for i in range(t)]
    return PixelStack(
        grid=_grid((1, n)),
        dates=dates,
        ndvi=ndvi.astype(np.float32),
        n_clear=(~np.isnan(ndvi)).sum(axis=1).astype(np.int16),
        in_bbox=in_bbox,
        date_clear_share=dict.fromkeys(dates, 1.0),
        dropped_dates=[],
        cdl=np.zeros(n, dtype=np.uint8),
        labels=labels.astype(np.int16),
    )


# --- Fmask and NDVI -----------------------------------------------------------------------------------------


def test_ndvi_masked_drops_only_bits_1_to_3_and_fill() -> None:
    #           clear cirrus(b0) cloud(b1) adj(b2) shadow(b3) snow(b4) water(b5) fill
    fmask = np.array([0, 1, 2, 4, 8, 16, 32, 255], dtype=np.uint8)
    nir = np.full(fmask.shape, 3000, dtype=np.int16)
    red = np.full(fmask.shape, 1000, dtype=np.int16)
    out = ndvi_masked(nir, red, fmask, [1, 2, 3])
    expected_nan = [False, False, True, True, True, False, False, True]
    assert np.isnan(out).tolist() == expected_nan
    assert np.allclose(out[~np.isnan(out)], 0.5)


def test_ndvi_masked_treats_non_positive_bands_as_missing() -> None:
    nir = np.array([3000, 0, -9999, 3000], dtype=np.int16)
    red = np.array([1000, 1000, 1000, -9999], dtype=np.int16)
    out = ndvi_masked(nir, red, np.zeros(4, dtype=np.uint8), [1, 2, 3])
    assert np.isnan(out).tolist() == [False, True, True, True]


# --- date selection and bbox --------------------------------------------------------------------------------


def test_select_dates_uses_in_bbox_pixels_and_keeps_share_equal_to_threshold() -> None:
    nan = np.nan
    ndvi = np.array(
        [
            [0.1, 0.3, nan],
            [0.2, 0.3, nan],
            [0.3, nan, nan],
            [0.4, nan, nan],
            [0.5, 0.5, 0.6],  # outside bbox: must not count for any date
        ]
    )
    in_bbox = np.array([True, True, True, True, False])
    share, keep = select_dates(ndvi, in_bbox, 0.5)
    assert share.tolist() == [1.0, 0.5, 0.0]
    assert keep.tolist() == [True, True, False]


def test_select_dates_raises_without_pixels_or_dates() -> None:
    ndvi = np.array([[np.nan, np.nan]])
    with pytest.raises(RuntimeError):
        select_dates(ndvi, np.array([False]), 0.5)
    with pytest.raises(RuntimeError):
        select_dates(ndvi, np.array([True]), 0.5)


def test_min_clear_dates_rounds_up() -> None:
    assert min_clear_dates(0.5, 27) == 14
    assert min_clear_dates(0.5, 22) == 11


def test_inside_bbox_includes_edges_and_drops_outside() -> None:
    lon = np.array([-112.0, -111.95, -111.9, -112.01, -111.95])
    lat = np.array([32.9, 32.95, 33.0, 32.95, 33.01])
    assert inside_bbox(lon, lat, (-112.0, 32.9, -111.9, 33.0)).tolist() == [True, True, True, False, False]


def test_usable_mask_needs_label_bbox_and_enough_clear_dates() -> None:
    nan = np.nan
    ndvi = np.array([[0.1, 0.2], [0.1, nan], [nan, nan], [0.1, 0.2], [0.1, 0.2]])
    labels = np.array([0, 0, 0, -1, 0])
    in_bbox = np.array([True, True, True, True, False])
    usable = usable_mask(_stack(ndvi, labels, in_bbox), _baseline())
    # min_clear_fraction 0.5 of 2 dates -> 1 clear date needed
    assert usable.tolist() == [True, True, False, False, False]


# --- decision rule ------------------------------------------------------------------------------------------


def _proba(rows: list[dict[str, float]]) -> np.ndarray:
    out = np.zeros((len(rows), len(NAMES)), dtype=np.float32)
    for i, row in enumerate(rows):
        for name, value in row.items():
            out[i, NAMES.index(name)] = value
    return out


def test_decide_without_rules_is_argmax() -> None:
    proba = _proba([{"fallow_idle": 0.55, "other": 0.45}, {"cotton": 0.7, "corn": 0.3}])
    stack = _stack(np.full((2, 1), 0.5), np.zeros(2), np.ones(2, dtype=bool))
    cfg = _baseline()
    assert decide(proba, stack, _space(), cfg).tolist() == [FALLOW, NAMES.index("cotton")]
    assert decision_rule_text(cfg) == "argmax"


def test_decide_min_proba_moves_weak_fallow_to_next_class() -> None:
    proba = _proba(
        [
            {"fallow_idle": 0.55, "other": 0.30, "alfalfa": 0.15},  # below 0.6 -> other
            {"fallow_idle": 0.65, "other": 0.35},  # above 0.6 -> stays
            {"cotton": 0.50, "fallow_idle": 0.40, "corn": 0.10},  # fallow not top -> unchanged
        ]
    )
    stack = _stack(np.full((3, 1), 0.5), np.zeros(3), np.ones(3, dtype=bool))
    labels = decide(proba, stack, _space(), _baseline(min_proba={"fallow_idle": 0.6}))
    assert labels.tolist() == [NAMES.index("other"), FALLOW, NAMES.index("cotton")]


def test_decide_desert_mask_blocks_fallow_on_low_ndvi_only() -> None:
    nan = np.nan
    ndvi = np.array([[0.05, 0.1], [0.3, 0.6], [nan, nan]])
    proba = _proba([{"fallow_idle": 0.9, "other": 0.1}] * 3)
    stack = _stack(ndvi, np.zeros(3), np.ones(3, dtype=bool))
    cfg = _baseline(desert_max_ndvi=0.2, desert_excluded_groups=["fallow_idle"])
    # low NDVI -> other; green -> fallow kept; no clear date -> mask does not apply
    assert decide(proba, stack, _space(), cfg).tolist() == [NAMES.index("other"), FALLOW, FALLOW]


def test_decide_rejects_unknown_group() -> None:
    stack = _stack(np.full((1, 1), 0.5), np.zeros(1), np.ones(1, dtype=bool))
    with pytest.raises(ValueError):
        decide(_proba([{"other": 1.0}]), stack, _space(), _baseline(min_proba={"wheat": 0.5}))


# --- Furrow export ------------------------------------------------------------------------------------------


def _cell_and_bbox() -> tuple[str, tuple[float, float, float, float]]:
    cell = h3.latlng_to_cell(33.0, -111.95, 5)
    lat, lon = h3.cell_to_latlng(cell)
    return cell, (lon - 0.1, lat - 0.1, lon + 0.1, lat + 0.1)


def _pixels(groups: list[str], bbox: tuple[float, float, float, float], extra_outside: int = 0) -> pd.DataFrame:
    west, south, east, north = bbox
    lon_c, lat_c = (west + east) / 2, (south + north) / 2
    rows = [{"lon": lon_c + 0.001 * i, "lat": lat_c, "dominant_crop": g, "in_bbox": True} for i, g in enumerate(groups)]
    rows += [{"lon": lon_c, "lat": lat_c, "dominant_crop": "alfalfa", "in_bbox": False}] * extra_outside
    return pd.DataFrame(rows)


def test_aggregate_writes_majority_with_furrow_key_and_ignores_outside_pixels() -> None:
    cell, bbox = _cell_and_bbox()
    keys = {"cotton": "cotton", "fallow_idle": "fallow"}
    pixels = _pixels(["cotton", "cotton", "cotton", "corn"], bbox, extra_outside=10)
    cells, report = aggregate(pixels, bbox, 5, keys, NAMES, {})
    assert list(cells) == [cell]
    assert cells[cell].crop == "cotton"
    assert cells[cell].crop_pixel_share == 0.75
    assert report.written[cell]["n_pixels_in_bbox"] == 4
    assert cell not in report.centre_outside_bbox
    assert report.centre_outside_bbox  # neighbours overlap the bbox but their centres are outside


def test_aggregate_skips_group_without_key_and_blocked_group() -> None:
    cell, bbox = _cell_and_bbox()
    keys = {"cotton": "cotton", "fallow_idle": "fallow"}
    cells, report = aggregate(_pixels(["alfalfa", "alfalfa", "cotton"], bbox), bbox, 5, keys, NAMES, {})
    assert cells == {}
    assert "без ключа Furrow" in report.skipped_reasons[cell]
    assert "alfalfa" in report.groups_without_furrow_key

    blocked = {"fallow_idle": {"precision_bbox": 0.6, "min_precision": 0.7}}
    cells, report = aggregate(_pixels(["fallow_idle", "fallow_idle", "cotton"], bbox), bbox, 5, keys, NAMES, blocked)
    assert cells == {}
    assert "не выше порога" in report.skipped_reasons[cell]


def test_blocked_groups_uses_not_above_threshold() -> None:
    metrics = pd.DataFrame({"precision_bbox": [0.7, 0.71]}, index=pd.Index(["fallow_idle", "cotton"], name="group"))
    blocked = blocked_groups(metrics, ["fallow_idle", "cotton"], 0.7)
    assert list(blocked) == ["fallow_idle"]


def test_composition_text_orders_marks_low_confidence_and_hides_other() -> None:
    shares = {"fallow_idle": 0.32, "alfalfa": 0.202, "corn": 0.193, "other": 0.151, "cotton": 0.133, "small_grains": 0.001}
    keys = {"cotton": "cotton", "fallow_idle": "fallow"}
    text, crop_p = composition_text(shares, keys, "other", {"fallow_idle", "small_grains"})
    assert text == "alfalfa 20%, corn 19%, cotton 13%, fallow 32% (низкая достоверность)"
    assert crop_p == 0.202
    text, crop_p = composition_text({"fallow_idle": 0.6, "other": 0.4}, keys, "other", {"fallow_idle"})
    assert text == "fallow 60% (низкая достоверность)"
    assert crop_p is None
    assert composition_text({"other": 1.0}, keys, "other", set()) == ("", None)


def test_compose_writes_centre_inside_cell_with_composition() -> None:
    cell, bbox = _cell_and_bbox()
    keys = {"cotton": "cotton", "fallow_idle": "fallow"}
    pixels = _pixels(["fallow_idle", "fallow_idle", "alfalfa", "other"], bbox, extra_outside=10)
    blocked = {"fallow_idle": {"precision_bbox": 0.69, "min_precision": 0.7}}
    cells, report = compose(pixels, bbox, 5, keys, "other", blocked)
    assert list(cells) == [cell]
    assert cells[cell].crop == "alfalfa 25%, fallow 50% (низкая достоверность)"
    assert cells[cell].crop_p == 0.25
    # crop_top: largest non-other group even when low confidence, with its Furrow key and the flag
    assert (cells[cell].crop_top, cells[cell].crop_top_share, cells[cell].low_confidence) == ("fallow", 0.5, True)
    assert report.written[cell]["n_pixels_in_bbox"] == 4
    confident, _ = compose(pixels, bbox, 5, keys, "other", {})
    assert confident[cell].low_confidence is False
    only_other, report = compose(_pixels(["other"], bbox), bbox, 5, keys, "other", {})
    assert only_other == {}
    assert "только other" in report.skipped_reasons[cell]


def test_low_confidence_notes_cover_exported_and_blocked() -> None:
    metrics = pd.DataFrame(
        {"precision_bbox": [0.6], "recall_bbox": [0.8]}, index=pd.Index(["fallow_idle"], name="group")
    )
    crops = _crops()
    assert "низкая достоверность" in low_confidence_notes(crops, metrics, {})[0]
    blocked = {"fallow_idle": {"precision_bbox": 0.6, "min_precision": 0.7}}
    assert "не выше порога 0.7" in low_confidence_notes(crops, metrics, blocked)[0]
    no_low = crops.model_copy(update={"low_confidence_groups": []})
    assert low_confidence_notes(no_low, metrics, {}) == []


def test_check_baseline_matches_requires_same_config() -> None:
    ml, crops = _ml(), _crops()
    summary = {
        "config_sha256": baseline_fingerprint(ml, crops),
        "decision_rule": decision_rule_text(ml.baseline),
        "min_proba": {},
    }
    check_baseline_matches(summary, ml, crops)
    changed = _ml(min_proba={"fallow_idle": 0.6})
    with pytest.raises(RuntimeError, match="перезапусти baseline"):
        check_baseline_matches(summary, changed, crops)
    with pytest.raises(RuntimeError, match="перезапусти baseline"):
        check_baseline_matches({}, ml, crops)


def test_fingerprint_ignores_furrow_keys_but_not_baseline_settings() -> None:
    ml, crops = _ml(), _crops()
    base = baseline_fingerprint(ml, crops)
    assert baseline_fingerprint(ml, crops.model_copy(update={"furrow_keys": {}})) == base
    assert baseline_fingerprint(_ml(buffer_px=90), crops) != base


# --- CDL grid and clip --------------------------------------------------------------------------------------


def _write_tif(path: Path, data: np.ndarray, crs: CRS, affine: Any) -> None:
    profile = {
        "driver": "GTiff",
        "width": data.shape[1],
        "height": data.shape[0],
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": affine,
        "nodata": 0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def test_grid_mismatch_reports_each_difference() -> None:
    grid = _grid((4, 5))
    assert grid_mismatch(UTM12, grid.transform, (4, 5), grid) == []
    problems = grid_mismatch(CRS.from_epsg(5070), from_origin(400030.0, 3650000.0, 30.0, 30.0), (4, 6), grid)
    assert len(problems) == 3


def test_cdl_on_grid_reads_matching_clip_and_rejects_other_grid(tmp_path: Path) -> None:
    grid = _grid((4, 5))
    data = np.arange(1, 21, dtype=np.uint8).reshape(4, 5)
    good = tmp_path / "good.tif"
    _write_tif(good, data, UTM12, grid.transform)
    assert np.array_equal(cdl_on_grid(good, grid), data)

    shifted = tmp_path / "shifted.tif"
    _write_tif(shifted, data, UTM12, from_origin(400030.0, 3650000.0, 30.0, 30.0))
    with pytest.raises(RuntimeError, match="не на сетке HLS"):
        cdl_on_grid(shifted, grid)


def test_clip_to_grid_reads_window_from_zip_and_matches_full_nearest_reproject(tmp_path: Path) -> None:
    albers = CRS.from_epsg(5070)
    utm_x, utm_y = 405000.0, 3650000.0
    (ax,), (ay,) = transform(UTM12, albers, [utm_x], [utm_y])
    src_affine = from_origin(ax - 3000.0, ay + 3000.0, 30.0, 30.0)
    rng = np.random.default_rng(0)
    source = rng.integers(1, 255, size=(200, 200), dtype=np.uint8)
    tif = tmp_path / "cdl.tif"
    _write_tif(tif, source, albers, src_affine)
    archive = tmp_path / "cdl.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(tif, arcname="cdl.tif")

    grid = _grid((12, 10), x0=utm_x, y0=utm_y)
    out = tmp_path / "clip" / "2021.tif"
    result = clip_to_grid(CdlSource(zip_path=archive, member="cdl.tif", crs="EPSG:5070", margin_px=4), grid, out)

    expected = np.zeros(grid.shape, dtype=np.uint8)
    reproject(
        source=source,
        destination=expected,
        src_transform=src_affine,
        src_crs=albers,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        resampling=Resampling.nearest,
        src_nodata=0,
        dst_nodata=0,
    )
    clipped = cdl_on_grid(out, grid)  # also asserts the clip lies on the HLS grid
    assert np.array_equal(clipped, expected)
    assert result.nodata_share == 0.0
    assert result.source_window.width < source.shape[1]  # only a window was read


def test_clip_to_grid_rejects_wrong_source_crs(tmp_path: Path) -> None:
    tif = tmp_path / "cdl.tif"
    _write_tif(tif, np.ones((10, 10), dtype=np.uint8), UTM12, from_origin(400000.0, 3650300.0, 30.0, 30.0))
    archive = tmp_path / "cdl.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(tif, arcname="cdl.tif")
    source = CdlSource(zip_path=archive, member="cdl.tif", crs="EPSG:5070", margin_px=0)
    with pytest.raises(RuntimeError, match="CRS"):
        clip_to_grid(source, _grid((5, 5)), tmp_path / "out.tif")


# --- SWIR features ------------------------------------------------------------------------------------------


def test_swir_masked_uses_same_cloud_bits_and_drops_non_positive_bands() -> None:
    #                 clear cirrus cloud adj shadow  clear  clear
    fmask = np.array([0, 1, 2, 4, 8, 0, 0], dtype=np.uint8)
    nir = np.full(7, 3000, dtype=np.int16)
    swir1 = np.array([1000, 1000, 1000, 1000, 1000, 0, 1000], dtype=np.int16)
    swir2 = np.array([800, 800, 800, 800, 800, 800, -9999], dtype=np.int16)
    s1, s2, ndmi = swir_masked(nir, swir1, swir2, fmask, [1, 2, 3])
    assert np.isnan(s1).tolist() == [False, False, True, True, True, True, False]
    assert np.isnan(s2).tolist() == [False, False, True, True, True, False, True]
    # NDMI needs NIR and SWIR1: missing where SWIR1 is masked, present where only SWIR2 is bad
    assert np.isnan(ndmi).tolist() == np.isnan(s1).tolist()
    assert np.allclose(ndmi[0], 0.5)
    assert s1[0] == 1000.0 and s2[0] == 800.0


def test_feature_matrix_ndvi_only_is_unchanged_and_swir_appends_layers() -> None:
    ndvi = np.array([[0.1, 0.2], [0.3, np.nan]])
    stack = _stack(ndvi, np.zeros(2), np.ones(2, dtype=bool))
    assert feature_matrix(stack, ["ndvi"]) is stack.ndvi
    with pytest.raises(RuntimeError, match="нет слоёв"):
        feature_matrix(stack, ["ndvi", "swir"])

    stack.extra = {
        "swir1": np.full((2, 2), 1000.0, dtype=np.float32),
        "swir2": np.full((2, 2), 800.0, dtype=np.float32),
        "ndmi": np.full((2, 2), 0.5, dtype=np.float32),
    }
    full = feature_matrix(stack, ["ndvi", "swir"])
    assert full.shape == (2, 8)
    assert np.array_equal(full[:, :2], stack.ndvi, equal_nan=True)
    assert np.all(full[:, 2:4] == 1000.0) and np.all(full[:, 4:6] == 800.0) and np.all(full[:, 6:] == 0.5)


def test_features_config_validation() -> None:
    assert _baseline().features == ["ndvi"]
    with pytest.raises(ValueError):
        _baseline(features=["swir"])
    with pytest.raises(ValueError):
        _baseline(features=["ndvi", "swir"])  # no swir bands
    cfg = _baseline(features=["ndvi", "swir"], swir1_band="B06", swir2_band="B07")
    assert cfg.features == ["ndvi", "swir"]


def test_fingerprint_unchanged_for_default_features_changed_for_swir() -> None:
    crops = _crops()
    base = baseline_fingerprint(_ml(), crops)
    assert baseline_fingerprint(_ml(swir1_band="B06", swir2_band="B07"), crops) == base
    swir = _ml(features=["ndvi", "swir"], swir1_band="B06", swir2_band="B07")
    assert baseline_fingerprint(swir, crops) != base


def test_validation_text_quotes_checks_instead_of_not_checked_note() -> None:
    summary = {
        "macro_f1_agri_truth": 0.73, "macro_f1_rule_agri_truth": 0.9, "macro_f1_agri_bbox": 0.7,
        "accuracy_bbox": 0.86, "majority_baseline_bbox": 0.25, "buffer_px": 30, "dates_used": 22, "dates": 27,
        "features_text": "NDVI", "decision_rule": "argmax",
    }
    bbox = (0.0, 0.0, 1.0, 1.0)
    default = validation_text(summary, _baseline(), _crops(), [], bbox, 2021, "HLSL30")
    assert default.endswith("другие регионы и годы не проверялись")
    checked = validation_text(summary, _baseline(), _crops(), [], bbox, 2021, "HLSL30", ["перенос на 2020: 0.86"])
    assert checked.endswith("перенос на 2020: 0.86")
    assert "не проверялись" not in checked
