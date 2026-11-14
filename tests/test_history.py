"""Tests for the cultivation-history features and the safe CDL history clip (synthetic data, no D:)."""

import zipfile
from pathlib import Path

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin

from fs_ml.cdl import clip_if_missing
from fs_ml.config import CdlSource, HistoryCdl, baseline_fingerprint
from fs_ml.features import cultivated_lut, feature_matrix, history_features
from fs_ml.models import feature_text, n_features
from tests.test_pipeline import UTM12, _baseline, _crops, _grid, _ml, _stack, _write_tif

RANGES = [(1, 61), (66, 77), (204, 254)]


def _history(year: int) -> HistoryCdl:
    source = CdlSource(zip_path=Path(f"{year}.zip"), member=f"{year}.tif", crs="EPSG:5070", margin_px=0)
    return HistoryCdl(year=year, path=Path(f"{year}.tif"), source=source)


def test_cultivated_lut_code_assignment() -> None:
    lut = cultivated_lut(RANGES)
    cultivated = [1, 2, 21, 22, 36, 37, 61, 66, 74, 77, 204, 209, 225, 254]
    not_cultivated = [62, 63, 64, 65, 81, 111, 121, 124, 131, 141, 152, 176, 190, 195]
    assert all(lut[c] == 1.0 for c in cultivated)
    assert all(lut[c] == 0.0 for c in not_cultivated)
    assert np.isnan(lut[0])


def test_cultivated_lut_rejects_bad_ranges() -> None:
    with pytest.raises(ValueError):
        cultivated_lut([])
    with pytest.raises(ValueError):
        cultivated_lut([(0, 10)])
    with pytest.raises(ValueError):
        cultivated_lut([(20, 10)])


def test_history_features_flags_sum_and_missing_year() -> None:
    lut = cultivated_lut(RANGES)
    y2019 = np.array([[36, 152], [61, 0]], dtype=np.uint8)
    y2020 = np.array([[36, 176], [121, 2]], dtype=np.uint8)
    out = history_features([y2019, y2020], lut)
    assert out.shape == (4, 3)
    assert out[0].tolist() == [1.0, 1.0, 2.0]  # alfalfa both years
    assert out[1].tolist() == [0.0, 0.0, 0.0]  # shrubland, grassland
    assert out[2].tolist() == [1.0, 0.0, 1.0]  # fallow, then developed
    assert np.isnan(out[3, 0]) and out[3, 1] == 1.0 and np.isnan(out[3, 2])  # no data stays missing


def test_feature_matrix_appends_history_after_swir() -> None:
    stack = _stack(np.array([[0.1, 0.2], [0.3, 0.4]]), np.zeros(2), np.ones(2, dtype=bool))
    with pytest.raises(RuntimeError, match="history"):
        feature_matrix(stack, ["ndvi", "history"])
    stack.extra = {"history": np.array([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32)}
    full = feature_matrix(stack, ["ndvi", "history"])
    assert full.shape == (2, 5)
    assert full[:, 2:].tolist() == [[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]


def test_history_config_validation_and_feature_count() -> None:
    with pytest.raises(ValueError):
        _baseline(features=["ndvi", "history"])
    with pytest.raises(ValueError, match="метки"):
        _baseline(
            features=["ndvi", "history"],
            history_cdl=[{**_history(2020).model_dump(mode="json"), "path": "cdl.tif"}],
        )
    cfg = _baseline(
        features=["ndvi", "swir", "history"],
        swir1_band="B06",
        swir2_band="B07",
        history_cdl=[_history(2019).model_dump(mode="json"), _history(2020).model_dump(mode="json")],
    )
    assert n_features(cfg, 22) == 22 * 4 + 3
    assert "CDL 2019, 2020" in feature_text(cfg, 22)


def test_fingerprint_ignores_history_settings_until_enabled() -> None:
    crops = _crops()
    with_codes = crops.model_copy(update={"cultivated_codes": RANGES})
    hist = [_history(2019).model_dump(mode="json")]
    base = baseline_fingerprint(_ml(), crops)
    assert baseline_fingerprint(_ml(history_cdl=hist), with_codes) == base
    enabled = _ml(features=["ndvi", "history"], history_cdl=hist)
    assert baseline_fingerprint(enabled, with_codes) != base


def _zip_source(tmp_path: Path) -> CdlSource:
    albers = CRS.from_epsg(5070)
    from rasterio.warp import transform

    (ax,), (ay,) = transform(UTM12, albers, [405000.0], [3650000.0])
    tif = tmp_path / "src.tif"
    _write_tif(tif, np.full((100, 100), 36, dtype=np.uint8), albers, from_origin(ax - 1500, ay + 1500, 30.0, 30.0))
    archive = tmp_path / "cdl.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(tif, arcname="src.tif")
    return CdlSource(zip_path=archive, member="src.tif", crs="EPSG:5070", margin_px=2)


def test_clip_if_missing_creates_then_reuses_without_overwriting(tmp_path: Path) -> None:
    grid = _grid((6, 5), x0=405000.0, y0=3650000.0)
    source = _zip_source(tmp_path)
    out = tmp_path / "bboxB" / "2019.tif"
    out.parent.mkdir()
    assert clip_if_missing(source, grid, out) == "создан"
    before = out.read_bytes()
    mtime = out.stat().st_mtime_ns
    assert clip_if_missing(source, grid, out) == "уже был, на сетке HLS, использую"
    assert out.read_bytes() == before and out.stat().st_mtime_ns == mtime
    assert not list(out.parent.glob(".*.tif"))  # no temporary files left


def test_clip_if_missing_refuses_existing_clip_on_other_grid(tmp_path: Path) -> None:
    grid = _grid((6, 5), x0=405000.0, y0=3650000.0)
    out = tmp_path / "2020.tif"
    _write_tif(out, np.ones((6, 5), dtype=np.uint8), UTM12, from_origin(405030.0, 3650000.0, 30.0, 30.0))
    before = out.read_bytes()
    with pytest.raises(RuntimeError, match="не перезаписываю"):
        clip_if_missing(_zip_source(tmp_path), grid, out)
    assert out.read_bytes() == before
