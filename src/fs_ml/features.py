"""HLS pixel stack: cloud-masked NDVI per date, and CDL labels on the same grid."""

import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
from rasterio.transform import xy
from rasterio.warp import transform, transform_bounds
from rasterio.windows import Window, from_bounds

from fs_ml.config import BaselineConfig, CropsConfig

_HLS_NAME = re.compile(r"HLS\.L30\.(T\w{5})\.(\d{4})(\d{3})T\d{6}\.v2\.0\.(\w+)\.tif$")
# Tolerance for comparing affine transforms of the CDL clip and the HLS window (metres / pixel).
_TRANSFORM_TOL = 1e-6


@dataclass(frozen=True)
class Grid:
    window: Window
    transform: Any
    crs: Any
    shape: tuple[int, int]
    clipped_to_raster: bool


@dataclass(frozen=True)
class LabelSpace:
    names: list[str]
    agricultural: list[int]
    lut: np.ndarray  # CDL code -> label index, -1 for no data


@dataclass
class PixelStack:
    grid: Grid
    dates: list[date]
    ndvi: np.ndarray  # (N, T) float32 over kept dates, NaN where cloudy, shadowed or invalid DN (never filled)
    n_clear: np.ndarray  # (N,) int16, number of clear kept dates
    in_bbox: np.ndarray  # (N,) bool, pixel centre inside the lon/lat bbox
    date_clear_share: dict[date, float]  # every requested date -> clear share over in-bbox pixels
    dropped_dates: list[date]  # dates removed whole because clear share < min_date_clear_share
    cdl: np.ndarray  # (N,) uint8, raw CDL code
    labels: np.ndarray  # (N,) int16, label index or -1
    # Extra per-date layers on kept dates, (N, T) float32 with NaN gaps: "swir1", "swir2", "ndmi" for features: swir.
    extra: dict[str, np.ndarray] = field(default_factory=dict)


def scene_files(hls_dir: Path, wanted: list[date], bands: list[str]) -> dict[date, dict[str, Path]]:
    grouped: dict[date, dict[str, Path]] = {}
    tiles: set[str] = set()
    wanted_set = set(wanted)
    for path in hls_dir.glob("*.tif"):
        match = _HLS_NAME.search(path.name)
        if match is None:
            continue
        day = datetime.strptime(f"{match.group(2)}{match.group(3)}", "%Y%j").date()
        if day not in wanted_set:
            continue
        tiles.add(match.group(1))
        grouped.setdefault(day, {})[match.group(4)] = path
    if len(tiles) != 1:
        raise RuntimeError(f"ожидается один тайл HLS, найдено: {sorted(tiles)}")
    missing = [d.isoformat() for d in wanted if d not in grouped]
    if missing:
        raise RuntimeError(f"нет файлов HLS на даты: {missing}")
    for day, files in grouped.items():
        absent = [b for b in bands if b not in files]
        if absent:
            raise RuntimeError(f"на {day.isoformat()} нет каналов {absent}")
    return grouped


def bbox_grid(reference: Path, bbox: tuple[float, float, float, float]) -> Grid:
    """Window of the bbox on the HLS grid, intersected with the raster so reads never run off the tile edge."""
    with rasterio.open(reference) as ds:
        left, bottom, right, top = transform_bounds("EPSG:4326", ds.crs, *bbox)
        requested = from_bounds(left, bottom, right, top, ds.transform).round_offsets().round_lengths()
        window = requested.intersection(Window(0, 0, ds.width, ds.height))
        transform = ds.window_transform(window)
        crs = ds.crs
    clipped = (int(window.width), int(window.height)) != (int(requested.width), int(requested.height))
    shape = (int(window.height), int(window.width))
    return Grid(window=window, transform=transform, crs=crs, shape=shape, clipped_to_raster=clipped)


def grid_mismatch(crs: Any, affine: Any, shape: tuple[int, int], grid: Grid) -> list[str]:
    """Differences between a raster grid and the HLS window grid (empty list = same grid)."""
    problems: list[str] = []
    if crs != grid.crs:
        problems.append(f"CRS {crs} != {grid.crs}")
    if not affine.almost_equals(grid.transform, precision=_TRANSFORM_TOL):
        problems.append(f"transform {tuple(affine)[:6]} != {tuple(grid.transform)[:6]}")
    if tuple(shape) != tuple(grid.shape):
        problems.append(f"shape {tuple(shape)} != {tuple(grid.shape)}")
    return problems


def cdl_on_grid(cdl_path: Path, grid: Grid) -> np.ndarray:
    """Read the CDL clip that must already lie on the HLS window grid (python -m fs_ml.cdl clip).

    A clip on another grid (stale bbox, other tile) raises instead of being silently resampled.
    """
    with rasterio.open(cdl_path) as src:
        problems = grid_mismatch(src.crs, src.transform, (src.height, src.width), grid)
        if problems:
            raise RuntimeError(
                f"клип CDL {cdl_path} не на сетке HLS ({'; '.join(problems)}): "
                "пересоздай его командой python -m fs_ml.cdl clip <config>"
            )
        data: np.ndarray = src.read(1)
    return data


def label_space(crops: CropsConfig) -> LabelSpace:
    names = list(crops.groups) + [crops.other_group]
    lut = np.full(256, len(names) - 1, dtype=np.int16)
    for index, codes in enumerate(crops.groups.values()):
        for code in codes:
            lut[code] = index
    lut[0] = -1
    unknown = [name for name in crops.agricultural_groups if name not in names]
    if unknown:
        raise ValueError(f"agricultural_groups не найдены в groups: {unknown}")
    agricultural = [names.index(name) for name in crops.agricultural_groups]
    return LabelSpace(names=names, agricultural=agricultural, lut=lut)


def cloud_mask(fmask: np.ndarray, cloud_bits: list[int]) -> np.ndarray:
    """True where any of the Fmask cloud bits is set."""
    bad = np.zeros(fmask.shape, dtype=bool)
    for bit in cloud_bits:
        bad |= ((fmask >> bit) & 1).astype(bool)
    return bad


def ndvi_masked(nir: np.ndarray, red: np.ndarray, fmask: np.ndarray, cloud_bits: list[int]) -> np.ndarray:
    """NDVI from HLS DN (scale and offset cancel in the ratio); NaN where cloud bits or no data."""
    nir_f = nir.astype(np.float32)
    red_f = red.astype(np.float32)
    bad = cloud_mask(fmask, cloud_bits)
    denom = nir_f + red_f
    ok = (nir > 0) & (red > 0) & ~bad & (denom > 0)
    out = np.full(nir.shape, np.nan, dtype=np.float32)
    out[ok] = (nir_f[ok] - red_f[ok]) / denom[ok]
    return out


def swir_masked(
    nir: np.ndarray, swir1: np.ndarray, swir2: np.ndarray, fmask: np.ndarray, cloud_bits: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SWIR1, SWIR2 (HLS DN as float; RandomForest is scale-invariant) and NDMI = (NIR - SWIR1) / (NIR + SWIR1).

    Same Fmask cloud bits as NDVI; a band <= 0 is missing. NaN where masked, nothing is filled.
    """
    bad = cloud_mask(fmask, cloud_bits)
    nir_f = nir.astype(np.float32)
    s1 = swir1.astype(np.float32)
    s2 = swir2.astype(np.float32)
    ok1 = ~bad & (swir1 > 0)
    ok2 = ~bad & (swir2 > 0)
    denom = nir_f + s1
    ok_ndmi = ok1 & (nir > 0) & (denom > 0)
    out1 = np.where(ok1, s1, np.nan).astype(np.float32)
    out2 = np.where(ok2, s2, np.nan).astype(np.float32)
    ndmi = np.full(nir.shape, np.nan, dtype=np.float32)
    ndmi[ok_ndmi] = (nir_f[ok_ndmi] - s1[ok_ndmi]) / denom[ok_ndmi]
    return out1, out2, ndmi


def feature_matrix(stack: "PixelStack", features: list[str]) -> np.ndarray:
    """Model inputs (N, F): NDVI per kept date, then SWIR1, SWIR2 and NDMI per kept date if features has swir."""
    if features == ["ndvi"]:
        return stack.ndvi
    blocks = [stack.ndvi]
    if "swir" in features:
        missing = [k for k in ("swir1", "swir2", "ndmi") if k not in stack.extra]
        if missing:
            raise RuntimeError(f"в стеке нет слоёв {missing}: собери стек с features, содержащим swir")
        blocks += [stack.extra["swir1"], stack.extra["swir2"], stack.extra["ndmi"]]
    return np.ascontiguousarray(np.hstack(blocks), dtype=np.float32)


def min_clear_dates(fraction: float, n_dates: int) -> int:
    """Smallest clear-date count that is at least `fraction` of all dates (0.5 of 27 -> 14)."""
    return math.ceil(fraction * n_dates - 1e-9)


def pixel_lonlat(grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Longitude and latitude of every pixel centre, row-major like the flattened stack."""
    height, width = grid.shape
    rows = np.repeat(np.arange(height, dtype=np.int32), width)
    cols = np.tile(np.arange(width, dtype=np.int32), height)
    xs, ys = xy(grid.transform, rows, cols, offset="center")
    lon, lat = transform(grid.crs, "EPSG:4326", xs, ys)
    return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def inside_bbox(lon: np.ndarray, lat: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    west, south, east, north = bbox
    inside: np.ndarray = (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)
    return inside


def select_dates(ndvi: np.ndarray, in_bbox: np.ndarray, min_share: float) -> tuple[np.ndarray, np.ndarray]:
    """Clear share per date over in-bbox pixels and the mask of dates kept (share >= min_share).

    `ndvi` is (N, T) with NaN where the date is not clear. Raises if no pixel is in the bbox or no date is kept.
    """
    if not in_bbox.any():
        raise RuntimeError("ни один пиксель сетки не попал в рамку bbox")
    share: np.ndarray = np.mean(~np.isnan(ndvi[in_bbox]), axis=0)
    keep: np.ndarray = share >= min_share
    if not keep.any():
        raise RuntimeError(f"ни одна дата не набрала {min_share:.0%} чистых пикселей в bbox")
    return share, keep


def build_stack(
    cfg: BaselineConfig,
    crops: CropsConfig,
    bbox: tuple[float, float, float, float],
    dates: list[date],
) -> PixelStack:
    use_swir = "swir" in cfg.features
    bands = [cfg.nir_band, cfg.red_band, cfg.fmask_band]
    if use_swir:
        if cfg.swir1_band is None or cfg.swir2_band is None:
            raise ValueError("для features: swir нужны swir1_band и swir2_band")
        bands += [cfg.swir1_band, cfg.swir2_band]
    files = scene_files(cfg.hls_dir, dates, bands)
    ordered = sorted(dates)
    first_band = files[ordered[0]][cfg.nir_band]
    grid = bbox_grid(first_band, bbox)
    cdl = cdl_on_grid(cfg.cdl_path, grid)
    space = label_space(crops)
    labels = space.lut[cdl]

    lon, lat = pixel_lonlat(grid)
    in_bbox = inside_bbox(lon, lat, bbox)

    n_pixels = grid.shape[0] * grid.shape[1]
    ndvi = np.empty((n_pixels, len(ordered)), dtype=np.float32)
    swir_layers = {k: np.empty((n_pixels, len(ordered)), dtype=np.float32) for k in ("swir1", "swir2", "ndmi")}
    for t, day in enumerate(ordered):
        with rasterio.open(files[day][cfg.nir_band]) as nir_ds:
            nir = nir_ds.read(1, window=grid.window)
        with rasterio.open(files[day][cfg.red_band]) as red_ds:
            red = red_ds.read(1, window=grid.window)
        with rasterio.open(files[day][cfg.fmask_band]) as fmask_ds:
            fmask = fmask_ds.read(1, window=grid.window)
        ndvi[:, t] = ndvi_masked(nir, red, fmask, cfg.cloud_bits).ravel()
        if use_swir and cfg.swir1_band is not None and cfg.swir2_band is not None:
            with rasterio.open(files[day][cfg.swir1_band]) as s1_ds:
                swir1 = s1_ds.read(1, window=grid.window)
            with rasterio.open(files[day][cfg.swir2_band]) as s2_ds:
                swir2 = s2_ds.read(1, window=grid.window)
            s1, s2, ndmi = swir_masked(nir, swir1, swir2, fmask, cfg.cloud_bits)
            swir_layers["swir1"][:, t] = s1.ravel()
            swir_layers["swir2"][:, t] = s2.ravel()
            swir_layers["ndmi"][:, t] = ndmi.ravel()
    share, keep = select_dates(ndvi, in_bbox, cfg.min_date_clear_share)
    kept_ndvi = np.ascontiguousarray(ndvi[:, keep])
    return PixelStack(
        grid=grid,
        dates=[d for d, k in zip(ordered, keep, strict=True) if k],
        ndvi=kept_ndvi,
        n_clear=(~np.isnan(kept_ndvi)).sum(axis=1).astype(np.int16),
        in_bbox=in_bbox,
        date_clear_share={d: float(v) for d, v in zip(ordered, share, strict=True)},
        dropped_dates=[d for d, k in zip(ordered, keep, strict=True) if not k],
        cdl=cdl.ravel(),
        labels=labels.ravel(),
        extra={k: np.ascontiguousarray(v[:, keep]) for k, v in swir_layers.items()} if use_swir else {},
    )


def write_pixel_table(stack: PixelStack, crops: CropsConfig, min_clear: int, path: Path) -> int:
    """One row per labelled pixel inside the lon/lat bbox: CDL code and group, NDVI per kept date.

    NDVI is NaN where the date is not clear for that pixel (Fmask bits 1-3 or B04/B05 <= 0); nothing is filled.
    `clear_mask` bit t is set when kept date t (column order of ndvi_*) was clear.
    `usable` marks pixels with at least `min_clear` clear kept dates: only these go to training and metrics.
    """
    space = label_space(crops)
    height, width = stack.grid.shape
    rows = np.repeat(np.arange(height, dtype=np.int32), width)
    cols = np.tile(np.arange(width, dtype=np.int32), height)
    labeled = (stack.labels >= 0) & stack.in_bbox
    clear_bits = (~np.isnan(stack.ndvi)).astype(np.uint32) << np.arange(len(stack.dates), dtype=np.uint32)
    columns: dict[str, pa.Array] = {
        "row": pa.array(rows[labeled]),
        "col": pa.array(cols[labeled]),
        "cdl_code": pa.array(stack.cdl[labeled]),
        "label": pa.DictionaryArray.from_arrays(
            pa.array(stack.labels[labeled].astype(np.int32)), pa.array(space.names)
        ),
        "n_clear_dates": pa.array(stack.n_clear[labeled]),
        "clear_mask": pa.array(clear_bits.sum(axis=1, dtype=np.uint32)[labeled]),
        "usable": pa.array(stack.n_clear[labeled] >= min_clear),
    }
    for t, day in enumerate(stack.dates):
        columns[f"ndvi_{day.isoformat()}"] = pa.array(stack.ndvi[labeled, t], from_pandas=True)
    table = pa.table(columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return int(table.num_rows)

