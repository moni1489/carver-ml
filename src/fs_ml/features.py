"""HLS pixel stack: cloud-masked NDVI per date, and CDL labels on the same grid."""

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds

from fs_ml.config import BaselineConfig, CropsConfig

_HLS_NAME = re.compile(r"HLS\.L30\.(T\w{5})\.(\d{4})(\d{3})T\d{6}\.v2\.0\.(\w+)\.tif$")
_EPOCH = date(1970, 1, 1)


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
    ndvi: np.ndarray  # (N, T) float32, NaN where cloudy, shadowed or no data
    cdl: np.ndarray  # (N,) uint8, raw CDL code
    labels: np.ndarray  # (N,) int16, label index or -1


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


def cdl_on_grid(cdl_path: Path, grid: Grid) -> np.ndarray:
    """Reproject CDL to the HLS window grid with nearest-neighbour, so pixels line up exactly."""
    destination = np.zeros(grid.shape, dtype=np.uint8)
    with rasterio.open(cdl_path) as src:
        reproject(
            source=src.read(1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            resampling=Resampling.nearest,
            src_nodata=None,
            dst_nodata=None,
        )
    return destination


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


def ndvi_masked(nir: np.ndarray, red: np.ndarray, fmask: np.ndarray, cloud_bits: list[int]) -> np.ndarray:
    """NDVI from HLS DN (scale and offset cancel in the ratio); NaN where cloud bits or no data."""
    nir_f = nir.astype(np.float32)
    red_f = red.astype(np.float32)
    bad = np.zeros(nir.shape, dtype=bool)
    for bit in cloud_bits:
        bad |= ((fmask >> bit) & 1).astype(bool)
    denom = nir_f + red_f
    ok = (nir > 0) & (red > 0) & ~bad & (denom > 0)
    out = np.full(nir.shape, np.nan, dtype=np.float32)
    out[ok] = (nir_f[ok] - red_f[ok]) / denom[ok]
    return out


def build_stack(
    cfg: BaselineConfig,
    crops: CropsConfig,
    bbox: tuple[float, float, float, float],
    dates: list[date],
) -> PixelStack:
    bands = [cfg.nir_band, cfg.red_band, cfg.fmask_band]
    files = scene_files(cfg.hls_dir, dates, bands)
    ordered = sorted(dates)
    first_band = files[ordered[0]][cfg.nir_band]
    grid = bbox_grid(first_band, bbox)
    cdl = cdl_on_grid(cfg.cdl_path, grid)
    space = label_space(crops)
    labels = space.lut[cdl]

    n_pixels = grid.shape[0] * grid.shape[1]
    ndvi = np.empty((n_pixels, len(ordered)), dtype=np.float32)
    for t, day in enumerate(ordered):
        with rasterio.open(files[day][cfg.nir_band]) as nir_ds:
            nir = nir_ds.read(1, window=grid.window)
        with rasterio.open(files[day][cfg.red_band]) as red_ds:
            red = red_ds.read(1, window=grid.window)
        with rasterio.open(files[day][cfg.fmask_band]) as fmask_ds:
            fmask = fmask_ds.read(1, window=grid.window)
        ndvi[:, t] = ndvi_masked(nir, red, fmask, cfg.cloud_bits).ravel()
    return PixelStack(
        grid=grid,
        dates=ordered,
        ndvi=ndvi,
        cdl=cdl.ravel(),
        labels=labels.ravel(),
    )


def write_long_table(stack: PixelStack, crops: CropsConfig, path: Path) -> int:
    """Write Parquet rows (pixel, date, NDVI, CDL label), one date at a time to keep memory low."""
    space = label_space(crops)
    names = pa.array(space.names)
    rows_idx, cols_idx = np.indices(stack.grid.shape)
    rows_flat = rows_idx.ravel().astype(np.int32)
    cols_flat = cols_idx.ravel().astype(np.int32)
    labeled = stack.labels >= 0
    keep_rows = rows_flat[labeled]
    keep_cols = cols_flat[labeled]
    keep_cdl = stack.cdl[labeled]
    keep_labels = stack.labels[labeled].astype(np.int32)
    schema = pa.schema(
        [
            ("row", pa.int32()),
            ("col", pa.int32()),
            ("date", pa.date32()),
            ("ndvi", pa.float32()),
            ("cdl_code", pa.uint8()),
            ("label", pa.dictionary(pa.int32(), pa.string())),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with pq.ParquetWriter(path, schema) as writer:
        for t, day in enumerate(stack.dates):
            n = int(keep_rows.shape[0])
            day_index = (day - _EPOCH).days
            table = pa.Table.from_arrays(
                [
                    pa.array(keep_rows),
                    pa.array(keep_cols),
                    pa.array(np.full(n, day_index, dtype=np.int32), type=pa.date32()),
                    pa.array(stack.ndvi[labeled, t]),
                    pa.array(keep_cdl),
                    pa.DictionaryArray.from_arrays(pa.array(keep_labels), names),
                ],
                schema=schema,
            )
            writer.write_table(table)
            total += n
    return total
