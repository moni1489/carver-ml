"""National CDL: archive integrity check and bbox clip onto the HLS grid, read in place from the zip."""

import argparse
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds

from fs_ml.config import CdlSource, load_ml_config
from fs_ml.features import Grid, bbox_grid, grid_mismatch, scene_files

# CDL code 0 is "background / no data" in the USDA legend.
_CDL_NODATA = 0
# Points per edge when projecting the HLS window outline into the CDL CRS (curved edges in Albers).
_DENSIFY = 41


@dataclass(frozen=True)
class ArchiveCheck:
    path: Path
    ok: bool
    detail: str


@dataclass(frozen=True)
class ClipResult:
    source_window: Window
    grid_shape: tuple[int, int]
    nodata_share: float
    codes: dict[int, int]


def check_archive(path: Path) -> ArchiveCheck:
    """CRC check of every member (zipfile.testzip). A broken archive is reported, never repaired."""
    try:
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
    except (zipfile.BadZipFile, OSError) as exc:
        return ArchiveCheck(path=path, ok=False, detail=f"{type(exc).__name__}: {exc}")
    if bad is not None:
        return ArchiveCheck(path=path, ok=False, detail=f"первый битый файл в архиве: {bad}")
    return ArchiveCheck(path=path, ok=True, detail="testzip: ошибок нет")


def source_window(source: CdlSource, grid: Grid) -> Window:
    """Window of the national CDL (native CRS) that covers the HLS grid, padded by margin_px."""
    height, width = grid.shape
    left, top = grid.transform @ (0, 0)
    right, bottom = grid.transform @ (width, height)
    with rasterio.open(source.vsi_path()) as src:
        expected = CRS.from_user_input(source.crs)
        if src.crs != expected:
            raise RuntimeError(f"CRS CDL {src.crs} не совпадает с ожидаемым {expected}")
        bounds = transform_bounds(grid.crs, src.crs, left, bottom, right, top, densify_pts=_DENSIFY)
        window = from_bounds(*bounds, transform=src.transform)
        pad = source.margin_px
        padded = Window(
            int(np.floor(window.col_off)) - pad,
            int(np.floor(window.row_off)) - pad,
            int(np.ceil(window.width)) + 2 * pad,
            int(np.ceil(window.height)) + 2 * pad,
        )
        clipped: Window = padded.intersection(Window(0, 0, src.width, src.height))
    return clipped


def clip_to_grid(source: CdlSource, grid: Grid, out_path: Path) -> ClipResult:
    """Read only the bbox window from the zipped CDL and resample it with nearest onto the HLS grid."""
    window = source_window(source, grid)
    destination = np.full(grid.shape, _CDL_NODATA, dtype=np.uint8)
    with rasterio.open(source.vsi_path()) as src:
        block = src.read(1, window=window)
        reproject(
            source=block,
            destination=destination,
            src_transform=src.window_transform(window),
            src_crs=src.crs,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            resampling=Resampling.nearest,
            src_nodata=_CDL_NODATA,
            dst_nodata=_CDL_NODATA,
        )
        try:
            colormap = src.colormap(1)
        except ValueError:
            colormap = None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "width": grid.shape[1],
        "height": grid.shape[0],
        "count": 1,
        "dtype": "uint8",
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": _CDL_NODATA,
        "compress": "deflate",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(destination, 1)
        if colormap is not None:
            dst.write_colormap(1, colormap)
    values, counts = np.unique(destination, return_counts=True)
    return ClipResult(
        source_window=window,
        grid_shape=grid.shape,
        nodata_share=float(np.mean(destination == _CDL_NODATA)),
        codes={int(v): int(c) for v, c in zip(values, counts, strict=True)},
    )


def clip_if_missing(source: CdlSource, grid: Grid, out_path: Path) -> str:
    """Clip only if `out_path` does not exist; never overwrite an existing clip (another job may use it).

    An existing clip on the HLS grid is reused; on another grid it raises. New clips are written to a temporary
    file and renamed, so a parallel reader never sees a half-written file.
    """
    if out_path.exists():
        with rasterio.open(out_path) as ds:
            problems = grid_mismatch(ds.crs, ds.transform, (ds.height, ds.width), grid)
        if problems:
            raise RuntimeError(f"{out_path} уже есть, но не на сетке HLS ({'; '.join(problems)}); не перезаписываю")
        return "уже был, на сетке HLS, использую"
    tmp = out_path.with_name(f".{out_path.stem}.{os.getpid()}.tmp.tif")
    clip_to_grid(source, grid, tmp)
    try:
        os.rename(tmp, out_path)  # Windows: fails if another job created out_path meanwhile
    except FileExistsError:
        tmp.replace(out_path.with_name(f".{out_path.stem}.{os.getpid()}.unused.tif"))
        return clip_if_missing(source, grid, out_path)
    return "создан"


def main() -> None:
    parser = argparse.ArgumentParser(description="CDL archive check and bbox clip onto the HLS grid.")
    parser.add_argument("command", choices=["check", "clip", "clip-history"])
    parser.add_argument("config", type=Path, help="путь к configs/ml_bboxB.yaml")
    args = parser.parse_args()
    ml = load_ml_config(args.config)
    bcfg = ml.baseline

    if args.command == "check":
        if not bcfg.cdl_archives:
            raise ValueError(f"{args.config}: baseline.cdl_archives пуст")
        checks = [check_archive(path) for path in bcfg.cdl_archives]
        for check in checks:
            print(f"{check.path}: {'OK' if check.ok else 'BROKEN'} ({check.detail})")
        if not all(c.ok for c in checks):
            raise SystemExit(1)
        return

    bands = [bcfg.nir_band, bcfg.red_band, bcfg.fmask_band]
    files = scene_files(bcfg.hls_dir, ml.ingest.dates, bands)
    reference = files[min(files)][bcfg.nir_band]
    grid = bbox_grid(reference, ml.ingest.bbox)
    if args.command == "clip-history":
        if not bcfg.history_cdl:
            raise ValueError(f"{args.config}: baseline.history_cdl пуст")
        for item in sorted(bcfg.history_cdl, key=lambda h: h.year):
            print(f"{item.year}: {item.path}: {clip_if_missing(item.source, grid, item.path)}")
        return
    if bcfg.cdl_source is None:
        raise ValueError(f"{args.config}: нет baseline.cdl_source")
    result = clip_to_grid(bcfg.cdl_source, grid, bcfg.cdl_path)
    print(f"HLS grid: {grid.shape}, crs {grid.crs}, clipped to tile edge: {grid.clipped_to_raster}")
    print(f"CDL window ({bcfg.cdl_source.crs}): {result.source_window}")
    print(f"written: {bcfg.cdl_path}, nodata share {result.nodata_share:.4f}")
    top = sorted(result.codes.items(), key=lambda kv: -kv[1])[:12]
    total = grid.shape[0] * grid.shape[1]
    print("top codes:", ", ".join(f"{code}: {count / total:.3f}" for code, count in top))


if __name__ == "__main__":
    main()
