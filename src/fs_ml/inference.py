"""Per-pixel summary for the 2021 season: dominant predicted crop and count of bare-soil observations."""

from pathlib import Path

import numpy as np
import pandas as pd
from rasterio.transform import xy
from rasterio.warp import transform as warp_transform

from fs_ml.features import LabelSpace, PixelStack


def _export_names(space: LabelSpace, export_merge: dict[str, list[str]]) -> np.ndarray:
    """Map each label (plus 'unpredicted') to its export name; members of a merge group share one name."""
    target = {member: group for group, members in export_merge.items() for member in members}
    names = space.names + ["unpredicted"]
    return np.array([target.get(name, name) for name in names])


def pixel_summary(
    stack: PixelStack,
    oof: np.ndarray,
    space: LabelSpace,
    bare_ndvi: float,
    export_merge: dict[str, list[str]],
    low_confidence_groups: list[str],
) -> pd.DataFrame:
    height, width = stack.grid.shape
    rows = np.repeat(np.arange(height, dtype=np.int32), width)
    cols = np.tile(np.arange(width, dtype=np.int32), height)
    xs, ys = xy(stack.grid.transform, rows, cols, offset="center")
    lon, lat = warp_transform(stack.grid.crs, "EPSG:4326", xs, ys)

    valid = ~np.isnan(stack.ndvi)
    bare_obs = np.sum((stack.ndvi < bare_ndvi) & valid, axis=1).astype(np.int16)
    valid_obs = np.sum(valid, axis=1).astype(np.int16)
    index = np.where(oof >= 0, oof, len(space.names))
    crop = np.array(space.names + ["unpredicted"])[index]
    crop_export = _export_names(space, export_merge)[index]
    low_conf = np.isin(crop, low_confidence_groups)
    return pd.DataFrame(
        {
            "row": rows,
            "col": cols,
            "lon": np.asarray(lon, dtype=np.float64),
            "lat": np.asarray(lat, dtype=np.float64),
            "dominant_crop": crop,
            "dominant_crop_export": crop_export,
            "low_confidence": low_conf,
            "bare_obs_dates": bare_obs,
            "valid_obs_dates": valid_obs,
        }
    )


def write_pixel_summary(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
