"""Run the 2021 baseline: pixel stack, long table, spatial CV, reports and per-pixel summary."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from pathlib import Path

import numpy as np

from fs_ml.config import baseline_fingerprint, load_crops_config, load_ml_config
from fs_ml.features import build_stack, label_space, min_clear_dates, write_pixel_table
from fs_ml.inference import (
    history_groups,
    pixel_summary,
    season_months,
    write_pixel_summary,
)
from fs_ml.models import evaluate, spatial_cv, write_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="2021 crop baseline on HLS NDVI with CDL labels.")
    parser.add_argument("config", type=Path, help="путь к configs/ml.yaml")
    parser.add_argument("crops", type=Path, help="путь к configs/crops.yaml")
    parser.add_argument("--parquet-only", action="store_true", help="только pixels_2021.parquet, без модели")
    args = parser.parse_args()

    ml = load_ml_config(args.config)
    crops = load_crops_config(args.crops)
    bcfg = ml.baseline
    space = label_space(crops)

    stack = build_stack(bcfg, crops, ml.ingest.bbox, ml.ingest.dates)
    print(f"grid: {stack.grid.shape}, clipped to tile edge: {stack.grid.clipped_to_raster}")
    min_clear = min_clear_dates(bcfg.min_clear_fraction, len(stack.dates))
    labeled = (stack.labels >= 0) & stack.in_bbox
    usable = labeled & (stack.n_clear >= min_clear)
    dropped = [d.isoformat() for d in stack.dropped_dates]
    print(f"dates kept: {len(stack.dates)} of {len(stack.date_clear_share)}, dropped {len(dropped)}: {dropped}")
    print(f"grid pixels: {stack.labels.shape[0]}, inside lon/lat bbox: {int(stack.in_bbox.sum())}")
    print(f"labeled in bbox (CDL != 0): {int(labeled.sum())}")
    print(f"usable (clear kept dates >= {min_clear} of {len(stack.dates)}): {int(usable.sum())}")
    print(f"clear on all kept dates: {int((labeled & (stack.n_clear == len(stack.dates))).sum())}")
    clear_hist = np.bincount(stack.n_clear[labeled], minlength=len(stack.dates) + 1)
    print(f"clear-date histogram, labeled in bbox (0..{len(stack.dates)}): {clear_hist.tolist()}")

    table_path = bcfg.output_dir / "pixels_2021.parquet"
    n_rows = write_pixel_table(stack, crops, min_clear, table_path)
    print(f"pixel table rows: {n_rows} -> {table_path}")
    if args.parquet_only:
        return

    oof = spatial_cv(stack, bcfg, space)
    metrics = evaluate(stack, oof, space, bcfg)
    summary = write_reports(stack, oof, space, bcfg, metrics, baseline_fingerprint(ml, crops))
    print("summary:", summary)

    table = pixel_summary(
        stack,
        oof,
        space,
        ml.thresholds.bare_soil_ndvi,
        crops.export_merge,
        crops.low_confidence_groups,
        months=season_months(ml.ingest.temporal),
        previous=history_groups(stack.grid, bcfg.history_cdl, space) if bcfg.history_cdl else None,
    )
    summary_path = bcfg.output_dir / "pixel_summary_2021.parquet"
    write_pixel_summary(table, summary_path)
    print(f"pixel summary rows: {len(table)} -> {summary_path}")


if __name__ == "__main__":
    main()
