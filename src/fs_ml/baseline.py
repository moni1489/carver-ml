"""Run the 2021 baseline: pixel stack, long table, spatial CV, reports and per-pixel summary."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from pathlib import Path

from fs_ml.config import load_crops_config, load_ml_config
from fs_ml.features import build_stack, label_space, write_long_table
from fs_ml.inference import pixel_summary, write_pixel_summary
from fs_ml.models import evaluate, spatial_cv, write_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="2021 crop baseline on HLS NDVI with CDL labels.")
    parser.add_argument("config", type=Path, help="путь к configs/ml.yaml")
    parser.add_argument("crops", type=Path, help="путь к configs/crops.yaml")
    args = parser.parse_args()

    ml = load_ml_config(args.config)
    crops = load_crops_config(args.crops)
    bcfg = ml.baseline
    space = label_space(crops)

    stack = build_stack(bcfg, crops, ml.ingest.bbox, ml.ingest.dates)
    print(f"grid: {stack.grid.shape}, clipped to tile edge: {stack.grid.clipped_to_raster}")
    print(f"dates: {[d.isoformat() for d in stack.dates]}")
    print(f"labeled pixels (CDL != 0): {int((stack.labels >= 0).sum())}")

    long_path = bcfg.output_dir / "pixels_2021.parquet"
    n_rows = write_long_table(stack, crops, long_path)
    print(f"long table rows: {n_rows} -> {long_path}")

    oof = spatial_cv(stack, bcfg)
    metrics = evaluate(stack, oof, space, bcfg)
    summary = write_reports(stack, oof, space, bcfg, metrics)
    print("summary:", summary)

    table = pixel_summary(
        stack,
        oof,
        space,
        ml.thresholds.bare_soil_ndvi,
        crops.export_merge,
        crops.low_confidence_groups,
    )
    summary_path = bcfg.output_dir / "pixel_summary_2021.parquet"
    write_pixel_summary(table, summary_path)
    print(f"pixel summary rows: {len(table)} -> {summary_path}")


if __name__ == "__main__":
    main()
