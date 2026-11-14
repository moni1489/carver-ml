"""Buffer sensitivity of the spatial CV on one bbox, plus the fallow / alfalfa error map for the main buffer."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from fs_ml.config import load_crops_config, load_ml_config
from fs_ml.features import build_stack, label_space
from fs_ml.models import evaluate, spatial_cv, write_error_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Spatial CV buffer sweep and error map.")
    parser.add_argument("config", type=Path, help="путь к configs/ml_bboxB.yaml")
    parser.add_argument("crops", type=Path, help="путь к configs/crops.yaml")
    args = parser.parse_args()

    ml = load_ml_config(args.config)
    crops = load_crops_config(args.crops)
    bcfg = ml.baseline
    if not bcfg.buffer_sweep:
        raise ValueError(f"{args.config}: baseline.buffer_sweep пуст")
    if bcfg.buffer_px not in bcfg.buffer_sweep:
        raise ValueError(f"{args.config}: основной buffer_px={bcfg.buffer_px} должен входить в buffer_sweep")
    space = label_space(crops)
    stack = build_stack(bcfg, crops, ml.ingest.bbox, ml.ingest.dates)
    print(f"grid: {stack.grid.shape}, clipped to tile edge: {stack.grid.clipped_to_raster}")

    class_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    main_oof = None
    for buffer_px in bcfg.buffer_sweep:
        cfg_b = bcfg.model_copy(update={"buffer_px": buffer_px})
        diagnostics: list[dict[str, Any]] = []
        oof = spatial_cv(stack, cfg_b, diagnostics)
        metrics = evaluate(stack, oof, space, cfg_b)
        class_rows.extend({"buffer_px": buffer_px, **row} for row in metrics["class_rows"])
        fold_rows.extend({"buffer_px": buffer_px, **row} for row in diagnostics)
        summary_rows.append(
            {
                "buffer_px": buffer_px,
                "accuracy_bbox": metrics["accuracy_bbox"],
                "majority_baseline_bbox": metrics["majority_baseline_bbox"],
                "accuracy_agri_truth": metrics["accuracy_agri"],
                "majority_baseline_agri_truth": metrics["majority_baseline_agri"],
                "macro_f1_agri_truth": metrics["macro_f1_agri"],
                "min_train_candidates_per_fold": min(d["n_train_candidates"] for d in diagnostics),
                "min_classes_in_train_per_fold": min(d["classes_in_train"] for d in diagnostics),
            }
        )
        print(f"buffer {buffer_px}: done, macro F1 agri {metrics['macro_f1_agri']:.3f}")
        if buffer_px == bcfg.buffer_px:
            main_oof = oof

    reports = bcfg.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(class_rows).to_csv(reports / "buffer_sweep_classes_2021.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(reports / "buffer_sweep_summary_2021.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(reports / "buffer_sweep_folds_2021.csv", index=False)

    if main_oof is None:
        raise RuntimeError("основной буфер не посчитан")
    counts = write_error_map(
        stack, main_oof, space, bcfg.error_map_groups, reports / f"errors_fallow_alfalfa_buf{bcfg.buffer_px}_2021.png"
    )
    (reports / f"error_counts_buf{bcfg.buffer_px}_2021.json").write_text(
        json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("error counts:", counts)


if __name__ == "__main__":
    main()
