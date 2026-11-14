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
from fs_ml.features import LabelSpace, build_stack, label_space
from fs_ml.models import evaluate, rule_groups, spatial_cv, write_error_map

# Group whose precision and recall are reported separately in the sweep table.
_FALLOW = "fallow_idle"


def _row(buffer_px: int, metrics: dict[str, Any], space: LabelSpace) -> dict[str, Any]:
    by_group = {r["group"]: r for r in metrics["class_rows"]}
    row: dict[str, Any] = {
        "buffer_px": buffer_px,
        "macro_f1_rule_on_agri_px": metrics["macro_f1_rule"],
        "macro_f1_rule_on_bbox": metrics["macro_f1_rule_bbox"],
        "macro_f1_all5_on_agri_px": metrics["macro_f1_agri"],
    }
    for k in space.agricultural:
        name = space.names[k]
        row[f"f1_{name}_agri_px"] = by_group[name]["f1_agri_truth"]
        row[f"f1_{name}_bbox"] = by_group[name]["f1_bbox"]
    if _FALLOW in by_group:
        row["fallow_precision_bbox"] = by_group[_FALLOW]["precision_bbox"]
        row["fallow_recall_bbox"] = by_group[_FALLOW]["recall_bbox"]
    row["accuracy_bbox"] = metrics["accuracy_bbox"]
    row["trivial_class_bbox"] = metrics["majority_class_bbox"]
    row["trivial_acc_bbox"] = metrics["majority_baseline_bbox"]
    row["trivial_class_bbox_no_other"] = metrics["majority_class_bbox_no_other"]
    row["trivial_acc_bbox_no_other"] = metrics["majority_baseline_bbox_no_other"]
    row["n_usable_px"] = metrics["n_labeled"]
    return row


def f1_drops(table: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    """F1 drop from the smallest to the largest buffer (positive = worse)."""
    first = table.loc[table["buffer_px"].idxmin()]
    last = table.loc[table["buffer_px"].idxmax()]
    return {c: float(first[c] - last[c]) for c in columns}


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
    rule = [space.names[k] for k in rule_groups(space, bcfg)]
    stack = build_stack(bcfg, crops, ml.ingest.bbox, ml.ingest.dates)
    print(f"grid: {stack.grid.shape}, clipped to tile edge: {stack.grid.clipped_to_raster}", flush=True)
    print(f"dates used {len(stack.dates)}, dropped {[d.isoformat() for d in stack.dropped_dates]}", flush=True)
    print(f"stop rule: macro F1 over {rule}", flush=True)

    class_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    main_oof = None
    for buffer_px in bcfg.buffer_sweep:
        cfg_b = bcfg.model_copy(update={"buffer_px": buffer_px})
        diagnostics: list[dict[str, Any]] = []
        oof = spatial_cv(stack, cfg_b, space, diagnostics)
        metrics = evaluate(stack, oof, space, cfg_b)
        class_rows.extend({"buffer_px": buffer_px, **row} for row in metrics["class_rows"])
        fold_rows.extend({"buffer_px": buffer_px, **row} for row in diagnostics)
        row = _row(buffer_px, metrics, space)
        row["min_train_candidates_per_fold"] = min(d["n_train_candidates"] for d in diagnostics)
        row["min_classes_in_train_per_fold"] = min(d["classes_in_train"] for d in diagnostics)
        summary_rows.append(row)
        print(f"buffer {buffer_px}: macro F1 rule {metrics['macro_f1_rule']:.3f}", flush=True)
        excluded_support = {
            r["group"]: r["support_agri_truth"] for r in metrics["class_rows"] if r["group"] in bcfg.buffer_rule_exclude_groups
        }
        if buffer_px == bcfg.buffer_px:
            main_oof = oof

    reports = bcfg.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(summary_rows)
    pd.DataFrame(class_rows).to_csv(reports / "buffer_sweep_classes_2021.csv", index=False)
    table.to_csv(reports / "buffer_sweep_summary_2021.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(reports / "buffer_sweep_folds_2021.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.float_format", "{:.3f}".format):
        print(table.set_index("buffer_px").T.to_string())
    for group, support in excluded_support.items():
        values = ", ".join(f"{b}px {v:.3f}" for b, v in zip(table["buffer_px"], table[f"f1_{group}_agri_px"], strict=True))
        print(f"{group}: мало данных ({support} пикселей), F1 на сельхоз-пикселях: {values}; в правило не входит")

    if main_oof is None:
        raise RuntimeError("основной буфер не посчитан")
    counts = write_error_map(
        stack,
        main_oof,
        space,
        bcfg,
        bcfg.error_map_groups,
        reports / f"errors_fallow_alfalfa_buf{bcfg.buffer_px}_2021.png",
    )
    (reports / f"error_counts_buf{bcfg.buffer_px}_2021.json").write_text(
        json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("error counts:", counts)

    drops = f1_drops(table, ["macro_f1_rule_on_agri_px", "macro_f1_rule_on_bbox"])
    print("F1 drop smallest -> largest buffer (stop rule):", {k: round(v, 3) for k, v in drops.items()})
    limit = bcfg.buffer_max_f1_drop
    if limit is not None:
        over = {k: v for k, v in drops.items() if v > limit}
        if over:
            print(f"STOP: падение F1 больше {limit}: {over}")
            raise SystemExit(2)
        print(f"OK: ни одно падение F1 не превышает {limit}")


if __name__ == "__main__":
    main()
