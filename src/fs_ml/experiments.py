"""Experiments on bbox B: fallow decision rule (probability threshold, desert mask) and clear-date share."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from fs_ml.config import (
    BaselineConfig,
    CropsConfig,
    ExperimentConfig,
    MLConfig,
    load_crops_config,
    load_ml_config,
)
from fs_ml.features import (
    LabelSpace,
    PixelStack,
    build_stack,
    label_space,
    pixel_lonlat,
)
from fs_ml.furrow import aggregate, blocked_groups, build_layer
from fs_ml.models import (
    block_index,
    decide,
    decision_rule_text,
    evaluate,
    feature_text,
    n_features,
    rule_groups,
    spatial_cv_proba,
    usable_mask,
)


def crop_groups(space: LabelSpace, cfg: BaselineConfig, exp: ExperimentConfig) -> list[str]:
    """Crops compared in the experiments: stop-rule groups (agri minus buffer_rule_exclude_groups) minus the
    threshold group itself."""
    if exp.threshold_group not in space.names:
        raise ValueError(f"experiments.threshold_group {exp.threshold_group} не найден среди групп")
    return [space.names[k] for k in rule_groups(space, cfg) if space.names[k] != exp.threshold_group]


def _metrics_row(
    name: str, stack: PixelStack, labels: np.ndarray, space: LabelSpace, cfg: BaselineConfig, exp: ExperimentConfig
) -> dict[str, Any]:
    metrics = evaluate(stack, labels, space, cfg)
    by_group = {r["group"]: r for r in metrics["class_rows"]}
    group = exp.threshold_group
    other = space.names[-1]
    g = by_group[group]
    row: dict[str, Any] = {"variant": name}
    row |= {
        f"{group}_f1_bbox": g["f1_bbox"],
        f"{group}_precision_bbox": g["precision_bbox"],
        f"{group}_recall_bbox": g["recall_bbox"],
        f"{group}_f1_agri": g["f1_agri_truth"],
        f"{group}_precision_agri": g["precision_agri_truth"],
        f"{group}_recall_agri": g["recall_agri_truth"],
    }
    for crop in crop_groups(space, cfg, exp):
        row[f"f1_{crop}_agri"] = by_group[crop]["f1_agri_truth"]
    row["macro_f1_rule_groups_agri"] = metrics["macro_f1_rule"]
    row["macro_f1_rule_groups_bbox"] = metrics["macro_f1_rule_bbox"]
    row["accuracy_bbox"] = metrics["accuracy_bbox"]
    row[f"{other}_recall_bbox"] = by_group[other]["recall_bbox"]
    return row


def fallow_experiment(
    stack: PixelStack, proba: np.ndarray, space: LabelSpace, cfg: BaselineConfig, exp: ExperimentConfig
) -> pd.DataFrame:
    """Variant 1: probability threshold for the group. Variant 2: desert mask by season max NDVI."""
    plain = cfg.model_copy(update={"min_proba": {}, "desert_max_ndvi": None, "desert_excluded_groups": []})
    rows = [_metrics_row("argmax", stack, decide(proba, stack, space, plain), space, plain, exp)]
    for threshold in exp.min_proba_grid:
        variant = plain.model_copy(update={"min_proba": {exp.threshold_group: threshold}})
        labels = decide(proba, stack, space, variant)
        rows.append(_metrics_row(f"min_proba={threshold}", stack, labels, space, variant, exp))
    for ndvi in exp.desert_max_ndvi_grid:
        variant = plain.model_copy(update={"desert_max_ndvi": ndvi, "desert_excluded_groups": [exp.threshold_group]})
        labels = decide(proba, stack, space, variant)
        rows.append(_metrics_row(f"desert_max_ndvi<{ndvi}", stack, labels, space, variant, exp))
    return pd.DataFrame(rows)


def per_block_f1(
    stack: PixelStack, labels: np.ndarray, scored: np.ndarray, space: LabelSpace, groups: list[str], blocks: int
) -> pd.DataFrame:
    """F1 per group inside every CV block (scored pixels only)."""
    ids = block_index(stack.grid.shape, blocks)
    rows: list[dict[str, Any]] = []
    for b in range(blocks * blocks):
        m = scored & (ids == b)
        row: dict[str, Any] = {"block": b, "n": int(m.sum())}
        for g in groups:
            k = space.names.index(g)
            row[g] = float(f1_score(stack.labels[m] == k, labels[m] == k, zero_division=0)) if m.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# Feature sets compared by --swir: current NDVI-only model against NDVI + SWIR1 + SWIR2 + NDMI.
_SWIR_VARIANTS: list[list[str]] = [["ndvi"], ["ndvi", "swir"]]


def swir_experiment(ml: MLConfig, crops: CropsConfig, exp: ExperimentConfig) -> None:
    """Same pixels, same CV (current blocks, buffer, decision rule): NDVI only vs NDVI + SWIR features.

    Writes reports_dir/experiments/swir/; never touches the main reports or data/ml/az.json.
    """
    bcfg = ml.baseline
    space = label_space(crops)
    out_dir = bcfg.reports_dir / "experiments" / "swir"
    out_dir.mkdir(parents=True, exist_ok=True)
    stack_cfg = bcfg.model_copy(update={"features": ["ndvi", "swir"]})
    stack = build_stack(stack_cfg, crops, ml.ingest.bbox, ml.ingest.dates)
    print(f"dates used {len(stack.dates)}, usable px {int(usable_mask(stack, bcfg).sum())}", flush=True)

    rows: list[dict[str, Any]] = []
    labels_by_variant: dict[str, np.ndarray] = {}
    metrics_by_variant: dict[str, dict[str, Any]] = {}
    for features in _SWIR_VARIANTS:
        name = "+".join(features)
        cfg_v = bcfg.model_copy(update={"features": features})
        proba = spatial_cv_proba(stack, cfg_v, len(space.names))
        labels = decide(proba, stack, space, cfg_v)
        np.save(bcfg.output_dir / f"oof_proba_{name}_2021.npy", proba)
        labels_by_variant[name] = labels
        metrics_by_variant[name] = evaluate(stack, labels, space, cfg_v)
        row = _metrics_row(name, stack, labels, space, cfg_v, exp)
        row["n_features"] = n_features(cfg_v, len(stack.dates))
        rows.append(row)
        print(f"{name}: CV done", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "swir_vs_ndvi_2021.csv", index=False)
    with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format):
        print(table.set_index("variant").T.to_string())

    base, swir = (table.iloc[0], table.iloc[1])
    group = exp.threshold_group
    crops_cols = [f"f1_{c}_agri" for c in crop_groups(space, bcfg, exp)]
    drops = {c: float(base[c] - swir[c]) for c in crops_cols}
    precision = float(swir[f"{group}_precision_bbox"])
    ok = precision >= exp.preview_min_precision and all(
        d <= exp.preview_max_crop_drop for d in drops.values()
    )
    verdict: dict[str, Any] = {
        f"{group}_precision_bbox_swir": precision,
        "crop_f1_drop_vs_ndvi": drops,
        "preview_condition_met": ok,
    }
    if ok and ml.export is not None and ml.export.furrow is not None:
        furrow = ml.export.furrow
        name = "+".join(_SWIR_VARIANTS[1])
        class_metrics = pd.DataFrame(metrics_by_variant[name]["class_rows"]).set_index("group")
        blocked = blocked_groups(class_metrics, list(crops.groups), furrow.min_precision)
        lon, lat = pixel_lonlat(stack.grid)
        names = np.array(space.names)
        pixels = pd.DataFrame(
            {"lon": lon, "lat": lat, "in_bbox": stack.in_bbox, "dominant_crop": names[labels_by_variant[name]]}
        )
        groups = list(crops.groups) + [crops.other_group]
        cells, report = aggregate(pixels, ml.ingest.bbox, furrow.h3_resolution, crops.furrow_keys, groups, blocked)
        verdict["az_json_preview"] = {
            "cells": {k: v.model_dump() for k, v in cells.items()},
            "cells_written": report.written,
            "skipped_reasons": report.skipped_reasons,
            "cells_centre_outside_bbox": report.centre_outside_bbox,
            "groups_below_precision": report.groups_below_precision,
        }
    (out_dir / "swir_verdict_2021.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


# Feature sets compared by --history: current NDVI+SWIR model against the same plus previous-year cultivation.
_HISTORY_VARIANTS: list[list[str]] = [["ndvi", "swir"], ["ndvi", "swir", "history"]]


def _variant_row(
    variant: str,
    buffer_px: int,
    metrics: dict[str, Any],
    space: LabelSpace,
    cfg: BaselineConfig,
    exp: ExperimentConfig,
    n_dates: int,
) -> dict[str, Any]:
    by_group = {r["group"]: r for r in metrics["class_rows"]}
    group, other = exp.threshold_group, space.names[-1]
    row: dict[str, Any] = {"variant": variant, "buffer_px": buffer_px}
    for g in (group, other):
        row[f"{g}_precision_bbox"] = by_group[g]["precision_bbox"]
        row[f"{g}_recall_bbox"] = by_group[g]["recall_bbox"]
        row[f"{g}_f1_bbox"] = by_group[g]["f1_bbox"]
    for crop in crop_groups(space, cfg, exp):
        row[f"f1_{crop}_agri"] = by_group[crop]["f1_agri_truth"]
    row["macro_f1_rule_groups_agri"] = metrics["macro_f1_rule"]
    row["accuracy_bbox"] = metrics["accuracy_bbox"]
    row["n_features"] = n_features(cfg, n_dates)
    return row


def history_experiment(ml: MLConfig, crops: CropsConfig, exp: ExperimentConfig) -> None:
    """NDVI+SWIR vs NDVI+SWIR+history on the same pixels, for every buffer of buffer_sweep.

    Writes reports_dir/experiments/history/; never touches the main reports or data/ml/az.json.
    """
    bcfg = ml.baseline
    if not bcfg.buffer_sweep:
        raise ValueError("baseline.buffer_sweep пуст")
    space = label_space(crops)
    out_dir = bcfg.reports_dir / "experiments" / "history"
    out_dir.mkdir(parents=True, exist_ok=True)
    stack = build_stack(bcfg.model_copy(update={"features": _HISTORY_VARIANTS[1]}), crops, ml.ingest.bbox, ml.ingest.dates)
    print(f"dates used {len(stack.dates)}, usable px {int(usable_mask(stack, bcfg).sum())}", flush=True)
    history = stack.extra["history"]
    usable = usable_mask(stack, bcfg)
    hist_rows: list[dict[str, Any]] = []
    for k, name in enumerate(space.names):
        m = usable & (stack.labels == k)
        hist_rows.append(
            {
                "group_2021": name,
                "n": int(m.sum()),
                **{f"cultivated_{h.year}": float(np.nanmean(history[m, i])) for i, h in enumerate(
                    sorted(bcfg.history_cdl, key=lambda h: h.year))},
            }
        )
    pd.DataFrame(hist_rows).to_csv(out_dir / "history_by_label_2021.csv", index=False)

    rows: list[dict[str, Any]] = []
    main_labels: dict[str, np.ndarray] = {}
    main_metrics: dict[str, dict[str, Any]] = {}
    for features in _HISTORY_VARIANTS:
        name = "+".join(features)
        for buffer_px in bcfg.buffer_sweep:
            cfg_v = bcfg.model_copy(update={"features": features, "buffer_px": buffer_px})
            proba = spatial_cv_proba(stack, cfg_v, len(space.names))
            labels = decide(proba, stack, space, cfg_v)
            metrics = evaluate(stack, labels, space, cfg_v)
            rows.append(_variant_row(name, buffer_px, metrics, space, cfg_v, exp, len(stack.dates)))
            if buffer_px == bcfg.buffer_px:
                main_labels[name] = labels
                main_metrics[name] = metrics
            print(f"{name} buffer {buffer_px}: done", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "history_vs_swir_2021.csv", index=False)
    with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format):
        print(table.set_index(["variant", "buffer_px"]).T.to_string())

    base_name, hist_name = ("+".join(v) for v in _HISTORY_VARIANTS)
    group = exp.threshold_group
    largest = max(bcfg.buffer_sweep)
    crop_cols = [f"f1_{c}_agri" for c in crop_groups(space, bcfg, exp)]
    by = table.set_index(["variant", "buffer_px"])
    drops = {
        f"{c}@{b}": float(by.loc[(base_name, b), c] - by.loc[(hist_name, b), c])
        for b in bcfg.buffer_sweep
        for c in crop_cols
    }
    precision = float(by.loc[(hist_name, largest), f"{group}_precision_bbox"])
    ok = precision >= exp.preview_min_precision and all(d <= exp.preview_max_crop_drop for d in drops.values())
    verdict: dict[str, Any] = {
        f"{group}_precision_bbox_history_buffer{largest}": precision,
        "crop_f1_drop_vs_reference": drops,
        "preview_condition_met": ok,
        "applicability": "history требует CDL прошлых лет: только США; для регионов без CDL (vko) недоступен",
    }
    if ok and ml.export is not None and ml.export.furrow is not None:
        cfg_h = bcfg.model_copy(update={"features": _HISTORY_VARIANTS[1]})
        metrics = main_metrics[hist_name]
        class_metrics = pd.DataFrame(metrics["class_rows"]).set_index("group")
        lon, lat = pixel_lonlat(stack.grid)
        pixels = pd.DataFrame(
            {
                "lon": lon,
                "lat": lat,
                "in_bbox": stack.in_bbox,
                "dominant_crop": np.array(space.names)[main_labels[hist_name]],
            }
        )
        summary = {
            "macro_f1_agri_truth": metrics["macro_f1_agri"],
            "macro_f1_rule_agri_truth": metrics["macro_f1_rule"],
            "macro_f1_agri_bbox": metrics["macro_f1_agri_bbox"],
            "accuracy_bbox": metrics["accuracy_bbox"],
            "majority_baseline_bbox": metrics["majority_baseline_bbox"],
            "buffer_px": bcfg.buffer_px,
            "dates_used": len(stack.dates),
            "dates": len(stack.date_clear_share),
            "features_text": feature_text(cfg_h, len(stack.dates)),
            "decision_rule": decision_rule_text(cfg_h),
        }
        ml_h = ml.model_copy(update={"baseline": cfg_h})
        layer, report = build_layer(ml_h, crops, ml.export.furrow, summary, class_metrics, pixels)
        verdict["az_json_preview"] = json.loads(layer.model_dump_json(exclude_none=True))
        verdict["preview_report"] = {
            "cells_centre_outside_bbox": report.centre_outside_bbox,
            "skipped_reasons": report.skipped_reasons,
            "groups_below_precision": report.groups_below_precision,
        }
    (out_dir / "history_verdict_2021.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fallow rule, clear-date share and feature-set experiments.")
    parser.add_argument("config", type=Path, help="путь к configs/ml_bboxB.yaml")
    parser.add_argument("crops", type=Path, help="путь к configs/crops.yaml")
    parser.add_argument("--swir", action="store_true", help="только сравнение NDVI против NDVI+SWIR")
    parser.add_argument("--history", action="store_true", help="NDVI+SWIR против NDVI+SWIR+history, все буферы")
    args = parser.parse_args()
    ml = load_ml_config(args.config)
    crops = load_crops_config(args.crops)
    bcfg = ml.baseline
    exp = bcfg.experiments
    if exp is None:
        raise ValueError(f"{args.config}: нет baseline.experiments")
    if args.swir:
        swir_experiment(ml, crops, exp)
        return
    if args.history:
        history_experiment(ml, crops, exp)
        return
    if bcfg.min_date_clear_share not in exp.date_clear_share_grid:
        raise ValueError("текущий min_date_clear_share должен входить в date_clear_share_grid")
    space = label_space(crops)
    out_dir = bcfg.reports_dir / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    plain = bcfg.model_copy(update={"min_proba": {}, "desert_max_ndvi": None, "desert_excluded_groups": []})

    runs: dict[float, tuple[PixelStack, np.ndarray, BaselineConfig]] = {}
    for share in exp.date_clear_share_grid:
        cfg_s = plain.model_copy(update={"min_date_clear_share": share})
        stack = build_stack(cfg_s, crops, ml.ingest.bbox, ml.ingest.dates)
        print(f"share {share}: dates used {len(stack.dates)}, dropped {[d.isoformat() for d in stack.dropped_dates]}")
        proba = spatial_cv_proba(stack, cfg_s, len(space.names))
        np.save(bcfg.output_dir / f"oof_proba_share{share}_2021.npy", proba)
        runs[share] = (stack, proba, cfg_s)
        print(f"share {share}: CV done", flush=True)

    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.float_format", "{:.3f}".format):
        stack0, proba0, cfg0 = runs[bcfg.min_date_clear_share]
        table1 = fallow_experiment(stack0, proba0, space, cfg0, exp)
        table1.to_csv(out_dir / "exp1_fallow_2021.csv", index=False)
        print("\nEXP1 (share", bcfg.min_date_clear_share, ")")
        print(table1.set_index("variant").T.to_string())

        # Experiment 2: argmax per share, scored on pixels usable in every run so the sets are identical.
        common = np.logical_and.reduce([usable_mask(s, c) for s, _, c in runs.values()])
        groups = crop_groups(space, plain, exp) + [exp.threshold_group]
        rows2: list[dict[str, Any]] = []
        blocks: dict[float, pd.DataFrame] = {}
        for share, (stack, proba, cfg_s) in runs.items():
            labels = decide(proba, stack, space, cfg_s)
            agri = common & np.isin(stack.labels, space.agricultural)
            row: dict[str, Any] = {"date_clear_share": share, "dates_used": len(stack.dates), "n_common_px": int(common.sum())}
            for g in groups:
                k = space.names.index(g)
                row[f"f1_{g}_agri"] = float(f1_score(stack.labels[agri] == k, labels[agri] == k, zero_division=0))
            rule = rule_groups(space, cfg_s)
            row["macro_f1_rule_groups_agri"] = float(
                f1_score(stack.labels[agri], labels[agri], labels=rule, average="macro", zero_division=0)
            )
            rows2.append(row)
            blocks[share] = per_block_f1(stack, labels, agri, space, groups, cfg_s.spatial_blocks)
        table2 = pd.DataFrame(rows2)
        table2.to_csv(out_dir / "exp2_dates_2021.csv", index=False)
        print("\nEXP2")
        print(table2.set_index("date_clear_share").T.to_string())

        base = blocks[bcfg.min_date_clear_share]
        diffs: list[dict[str, Any]] = []
        for share, frame in blocks.items():
            if share == bcfg.min_date_clear_share:
                continue
            for g in groups:
                d = (frame[g] - base[g]).dropna()
                se = float(d.std(ddof=1) / np.sqrt(len(d)))
                diffs.append(
                    {
                        "date_clear_share": share,
                        "group": g,
                        "mean_block_diff": float(d.mean()),
                        "se": se,
                        "t": float(d.mean() / se) if se > 0 else np.nan,
                        "blocks_up": int((d > 0).sum()),
                        "blocks": len(d),
                    }
                )
        table3 = pd.DataFrame(diffs)
        table3.to_csv(out_dir / "exp2_block_diffs_2021.csv", index=False)
        print(f"\nEXP2 per-block F1 difference vs share {bcfg.min_date_clear_share} (9 blocks, paired)")
        print(table3.to_string(index=False))


if __name__ == "__main__":
    main()
