"""Baseline RandomForest: spatial block cross-validation, metrics and report files."""

import gc
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from fs_ml.config import BaselineConfig
from fs_ml.features import LabelSpace, PixelStack

# Pixels per predict() call: bounds the peak memory of tree traversal; does not change predictions.
_PREDICT_CHUNK = 20000


def _block_bounds(index: int, shape: tuple[int, int], blocks: int) -> tuple[int, int, int, int]:
    height, width = shape
    block_row, block_col = divmod(index, blocks)
    r0 = block_row * height // blocks
    r1 = (block_row + 1) * height // blocks - 1
    c0 = block_col * width // blocks
    c1 = (block_col + 1) * width // blocks - 1
    return r0, r1, c0, c1


def _sample_per_class(labels: np.ndarray, mask: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    picks: list[np.ndarray] = []
    for value in np.unique(labels[mask]):
        idx = np.flatnonzero(mask & (labels == value))
        if idx.size > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        picks.append(idx)
    return np.concatenate(picks)


def clean_mask(stack: PixelStack) -> np.ndarray:
    """Pixels with a valid NDVI on every date: no cloud (Fmask bits 1-3) and no invalid DN in any scene."""
    labeled = stack.labels >= 0
    return labeled & np.all(~np.isnan(stack.ndvi), axis=1)


def spatial_cv(
    stack: PixelStack, cfg: BaselineConfig, diagnostics: list[dict[str, Any]] | None = None
) -> np.ndarray:
    """Leave-one-block-out with a buffer: training pixels within buffer_px of the held-out block are dropped.

    Returns the out-of-fold label index for every pixel in the window (-1 if unpredicted).
    """
    height, width = stack.grid.shape
    rows = np.repeat(np.arange(height, dtype=np.int32), width)
    cols = np.tile(np.arange(width, dtype=np.int32), height)
    features = np.where(np.isnan(stack.ndvi), cfg.missing_fill, stack.ndvi).astype(np.float32)
    usable = clean_mask(stack)
    oof = np.full(stack.labels.shape[0], -1, dtype=np.int16)
    rng = np.random.default_rng(cfg.random_state)
    buf = cfg.buffer_px
    for index in range(cfg.spatial_blocks * cfg.spatial_blocks):
        r0, r1, c0, c1 = _block_bounds(index, stack.grid.shape, cfg.spatial_blocks)
        in_block = (rows >= r0) & (rows <= r1) & (cols >= c0) & (cols <= c1)
        in_halo = (rows >= r0 - buf) & (rows <= r1 + buf) & (cols >= c0 - buf) & (cols <= c1 + buf)
        train_idx = _sample_per_class(stack.labels, usable & ~in_halo, cfg.max_train_per_class, rng)
        if diagnostics is not None:
            diagnostics.append(
                {
                    "block": index,
                    "n_train_candidates": int(np.sum(usable & ~in_halo)),
                    "n_train_sampled": int(train_idx.size),
                    "classes_in_train": int(np.unique(stack.labels[train_idx]).size),
                }
            )
        model = RandomForestClassifier(
            n_estimators=cfg.random_forest.n_estimators,
            n_jobs=cfg.random_forest.n_jobs,
            class_weight=cfg.random_forest.class_weight,
            random_state=cfg.random_state,
        )
        model.fit(features[train_idx], stack.labels[train_idx])
        test_idx = np.flatnonzero(in_block)
        for start in range(0, test_idx.size, _PREDICT_CHUNK):
            chunk = test_idx[start : start + _PREDICT_CHUNK]
            oof[chunk] = model.predict(features[chunk]).astype(np.int16)
        del model
        gc.collect()
    return oof


def evaluate(stack: PixelStack, oof: np.ndarray, space: LabelSpace, cfg: BaselineConfig) -> dict[str, Any]:
    scored = clean_mask(stack)
    y_all = stack.labels[scored].astype(np.int64)
    p_all = oof[scored].astype(np.int64)
    n_classes = len(space.names)
    other = n_classes - 1
    agri_truth = np.isin(y_all, space.agricultural)
    y_a = y_all[agri_truth]
    p_a = p_all[agri_truth]
    all_labels = list(range(n_classes))

    prec_b, rec_b, f1_b, sup_b = precision_recall_fscore_support(y_all, p_all, labels=all_labels, zero_division=0)
    prec_a, rec_a, f1_a, sup_a = precision_recall_fscore_support(y_a, p_a, labels=all_labels, zero_division=0)
    macro_f1_agri = float(f1_score(y_a, p_a, labels=space.agricultural, average="macro", zero_division=0))
    cm = confusion_matrix(y_a, p_a, labels=all_labels)

    class_rows: list[dict[str, Any]] = []
    for k, name in enumerate(space.names):
        class_rows.append(
            {
                "group": name,
                "agricultural": k in space.agricultural,
                "support_bbox": int(sup_b[k]),
                "precision_bbox": float(prec_b[k]),
                "recall_bbox": float(rec_b[k]),
                "f1_bbox": float(f1_b[k]),
                "support_agri_truth": int(sup_a[k]),
                "precision_agri_truth": float(prec_a[k]),
                "recall_agri_truth": float(rec_a[k]),
                "f1_agri_truth": float(f1_a[k]),
            }
        )
    return {
        "class_rows": class_rows,
        "confusion_agri": cm,
        "macro_f1_agri": macro_f1_agri,
        "accuracy_agri": float(accuracy_score(y_a, p_a)),
        "accuracy_bbox": float(accuracy_score(y_all, p_all)),
        "majority_class_bbox": space.names[int(np.bincount(y_all, minlength=n_classes).argmax())],
        "majority_baseline_bbox": float(np.bincount(y_all, minlength=n_classes).max() / y_all.size),
        "majority_class_agri": space.names[int(np.bincount(y_a, minlength=n_classes).argmax())],
        "majority_baseline_agri": float(np.bincount(y_a, minlength=n_classes).max() / max(y_a.size, 1)),
        "other_share_bbox": float(np.mean(y_all == other)),
        "n_labeled": int(y_all.size),
        "n_agri_truth": int(y_a.size),
        "n_labeled_total": int((stack.labels >= 0).sum()),
        "validation": (
            f"пространственная блочная валидация {cfg.spatial_blocks}x{cfg.spatial_blocks} с буфером "
            f"{cfg.buffer_px} px; метки CDL 2021; HLS L30 v2.0, {len(stack.dates)} дат; "
            "обучение и метрики только на пикселях без облака и без пропусков на всех датах "
            "(Fmask биты 1-3); основная метрика на сельхозпикселях CDL, общая точность по bbox отдельно; "
            "предсказание для карты и сводной таблицы сделано для всех пикселей; "
            "не проверено на других регионах и годах"
        ),
    }


def write_reports(
    stack: PixelStack, oof: np.ndarray, space: LabelSpace, cfg: BaselineConfig, metrics: dict[str, Any]
) -> dict[str, Any]:
    reports = cfg.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics["class_rows"]).to_csv(reports / "class_metrics_2021.csv", index=False)
    pd.DataFrame(metrics["confusion_agri"], index=space.names, columns=space.names).to_csv(
        reports / "confusion_agri_2021.csv"
    )

    height, width = stack.grid.shape
    true_map = stack.labels.reshape(height, width).astype(float)
    pred_map = oof.reshape(height, width).astype(float)
    true_map[true_map < 0] = np.nan
    pred_map[pred_map < 0] = np.nan
    cmap = plt.get_cmap("tab20", len(space.names))
    figure, axes = plt.subplots(1, 2, figsize=(12, 6))
    image = axes[0].imshow(true_map, cmap=cmap, vmin=-0.5, vmax=len(space.names) - 0.5, interpolation="nearest")
    axes[0].set_title("CDL 2021 (groups)")
    axes[0].set_axis_off()
    axes[1].imshow(pred_map, cmap=cmap, vmin=-0.5, vmax=len(space.names) - 0.5, interpolation="nearest")
    axes[1].set_title("Predicted (spatial CV)")
    axes[1].set_axis_off()
    colorbar = figure.colorbar(image, ax=list(axes), ticks=range(len(space.names)), fraction=0.02)
    colorbar.set_ticklabels(space.names)
    figure.savefig(reports / "pred_vs_cdl_2021.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    summary: dict[str, Any] = {
        "accuracy_agri_truth": metrics["accuracy_agri"],
        "accuracy_bbox": metrics["accuracy_bbox"],
        "majority_class_bbox": metrics["majority_class_bbox"],
        "majority_baseline_bbox": metrics["majority_baseline_bbox"],
        "majority_class_agri_truth": metrics["majority_class_agri"],
        "majority_baseline_agri_truth": metrics["majority_baseline_agri"],
        "other_share_bbox": metrics["other_share_bbox"],
        "macro_f1_agri_truth": metrics["macro_f1_agri"],
        "grid_clipped_to_tile_edge": stack.grid.clipped_to_raster,
        "n_labeled_total_pixels": metrics["n_labeled_total"],
        "n_clean_scored_pixels": metrics["n_labeled"],
        "n_agri_truth_pixels": metrics["n_agri_truth"],
        "bare_obs_dates_meaning": (
            "число дат из 8 (даты наблюдений HLS) с NDVI < bare_soil_ndvi, не дни; "
            "в сводке по пикселям колонка bare_obs_dates"
        ),
        "dates": [d.isoformat() for d in stack.dates],
        "grid_shape": [height, width],
        "validation": metrics["validation"],
    }
    (reports / "summary_2021.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def write_error_map(
    stack: PixelStack, oof: np.ndarray, space: LabelSpace, groups: list[str], path: Path
) -> dict[str, dict[str, int]]:
    """CDL and prediction side by side, plus one panel per group with its errors on scored (cloud-free) pixels.

    Error panel colours: grey = not scored (cloudy or no CDL), white = no error for this group,
    red = false negative (CDL says the group, model says otherwise), blue = false positive.
    """
    unknown = [g for g in groups if g not in space.names]
    if unknown:
        raise ValueError(f"error_map_groups не найдены среди групп: {unknown}")
    height, width = stack.grid.shape
    scored = clean_mask(stack)
    n_classes = len(space.names)
    true_map = np.where(stack.labels >= 0, stack.labels, np.nan).astype(float).reshape(height, width)
    pred_map = np.where(oof >= 0, oof, np.nan).astype(float).reshape(height, width)
    class_cmap = plt.get_cmap("tab20", n_classes)
    error_cmap = ListedColormap(["#bdbdbd", "#ffffff", "#d62728", "#1f77b4"])

    panels = 2 + len(groups)
    figure, axes = plt.subplots(1, panels, figsize=(5 * panels, 5.5))
    image = axes[0].imshow(true_map, cmap=class_cmap, vmin=-0.5, vmax=n_classes - 0.5, interpolation="nearest")
    axes[0].set_title("CDL 2021 (groups)")
    axes[1].imshow(pred_map, cmap=class_cmap, vmin=-0.5, vmax=n_classes - 0.5, interpolation="nearest")
    axes[1].set_title("Predicted (spatial CV)")
    colorbar = figure.colorbar(image, ax=[axes[0], axes[1]], ticks=range(n_classes), fraction=0.03, location="bottom")
    colorbar.set_ticklabels(space.names)

    counts: dict[str, dict[str, int]] = {}
    for axis, group in zip(axes[2:], groups, strict=True):
        k = space.names.index(group)
        errors = np.zeros(stack.labels.shape[0], dtype=np.int8)
        errors[scored] = 1
        false_neg = scored & (stack.labels == k) & (oof != k)
        false_pos = scored & (oof == k) & (stack.labels != k)
        errors[false_neg] = 2
        errors[false_pos] = 3
        axis.imshow(errors.reshape(height, width), cmap=error_cmap, vmin=-0.5, vmax=3.5, interpolation="nearest")
        axis.set_title(f"{group}: FN {int(false_neg.sum())} (red), FP {int(false_pos.sum())} (blue)")
        counts[group] = {"false_negative": int(false_neg.sum()), "false_positive": int(false_pos.sum())}
    for axis in axes:
        axis.set_axis_off()
    figure.legend(
        handles=[
            Patch(color="#bdbdbd", label="not scored (cloudy / no CDL)"),
            Patch(facecolor="#ffffff", edgecolor="#999999", label="no error for this group"),
            Patch(color="#d62728", label="false negative"),
            Patch(color="#1f77b4", label="false positive"),
        ],
        loc="upper right",
        ncol=4,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return counts
