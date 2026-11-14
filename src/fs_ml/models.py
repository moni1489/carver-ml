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
from fs_ml.features import LabelSpace, PixelStack, feature_matrix, min_clear_dates

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


def usable_mask(stack: PixelStack, cfg: BaselineConfig) -> np.ndarray:
    """Labelled pixels inside the lon/lat bbox with enough clear kept dates: used for training and metrics."""
    labeled = (stack.labels >= 0) & stack.in_bbox
    usable: np.ndarray = labeled & (stack.n_clear >= min_clear_dates(cfg.min_clear_fraction, len(stack.dates)))
    return usable


def block_index(shape: tuple[int, int], blocks: int) -> np.ndarray:
    """CV block id (0 .. blocks*blocks-1) of every pixel, row-major like the flattened stack."""
    height, width = shape
    out = np.empty(height * width, dtype=np.int16)
    rows = np.repeat(np.arange(height, dtype=np.int32), width)
    cols = np.tile(np.arange(width, dtype=np.int32), height)
    for index in range(blocks * blocks):
        r0, r1, c0, c1 = _block_bounds(index, shape, blocks)
        out[(rows >= r0) & (rows <= r1) & (cols >= c0) & (cols <= c1)] = index
    return out


def max_clear_ndvi(stack: PixelStack) -> np.ndarray:
    """Season maximum NDVI over clear kept dates; NaN for pixels with no clear date."""
    out = np.full(stack.ndvi.shape[0], np.nan, dtype=np.float32)
    has = stack.n_clear > 0
    out[has] = np.nanmax(stack.ndvi[has], axis=1)
    return out


def decide(proba: np.ndarray, stack: PixelStack, space: LabelSpace, cfg: BaselineConfig) -> np.ndarray:
    """Label per pixel from out-of-fold probabilities.

    A group with min_proba[group] = t is removed from the candidates of pixels where its probability is below t,
    so a pixel whose top class is that group falls to the next class by probability. Pixels with season max NDVI
    below desert_max_ndvi lose desert_excluded_groups the same way. Without both rules this is plain argmax.
    """
    unknown = [g for g in [*cfg.min_proba, *cfg.desert_excluded_groups] if g not in space.names]
    if unknown:
        raise ValueError(f"min_proba / desert_excluded_groups: нет таких групп: {unknown}")
    scores = proba.copy()
    for group, threshold in cfg.min_proba.items():
        k = space.names.index(group)
        scores[proba[:, k] < threshold, k] = -1.0
    if cfg.desert_max_ndvi is not None and cfg.desert_excluded_groups:
        peak = max_clear_ndvi(stack)
        desert = ~np.isnan(peak) & (peak < cfg.desert_max_ndvi)
        for group in cfg.desert_excluded_groups:
            scores[desert, space.names.index(group)] = -1.0
    labels: np.ndarray = scores.argmax(axis=1).astype(np.int16)
    return labels


def n_features(cfg: BaselineConfig, n_dates: int) -> int:
    """Model inputs per pixel: NDVI per date, plus SWIR1, SWIR2 and NDMI per date for features: swir."""
    return n_dates * (4 if "swir" in cfg.features else 1)


def feature_text(cfg: BaselineConfig, n_dates: int) -> str:
    layers = "NDVI+SWIR1+SWIR2+NDMI" if "swir" in cfg.features else "NDVI"
    return f"{layers} по {n_dates} датам ({n_features(cfg, n_dates)})"


def decision_rule_text(cfg: BaselineConfig) -> str:
    parts = [f"{g}: вероятность < {v} -> следующий класс" for g, v in cfg.min_proba.items()]
    if cfg.desert_max_ndvi is not None and cfg.desert_excluded_groups:
        parts.append(f"max NDVI < {cfg.desert_max_ndvi} запрещает {cfg.desert_excluded_groups}")
    return "; ".join(parts) if parts else "argmax"


def spatial_cv(
    stack: PixelStack, cfg: BaselineConfig, space: LabelSpace, diagnostics: list[dict[str, Any]] | None = None
) -> np.ndarray:
    """Out-of-fold labels: spatial_cv_proba followed by the decision rule from cfg (min_proba, desert mask)."""
    return decide(spatial_cv_proba(stack, cfg, len(space.names), diagnostics), stack, space, cfg)


def spatial_cv_proba(
    stack: PixelStack, cfg: BaselineConfig, n_classes: int, diagnostics: list[dict[str, Any]] | None = None
) -> np.ndarray:
    """Leave-one-block-out with a buffer: training pixels within buffer_px of the held-out block are dropped.

    Returns out-of-fold class probabilities (N, n_classes) for every pixel in the window.
    """
    height, width = stack.grid.shape
    rows = np.repeat(np.arange(height, dtype=np.int32), width)
    cols = np.tile(np.arange(width, dtype=np.int32), height)
    # NaN stays NaN: scikit-learn RandomForest routes missing values itself, nothing is imputed.
    features = feature_matrix(stack, list(cfg.features))
    usable = usable_mask(stack, cfg)
    proba = np.zeros((stack.labels.shape[0], n_classes), dtype=np.float32)
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
            proba[np.ix_(chunk, model.classes_.astype(np.int64))] = model.predict_proba(features[chunk])
        del model
        gc.collect()
    return proba


def rule_groups(space: LabelSpace, cfg: BaselineConfig) -> list[int]:
    """Agricultural label indices used by the buffer stop rule (agri groups minus buffer_rule_exclude_groups)."""
    unknown = [g for g in cfg.buffer_rule_exclude_groups if g not in space.names]
    if unknown:
        raise ValueError(f"buffer_rule_exclude_groups не найдены среди групп: {unknown}")
    excluded = {space.names.index(g) for g in cfg.buffer_rule_exclude_groups}
    return [k for k in space.agricultural if k not in excluded]


def evaluate(stack: PixelStack, oof: np.ndarray, space: LabelSpace, cfg: BaselineConfig) -> dict[str, Any]:
    scored = usable_mask(stack, cfg)
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
    macro_f1_agri_bbox = float(f1_score(y_all, p_all, labels=space.agricultural, average="macro", zero_division=0))
    rule = rule_groups(space, cfg)
    macro_f1_rule = float(f1_score(y_a, p_a, labels=rule, average="macro", zero_division=0))
    macro_f1_rule_bbox = float(f1_score(y_all, p_all, labels=rule, average="macro", zero_division=0))
    counts_bbox = np.bincount(y_all, minlength=n_classes)
    counts_no_other = counts_bbox.copy()
    counts_no_other[other] = -1
    majority_no_other = int(counts_no_other.argmax())
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
        "macro_f1_agri_bbox": macro_f1_agri_bbox,
        "rule_groups": [space.names[k] for k in rule],
        "macro_f1_rule": macro_f1_rule,
        "macro_f1_rule_bbox": macro_f1_rule_bbox,
        "majority_class_bbox": space.names[int(counts_bbox.argmax())],
        "majority_baseline_bbox": float(counts_bbox.max() / y_all.size),
        "majority_class_bbox_no_other": space.names[majority_no_other],
        "majority_baseline_bbox_no_other": float(counts_bbox[majority_no_other] / y_all.size),
        "majority_class_agri": space.names[int(np.bincount(y_a, minlength=n_classes).argmax())],
        "majority_baseline_agri": float(np.bincount(y_a, minlength=n_classes).max() / max(y_a.size, 1)),
        "other_share_bbox": float(np.mean(y_all == other)),
        "n_labeled": int(y_all.size),
        "n_agri_truth": int(y_a.size),
        "n_labeled_total": int(((stack.labels >= 0) & stack.in_bbox).sum()),
        "min_clear_dates": min_clear_dates(cfg.min_clear_fraction, len(stack.dates)),
        "validation": (
            f"пространственная блочная валидация {cfg.spatial_blocks}x{cfg.spatial_blocks} с буфером "
            f"{cfg.buffer_px} px; метки CDL 2021; признаки: {feature_text(cfg, len(stack.dates))}; "
            f"{len(stack.dates)} из {len(stack.date_clear_share)} дат HLS L30 v2.0 после исключения дат, где "
            f"чистых пикселей в bbox меньше {cfg.min_date_clear_share:.0%} (облако, тень, рядом с облаком: "
            "Fmask биты 1-3; B04 или B05 <= 0 тоже пропуск); пропуски не заполняются; только пиксели внутри "
            "lon/lat рамки bbox; обучение и метрики на пикселях с чистыми датами не меньше "
            f"{min_clear_dates(cfg.min_clear_fraction, len(stack.dates))} из {len(stack.dates)}; "
            "основная метрика на сельхозпикселях CDL, общая точность по bbox отдельно; "
            "предсказание для карты и сводной таблицы сделано для всех пикселей; "
            "не проверено на других регионах и годах"
        ),
    }


def write_reports(
    stack: PixelStack,
    oof: np.ndarray,
    space: LabelSpace,
    cfg: BaselineConfig,
    metrics: dict[str, Any],
    fingerprint: str,
) -> dict[str, Any]:
    reports = cfg.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics["class_rows"]).to_csv(reports / "class_metrics_2021.csv", index=False)
    pd.DataFrame(metrics["confusion_agri"], index=space.names, columns=space.names).to_csv(
        reports / "confusion_agri_2021.csv"
    )

    height, width = stack.grid.shape
    outside = ~stack.in_bbox.reshape(height, width)
    true_map = stack.labels.reshape(height, width).astype(float)
    pred_map = oof.reshape(height, width).astype(float)
    true_map[(true_map < 0) | outside] = np.nan
    pred_map[(pred_map < 0) | outside] = np.nan
    cmap = plt.get_cmap("tab20", len(space.names))
    figure, axes = plt.subplots(1, 2, figsize=(12, 6))
    image = axes[0].imshow(true_map, cmap=cmap, vmin=-0.5, vmax=len(space.names) - 0.5, interpolation="nearest")
    axes[0].set_title("CDL 2021 (groups)")
    axes[0].set_axis_off()
    axes[1].imshow(pred_map, cmap=cmap, vmin=-0.5, vmax=len(space.names) - 0.5, interpolation="nearest")
    axes[1].set_title(f"Predicted (spatial CV, buffer {cfg.buffer_px} px)")
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
        "macro_f1_agri_bbox": metrics["macro_f1_agri_bbox"],
        "majority_class_bbox_no_other": metrics["majority_class_bbox_no_other"],
        "majority_baseline_bbox_no_other": metrics["majority_baseline_bbox_no_other"],
        "buffer_px": cfg.buffer_px,
        "decision_rule": decision_rule_text(cfg),
        "min_proba": dict(cfg.min_proba),
        "config_sha256": fingerprint,
        "features": list(cfg.features),
        "features_text": feature_text(cfg, len(stack.dates)),
        "n_features": n_features(cfg, len(stack.dates)),
        "grid_clipped_to_tile_edge": stack.grid.clipped_to_raster,
        "min_clear_dates": metrics["min_clear_dates"],
        "n_grid_pixels": int(stack.labels.shape[0]),
        "n_labeled_in_bbox_pixels": metrics["n_labeled_total"],
        "n_usable_pixels": metrics["n_labeled"],
        "n_agri_truth_pixels": metrics["n_agri_truth"],
        "bare_obs_dates_meaning": (
            f"число дат из {len(stack.dates)} (даты наблюдений HLS, только чистые) с NDVI < bare_soil_ndvi, "
            "не дни; в сводке по пикселям колонка bare_obs_dates"
        ),
        "dates": len(stack.date_clear_share),
        "dates_used": len(stack.dates),
        "dates_dropped_count": len(stack.dropped_dates),
        "dates_dropped": [d.isoformat() for d in stack.dropped_dates],
        "min_date_clear_share": cfg.min_date_clear_share,
        "date_clear_share_in_bbox": {d.isoformat(): round(v, 4) for d, v in stack.date_clear_share.items()},
        "date_list": [d.isoformat() for d in stack.dates],
        "n_grid_pixels_in_bbox": int(stack.in_bbox.sum()),
        "rule_groups": metrics["rule_groups"],
        "macro_f1_rule_agri_truth": metrics["macro_f1_rule"],
        "grid_shape": [height, width],
        "validation": metrics["validation"],
    }
    (reports / "summary_2021.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def write_error_map(
    stack: PixelStack, oof: np.ndarray, space: LabelSpace, cfg: BaselineConfig, groups: list[str], path: Path
) -> dict[str, dict[str, int]]:
    """CDL and prediction side by side, plus one panel per group with its errors on scored (usable) pixels.

    Error panel colours: grey = not scored (too few clear dates or no CDL), white = no error for this group,
    red = false negative (CDL says the group, model says otherwise), blue = false positive.
    """
    unknown = [g for g in groups if g not in space.names]
    if unknown:
        raise ValueError(f"error_map_groups не найдены среди групп: {unknown}")
    height, width = stack.grid.shape
    scored = usable_mask(stack, cfg)
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
            Patch(color="#bdbdbd", label="not scored (too few clear dates / no CDL)"),
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
