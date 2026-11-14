"""Bare months and crop rotation on bbox B: report and Furrow preview (never writes data/ml or the main reports).

2019 and 2020 come from USDA CDL (fact); 2021 is the model prediction from the current baseline run
(pixel_summary_2021.parquet). CDL 2021 is used only to check the prediction, never in the product columns.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fs_ml.config import load_crops_config, load_ml_config
from fs_ml.features import build_stack, label_space
from fs_ml.furrow import build_layer, check_baseline_matches
from fs_ml.inference import (
    BARE,
    NODATA_GROUP,
    UNKNOWN,
    history_groups,
    low_obs_months,
    pixel_summary,
    season_months,
)

_TOP = 10


def bare_shares(table: pd.DataFrame, group_column: str, months: list[int]) -> pd.DataFrame:
    """Share of bare months per group: bare pixels / pixels with observations in that month."""
    rows: list[dict[str, Any]] = []
    for group, frame in table.groupby(group_column):
        row: dict[str, Any] = {"group": group, "n": len(frame)}
        for month in months:
            status = frame[f"bare_status_{month:02d}"].to_numpy()
            known = status != UNKNOWN
            row[f"{month:02d}"] = float((status[known] == BARE).mean()) if known.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def top_sequences(table: pd.DataFrame, columns: list[str], top: int) -> pd.DataFrame:
    seq = table[columns].astype(str).agg(" → ".join, axis=1)
    counts = seq.value_counts(normalize=True)
    return pd.DataFrame({"sequence": counts.index[:top], "share": counts.to_numpy()[:top]})


def main() -> None:
    parser = argparse.ArgumentParser(description="Bare months, rotation report and Furrow preview with years/detail.")
    parser.add_argument("config", type=Path, help="путь к configs/ml_bboxB.yaml")
    parser.add_argument("crops", type=Path, help="путь к configs/crops.yaml")
    args = parser.parse_args()
    ml = load_ml_config(args.config)
    crops = load_crops_config(args.crops)
    bcfg = ml.baseline
    if ml.export is None or ml.export.furrow is None:
        raise ValueError(f"{args.config}: нет export.furrow")
    if not bcfg.history_cdl:
        raise ValueError(f"{args.config}: нет baseline.history_cdl (клипы CDL прошлых лет)")
    out_dir = bcfg.reports_dir / "experiments" / "rotation"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = json.loads((bcfg.reports_dir / "summary_2021.json").read_text(encoding="utf-8"))
    check_baseline_matches(summary, ml, crops)
    space = label_space(crops)
    stack = build_stack(bcfg.model_copy(update={"features": ["ndvi"]}), crops, ml.ingest.bbox, ml.ingest.dates)
    year = min(ml.ingest.dates).year

    predicted = pd.read_parquet(bcfg.output_dir / f"pixel_summary_{year}.parquet", columns=["row", "col", "dominant_crop"])
    height, width = stack.grid.shape
    expected_rows = np.repeat(np.arange(height), width)
    expected_cols = np.tile(np.arange(width), height)
    if not (np.array_equal(predicted["row"], expected_rows) and np.array_equal(predicted["col"], expected_cols)):
        raise RuntimeError("pixel_summary не совпадает с сеткой стека: перезапусти baseline")
    name_index = {n: i for i, n in enumerate(space.names)}
    oof = np.array([name_index.get(n, -1) for n in predicted["dominant_crop"]], dtype=np.int16)

    months = season_months(ml.ingest.temporal)
    low_obs = low_obs_months(stack.dates, months)
    previous = history_groups(stack.grid, bcfg.history_cdl, space)
    table = pixel_summary(
        stack,
        oof,
        space,
        ml.thresholds.bare_soil_ndvi,
        crops.export_merge,
        crops.low_confidence_groups,
        months=months,
        previous=previous,
        low_obs=list(low_obs),
    )
    # Check column only (CDL of the prediction year); not part of the product table.
    names = np.array(space.names + [NODATA_GROUP])
    check = names[np.where(stack.labels >= 0, stack.labels, len(space.names))]
    scope = table["in_bbox"].to_numpy() & (stack.labels >= 0)
    scoped = table.loc[scope].assign(**{f"cdl_check_{year}": check[scope]})

    kept_per_month = {f"{m:02d}": [d.isoformat() for d in stack.dates if d.month == m] for m in months}
    n_obs_mean = {f"{m:02d}": round(float(scoped[f"n_obs_{m:02d}"].mean()), 2) for m in months}

    by_cdl = bare_shares(scoped, f"cdl_check_{year}", months)
    by_pred = bare_shares(scoped, "dominant_crop", months)
    by_cdl.to_csv(out_dir / f"bare_month_share_by_cdl{year}.csv", index=False)
    by_pred.to_csv(out_dir / f"bare_month_share_by_pred{year}.csv", index=False)

    hist_cols = [f"cdl_group_{y}" for y in sorted(previous)]
    has_hist = np.logical_and.reduce([scoped[c].to_numpy() != NODATA_GROUP for c in hist_cols])
    seq_frame = scoped.loc[has_hist]
    product = top_sequences(seq_frame, [*hist_cols, "dominant_crop"], _TOP)
    prototype = top_sequences(seq_frame, [*hist_cols, f"cdl_check_{year}"], _TOP)
    product.to_csv(out_dir / f"top_sequences_product_{year}pred.csv", index=False)
    prototype.to_csv(out_dir / f"top_sequences_prototype_cdl{year}.csv", index=False)
    seq_p = seq_frame[[*hist_cols, "dominant_crop"]].astype(str).agg(" → ".join, axis=1).value_counts(normalize=True)
    seq_c = seq_frame[[*hist_cols, f"cdl_check_{year}"]].astype(str).agg(" → ".join, axis=1).value_counts(
        normalize=True
    )
    keys = list(dict.fromkeys([*product["sequence"], *prototype["sequence"]]))
    diff = pd.DataFrame(
        {
            "sequence": keys,
            f"share_product_{year}pred": [float(seq_p.get(k, 0.0)) for k in keys],
            f"share_prototype_cdl{year}": [float(seq_c.get(k, 0.0)) for k in keys],
        }
    )
    diff["diff"] = diff[f"share_product_{year}pred"] - diff[f"share_prototype_cdl{year}"]
    diff.to_csv(out_dir / f"sequences_product_vs_prototype_{year}.csv", index=False)

    class_metrics = pd.read_csv(bcfg.reports_dir / f"class_metrics_{year}.csv").set_index("group")
    layer, report = build_layer(ml, crops, ml.export.furrow, summary, class_metrics, table, low_obs)
    preview_path = out_dir / f"az_preview_{year}.json"
    preview_path.write_text(layer.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")
    n_detail = len(layer.detail.cells) if layer.detail is not None else 0

    meta = {
        "sources": {str(y): "USDA CDL (факт)" for y in sorted(previous)} | {str(year): "предсказание модели (2021*)"},
        "cdl_check_year_used_only_for_comparison": year,
        "kept_dates_per_month": kept_per_month,
        "low_obs_months": {f"{m:02d}": days for m, days in low_obs.items()},
        "mean_n_obs_per_month": n_obs_mean,
        "n_pixels_scope": int(scope.sum()),
        "n_pixels_with_history": int(has_hist.sum()),
        "prediction_agrees_with_cdl_check": float((scoped["dominant_crop"] == scoped[f"cdl_check_{year}"]).mean()),
        "preview": {
            "path": preview_path.as_posix(),
            "bytes": preview_path.stat().st_size,
            "cells_res": ml.export.furrow.h3_resolution,
            "n_cells": len(layer.cells),
            "detail_res": ml.export.furrow.detail_resolution,
            "n_detail_cells": n_detail,
            "skipped_res5": report.skipped_reasons,
        },
    }
    (out_dir / f"rotation_summary_{year}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    fmt = {"display.width": 250, "display.max_columns": 30, "display.float_format": "{:.3f}".format}
    with pd.option_context(*[x for kv in fmt.items() for x in kv]):
        print(f"kept dates per month: {kept_per_month}")
        print(f"low_obs months: {low_obs}")
        print(f"\nBARE MONTH SHARE by CDL {year} (check)\n{by_cdl.to_string(index=False)}")
        print(f"\nBARE MONTH SHARE by prediction {year}\n{by_pred.to_string(index=False)}")
        print(f"\nTOP {_TOP} product (CDL→CDL→pred)\n{product.to_string(index=False)}")
        print(f"\nTOP {_TOP} prototype (CDL→CDL→CDL)\n{prototype.to_string(index=False)}")
        print(f"\nDIFF\n{diff.to_string(index=False)}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
