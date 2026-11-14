"""Typed loaders for configs/ml.yaml and configs/crops.yaml."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from fs_ml.data import IngestConfig


class RandomForestParams(BaseModel):
    n_estimators: int = Field(gt=0)
    n_jobs: int
    class_weight: str | None


class BaselineConfig(BaseModel):
    cdl_path: Path
    hls_dir: Path
    output_dir: Path
    reports_dir: Path
    nir_band: str
    red_band: str
    fmask_band: str
    cloud_bits: list[int] = Field(min_length=1)
    missing_fill: float
    spatial_blocks: int = Field(ge=2)
    buffer_px: int = Field(ge=0)
    max_train_per_class: int = Field(gt=0)
    random_state: int
    random_forest: RandomForestParams
    buffer_sweep: list[int] = Field(default_factory=list)
    error_map_groups: list[str] = Field(default_factory=list)


class Thresholds(BaseModel):
    max_cloud_pct: float = Field(ge=0, le=100)
    bare_soil_ndvi: float = Field(ge=-1, le=1)


class MLConfig(BaseModel):
    ingest: IngestConfig
    thresholds: Thresholds
    baseline: BaselineConfig


class CropsConfig(BaseModel):
    other_group: str
    groups: dict[str, list[int]]
    agricultural_groups: list[str]
    export_merge: dict[str, list[str]] = Field(default_factory=dict)
    low_confidence_groups: list[str] = Field(default_factory=list)


def _read_yaml(path: Path) -> dict[str, Any]:
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: ожидается словарь верхнего уровня")
    return raw


def load_ml_config(path: Path) -> MLConfig:
    raw = _read_yaml(path)
    ingest_raw = raw.get("ingest")
    if not isinstance(ingest_raw, dict):
        raise TypeError(f"{path}: нет секции ingest")
    ingest_raw = {**ingest_raw, "max_cloud_pct": raw["thresholds"]["max_cloud_pct"]}
    return MLConfig.model_validate(
        {
            "ingest": ingest_raw,
            "thresholds": raw["thresholds"],
            "baseline": raw["baseline"],
        }
    )


def load_crops_config(path: Path) -> CropsConfig:
    return CropsConfig.model_validate(_read_yaml(path))
