"""Typed loaders for configs/ml.yaml and configs/crops.yaml."""

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from fs_ml.data import IngestConfig

FeatureSet = Literal["ndvi", "swir"]


def _default_features() -> list[FeatureSet]:
    return ["ndvi"]


class RandomForestParams(BaseModel):
    n_estimators: int = Field(gt=0)
    n_jobs: int
    class_weight: str | None


class CdlSource(BaseModel):
    """National CDL GeoTIFF read in place from its zip archive (GDAL /vsizip/), never fully extracted."""

    zip_path: Path
    member: str
    crs: str
    margin_px: int = Field(ge=0)

    def vsi_path(self) -> str:
        return f"/vsizip/{self.zip_path.as_posix()}/{self.member}"


class ExperimentConfig(BaseModel):
    """Grids for python -m fs_ml.experiments (fallow vs desert, clear-date share)."""

    threshold_group: str
    min_proba_grid: list[float] = Field(min_length=1)
    desert_max_ndvi_grid: list[float] = Field(min_length=1)
    date_clear_share_grid: list[float] = Field(min_length=1)
    # --swir: show the Furrow preview only if the threshold group reaches this bbox precision
    # and no crop F1 (agri pixels) drops by more than swir_preview_max_crop_drop against NDVI only.
    swir_preview_min_precision: float = Field(gt=0, le=1)
    swir_preview_max_crop_drop: float = Field(ge=0)


class BaselineConfig(BaseModel):
    cdl_path: Path
    cdl_source: CdlSource | None = None
    cdl_archives: list[Path] = Field(default_factory=list)
    hls_dir: Path
    output_dir: Path
    reports_dir: Path
    nir_band: str
    red_band: str
    fmask_band: str
    # Feature sets per kept date: ndvi; swir adds SWIR1, SWIR2 and NDMI = (NIR - SWIR1) / (NIR + SWIR1).
    features: list[FeatureSet] = Field(default_factory=_default_features)
    swir1_band: str | None = None
    swir2_band: str | None = None
    cloud_bits: list[int] = Field(min_length=1)
    min_date_clear_share: float = Field(gt=0, le=1)
    min_clear_fraction: float = Field(gt=0, le=1)
    spatial_blocks: int = Field(ge=2)
    buffer_px: int = Field(ge=0)
    max_train_per_class: int = Field(gt=0)
    random_state: int
    random_forest: RandomForestParams
    buffer_sweep: list[int] = Field(default_factory=list)
    buffer_max_f1_drop: float | None = Field(default=None, gt=0)
    buffer_rule_exclude_groups: list[str] = Field(default_factory=list)
    # Decision rule on out-of-fold probabilities: a group below its minimum probability gives way to the next class.
    min_proba: dict[str, float] = Field(default_factory=dict)
    # Pixels whose max NDVI over clear dates is below this cannot be predicted as desert_excluded_groups.
    desert_max_ndvi: float | None = None
    desert_excluded_groups: list[str] = Field(default_factory=list)
    experiments: ExperimentConfig | None = None
    error_map_groups: list[str] = Field(default_factory=list)


    @model_validator(mode="after")
    def _features_valid(self) -> "BaselineConfig":
        if "ndvi" not in self.features or len(set(self.features)) != len(self.features):
            raise ValueError(f"features должен содержать ndvi без повторов, сейчас {self.features}")
        if "swir" in self.features and (self.swir1_band is None or self.swir2_band is None):
            raise ValueError("для features: swir нужны swir1_band и swir2_band")
        return self


class Thresholds(BaseModel):
    max_cloud_pct: float = Field(ge=0, le=100)
    bare_soil_ndvi: float = Field(ge=-1, le=1)


class FurrowExport(BaseModel):
    path: Path
    h3_resolution: int = Field(ge=0, le=15)
    model: str
    version: str
    # Bbox precision a group needs to count as confident (precision <= min_precision = low confidence).
    min_precision: float = Field(gt=0, le=1)
    # composition: crop lists every predicted group with its share; dominant: one Furrow key per cell.
    crop_mode: Literal["composition", "dominant"] = "composition"


class ExportConfig(BaseModel):
    schema_path: Path
    sample_path: Path
    furrow: FurrowExport | None = None


class MLConfig(BaseModel):
    ingest: IngestConfig
    thresholds: Thresholds
    baseline: BaselineConfig
    export: ExportConfig | None = None


class CropsConfig(BaseModel):
    other_group: str
    groups: dict[str, list[int]]
    agricultural_groups: list[str]
    export_merge: dict[str, list[str]] = Field(default_factory=dict)
    low_confidence_groups: list[str] = Field(default_factory=list)
    furrow_keys: dict[str, str] = Field(default_factory=dict)


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
            "export": raw.get("export"),
        }
    )


def load_crops_config(path: Path) -> CropsConfig:
    return CropsConfig.model_validate(_read_yaml(path))


def baseline_fingerprint(ml: MLConfig, crops: CropsConfig) -> str:
    """sha256 of the settings that shape baseline outputs (ingest, baseline, crop groups).

    Taken from the parsed configs, so comments and key order do not change it. Left out: baseline.experiments
    (separate runs), the export section and crops.furrow_keys (read by the exporter itself).
    """
    excluded = {"experiments"}
    if ml.baseline.features == ["ndvi"]:
        # Default feature set: hash as before these fields existed, so NDVI-only reports stay valid.
        excluded |= {"features", "swir1_band", "swir2_band"}
    payload = {
        "ingest": ml.ingest.model_dump(mode="json"),
        "baseline": ml.baseline.model_dump(mode="json", exclude=excluded),
        "crops": crops.model_dump(mode="json", exclude={"furrow_keys"}),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
