"""Pydantic v2 output contracts: FieldAnalysis, CropYear, BareSoil."""

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class CropYear(BaseModel):
    year: int
    crop: str
    prob: float = Field(ge=0, le=1)


class BareSoil(BaseModel):
    year: int
    days: int = Field(ge=0, le=366)
    ci_low: int = Field(ge=0, le=366)
    ci_high: int = Field(ge=0, le=366)

    @model_validator(mode="after")
    def _ci_order(self) -> "BareSoil":
        if not self.ci_low <= self.days <= self.ci_high:
            raise ValueError("ожидается ci_low <= days <= ci_high")
        return self


class FieldAnalysis(BaseModel):
    field_id: str
    region: str
    model_version: str
    crop_history: list[CropYear]
    bare_soil: list[BareSoil]
    domain_shift: bool
    data_gaps: list[str]


def sample_analysis() -> FieldAnalysis:
    """Mock output with model_version='sample' so it is never mistaken for real results."""
    return FieldAnalysis(
        field_id="SAMPLE-FIELD-001",
        region="SAMPLE-REGION",
        model_version="sample",
        crop_history=[
            CropYear(year=2021, crop="SAMPLE_CROP_A", prob=0.5),
            CropYear(year=2022, crop="SAMPLE_CROP_B", prob=0.5),
        ],
        bare_soil=[BareSoil(year=2021, days=30, ci_low=20, ci_high=40)],
        domain_shift=False,
        data_gaps=["SAMPLE_GAP"],
    )


def _load_export_paths(config_path: Path) -> tuple[Path, Path]:
    raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "export" not in raw:
        raise ValueError(f"в {config_path} нет секции export")
    export = raw["export"]
    return Path(export["schema_path"]), Path(export["sample_path"])


def export_contracts(config_path: Path) -> None:
    """Write schema.json and examples/sample_output.json with paths from config."""
    schema_path, sample_path = _load_export_paths(config_path)
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(
        json.dumps(FieldAnalysis.model_json_schema(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    sample_path.write_text(
        sample_analysis().model_dump_json(indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    export_contracts(Path("configs") / "ml.yaml")
