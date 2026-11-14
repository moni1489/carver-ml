"""Тесты выходного контракта: CropYear, BareSoil, FieldAnalysis и экспортированная schema.json."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from fs_ml.contracts import BareSoil, CropYear, FieldAnalysis, sample_analysis

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "contracts" / "schema.json"


# --- CropYear -------------------------------------------------------------


@pytest.mark.parametrize("prob", [0.0, 0.5, 1.0])
def test_crop_year_accepts_prob_in_range(prob: float) -> None:
    item = CropYear(year=2021, crop="alfalfa", prob=prob)
    assert item.prob == prob


@pytest.mark.parametrize("prob", [-0.01, 1.01, -1.0, 2.0])
def test_crop_year_rejects_prob_out_of_range(prob: float) -> None:
    with pytest.raises(ValidationError):
        CropYear(year=2021, crop="alfalfa", prob=prob)


# --- BareSoil -------------------------------------------------------------


def test_bare_soil_accepts_ordered_interval() -> None:
    item = BareSoil(year=2021, days=30, ci_low=20, ci_high=40)
    assert item.ci_low <= item.days <= item.ci_high


def test_bare_soil_accepts_equal_bounds() -> None:
    BareSoil(year=2021, days=30, ci_low=30, ci_high=30)


@pytest.mark.parametrize(
    ("days", "ci_low", "ci_high"),
    [
        (10, 20, 40),  # days ниже ci_low
        (50, 20, 40),  # days выше ci_high
        (30, 35, 40),  # ci_low выше days
        (30, 20, 25),  # ci_high ниже days
    ],
)
def test_bare_soil_rejects_broken_order(days: int, ci_low: int, ci_high: int) -> None:
    with pytest.raises(ValidationError):
        BareSoil(year=2021, days=days, ci_low=ci_low, ci_high=ci_high)


# --- FieldAnalysis --------------------------------------------------------


def _analysis(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "field_id": "F-1",
        "region": "AZ",
        "model_version": "test",
        "crop_history": [],
        "bare_soil": [],
        "domain_shift": False,
        "data_gaps": [],
    }
    base.update(overrides)
    return base


def test_field_analysis_accepts_empty_lists() -> None:
    result = FieldAnalysis(**_analysis())
    assert result.crop_history == []
    assert result.bare_soil == []
    assert result.data_gaps == []


def test_field_analysis_accepts_filled_lists() -> None:
    result = FieldAnalysis(
        **_analysis(
            crop_history=[CropYear(year=2021, crop="cotton", prob=0.7)],
            bare_soil=[BareSoil(year=2021, days=10, ci_low=5, ci_high=15)],
            data_gaps=["2021-06-01: облачность"],
        )
    )
    assert len(result.crop_history) == 1
    assert len(result.bare_soil) == 1


def test_field_analysis_rejects_invalid_nested_prob() -> None:
    with pytest.raises(ValidationError):
        FieldAnalysis(
            **_analysis(crop_history=[{"year": 2021, "crop": "corn", "prob": 1.5}])
        )


# --- schema.json ----------------------------------------------------------


def test_schema_json_matches_model() -> None:
    """Экспортированная схема должна совпадать с текущей моделью FieldAnalysis.

    Если тест падает: запустите `make export` (python -m fs_ml.contracts) и закоммитьте обновлённый файл.
    """
    exported = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert exported == FieldAnalysis.model_json_schema()


def test_sample_analysis_is_valid_model() -> None:
    sample = sample_analysis()
    assert sample.model_version == "sample"
