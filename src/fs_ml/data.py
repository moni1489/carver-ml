"""Data ingestion: NASA HLS search by dates, size-guarded download and lazy band access."""

import argparse
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import earthaccess
import yaml
from earthaccess.exceptions import LoginStrategyUnavailable
from pydantic import BaseModel, Field

_GRANULE_DATE = re.compile(r"\.(\d{4})(\d{3})T\d{6}\.")


class IngestConfig(BaseModel):
    short_name: str
    version: str
    bbox: tuple[float, float, float, float]
    temporal: tuple[str, str]
    count: int = Field(gt=0)
    dates: list[date] = Field(min_length=1)
    max_cloud_pct: float = Field(ge=0, le=100)
    max_download_mb: float = Field(gt=0)
    local_path: Path


def load_ingest_config(config_path: Path) -> IngestConfig:
    raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "ingest" not in raw or "thresholds" not in raw:
        raise ValueError(f"в {config_path} нужны секции ingest и thresholds")
    merged: dict[str, Any] = {**raw["ingest"], "max_cloud_pct": raw["thresholds"]["max_cloud_pct"]}
    return IngestConfig.model_validate(merged)


def login() -> None:
    """Log in via the existing .netrc file. Interactive login is done once outside this module."""
    try:
        auth = earthaccess.login(strategy="netrc")
    except LoginStrategyUnavailable as exc:
        raise RuntimeError("Earthdata: .netrc не найден, выполните логин один раз") from exc
    if not auth.authenticated:
        raise RuntimeError("Earthdata: логин не прошёл, проверьте .netrc")


def search_granules(cfg: IngestConfig) -> list[Any]:
    """Search by bbox and dates; CMR pre-filters scenes by granule-level cloud cover."""
    results = earthaccess.search_data(
        short_name=cfg.short_name,
        version=cfg.version,
        bounding_box=cfg.bbox,
        temporal=cfg.temporal,
        cloud_cover=(0, cfg.max_cloud_pct),
        count=cfg.count,
    )
    return list(results)


def granule_date(granule: Any) -> date:
    match = _GRANULE_DATE.search(str(granule["meta"]["native-id"]))
    if match is None:
        raise ValueError(f"не удалось извлечь дату из {granule['meta']['native-id']}")
    return datetime.strptime(f"{match.group(1)}{match.group(2)}", "%Y%j").date()


def select_by_dates(granules: list[Any], wanted: list[date]) -> list[Any]:
    """Pick one granule per wanted date. Raises if a wanted date has no scene."""
    by_date: dict[date, Any] = {}
    for g in granules:
        by_date.setdefault(granule_date(g), g)
    missing = [d.isoformat() for d in wanted if d not in by_date]
    if missing:
        raise RuntimeError(f"нет сцен с облачностью <= порога на даты: {missing}")
    return [by_date[d] for d in sorted(wanted)]


def check_size(granules: list[Any], max_mb: float) -> float:
    """Return total size in MB; refuse if it exceeds the configured limit."""
    total_mb = sum(float(g.size) for g in granules)
    if total_mb > max_mb:
        raise RuntimeError(f"выбрано {total_mb:.1f} МБ, лимит {max_mb:.1f} МБ")
    return total_mb


def download(granules: list[Any], local_path: Path) -> list[Path]:
    local_path.mkdir(parents=True, exist_ok=True)
    files = earthaccess.download(granules, local_path=str(local_path))
    return [Path(str(f)) for f in files]


def list_tifs(root: Path) -> list[Path]:
    return sorted(root.rglob("*.tif"))


def open_band(path: Path) -> Any:
    """Open one band lazily. Pixel data is read only when accessed, windowed or per band."""
    import rioxarray

    return rioxarray.open_rasterio(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search, download and inspect HLS scenes.")
    parser.add_argument("config", type=Path, help="путь к configs/ml.yaml")
    parser.add_argument("--download", action="store_true", help="без флага только список дат и размер")
    args = parser.parse_args()

    cfg = load_ingest_config(args.config)
    login()
    granules = search_granules(cfg)
    print(f"candidates (cloud <= {cfg.max_cloud_pct}%): {len(granules)}")
    chosen = select_by_dates(granules, cfg.dates)
    for g in chosen:
        print(f"{granule_date(g).isoformat()}  {g['meta']['native-id']}  {float(g.size):.1f} MB")
    total_mb = check_size(chosen, cfg.max_download_mb)
    print(f"total: {len(chosen)} scenes, {total_mb:.1f} MB, target {cfg.local_path}")

    if not args.download:
        return

    files = download(chosen, cfg.local_path)
    tifs = sorted(f for f in files if f.suffix == ".tif")
    print(f"downloaded files: {len(files)}, tifs: {len(tifs)}")


if __name__ == "__main__":
    main()
