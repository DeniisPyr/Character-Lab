"""Pipeline settings: which models to use and how to sample.

Defaults live in configs/default.yaml. A user config only needs the values it changes; it is
merged over the defaults key by key, and validated as a whole.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

# The package runs from a checkout (pip install -e), so the configs sit next to src/.
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
# Keys holding file paths, resolved relative to the YAML file that sets them.
PATH_KEYS = ("recipe",)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HubFile(_Model):
    """One file in a Hugging Face repo."""

    repo: str
    file: str


class GenerationSettings(_Model):
    model: str
    transformer: Literal["bf16", "gguf"]
    gguf: HubFile
    lightning_lora: HubFile
    steps: int = Field(ge=1)
    cfg: float = Field(ge=0)


class PrepareSettings(_Model):
    max_file_mb: float = Field(gt=0)
    max_pixels: int = Field(gt=0)
    min_side: int = Field(gt=0)
    background: Literal["auto", "remove", "keep"]
    background_model: str
    crop_margin: float = Field(ge=0, le=1)


class Settings(_Model):
    seed: int = Field(ge=0)
    recipe: Path
    prepare: PrepareSettings
    generation: GenerationSettings


def merge(base: dict, override: dict) -> dict:
    """Return `base` updated with `override`: nested dicts merge, other values replace."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for key in PATH_KEYS:
        if isinstance(data.get(key), str):
            data[key] = path.parent / data[key]
    return data


def load_settings(
    override: str | Path | None = None, defaults: str | Path = DEFAULT_CONFIG
) -> Settings:
    data = _read(Path(defaults))
    if override is not None:
        data = merge(data, _read(Path(override)))
    return Settings.model_validate(data)
