"""Pipeline settings: which models to use and how to sample.

Defaults live in configs/default.yaml. A user config only needs the values it changes; it is
merged over the defaults key by key, and validated as a whole.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

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


class QualitySettings(_Model):
    face_detector: HubFile
    face_confidence: float = Field(gt=0, lt=1)
    ccip_features: HubFile
    ccip_metric: HubFile
    max_attempts: int = Field(ge=1)
    same_character_below: float = Field(gt=0)
    min_face_px: int = Field(ge=1)
    min_sharpness: float = Field(ge=0)
    duplicate_max_bits: int = Field(ge=0, le=64)


class FaceRepairSettings(_Model):
    mode: Literal["off", "failed", "small"]
    small_below_px: int = Field(ge=1)
    use_reference: bool
    min_layout_similarity: float = Field(gt=0, le=1)
    max_face_shift: float = Field(ge=0)
    identity_tolerance: float = Field(ge=0)

    @field_validator("mode", mode="before")
    @classmethod
    def _off(cls, value):
        # YAML reads a bare `off` as the boolean false.
        return "off" if value is False else value


class CaptionSettings(_Model):
    model: str
    task: Literal["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"]
    num_beams: int = Field(ge=1)
    max_new_tokens: int = Field(ge=1)


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
    style: Literal["anime"]
    prepare: PrepareSettings
    generation: GenerationSettings
    quality: QualitySettings
    face_repair: FaceRepairSettings
    caption: CaptionSettings


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
    override: str | Path | None = None,
    defaults: str | Path = DEFAULT_CONFIG,
    changes: dict | None = None,
) -> Settings:
    """The defaults, then the `override` file, then `changes` (a dict in the same shape)."""
    data = _read(Path(defaults))
    if override is not None:
        data = merge(data, _read(Path(override)))
    if changes:
        data = merge(data, changes)
    return Settings.model_validate(data)
