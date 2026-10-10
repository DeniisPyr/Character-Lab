"""Dataset recipes: which views and variants to generate, and with which prompts.

A recipe is data in YAML, so the dataset changes without code changes. Loading validates the
whole recipe, so a typo fails at startup instead of halfway through a GPU run.
"""

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

NAME = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)*$")
# Qwen-Image-Edit takes up to three input images: the reference plus two generated ones.
MAX_REFERENCES = 2
# Width and height must divide by 16 (8x VAE downsampling, then 2x2 patches).
SIZE_MULTIPLE = 16


def _check_names(names):
    bad = [name for name in names if not NAME.match(name)]
    if bad:
        raise ValueError(f"names must be lowercase snake_case, got {bad}")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Variant(_Model):
    instruction: str
    blocks: tuple[str, ...] = ()
    setting: str | None = None
    references: tuple[str, ...] = ()

    @field_validator("references")
    @classmethod
    def _check_references(cls, references):
        if len(references) > MAX_REFERENCES:
            raise ValueError(f"at most {MAX_REFERENCES} references, got {len(references)}")
        if len(set(references)) != len(references):
            raise ValueError(f"duplicate references: {list(references)}")
        return references


class View(_Model):
    size: tuple[int, int]
    variants: dict[str, Variant]

    @field_validator("size")
    @classmethod
    def _check_size(cls, size):
        if any(side <= 0 or side % SIZE_MULTIPLE for side in size):
            raise ValueError(
                f"width and height must be positive multiples of {SIZE_MULTIPLE}, got {size}"
            )
        return size

    @field_validator("variants")
    @classmethod
    def _check_variants(cls, variants):
        if not variants:
            raise ValueError("a view needs at least one variant")
        _check_names(variants)
        return variants


class Recipe(_Model):
    name: str
    identity: str
    blocks: dict[str, str] = {}
    views: dict[str, View]

    @field_validator("views")
    @classmethod
    def _check_views(cls, views):
        if not views:
            raise ValueError("a recipe needs at least one view")
        _check_names(views)
        return views

    @model_validator(mode="after")
    def _check_links(self):
        keys = {key for key, _ in self.variants()}
        for key, variant in self.variants():
            unknown = [block for block in variant.blocks if block not in self.blocks]
            if unknown:
                raise ValueError(f"{key}: unknown blocks {unknown}")
            for reference in variant.references:
                if reference == key:
                    raise ValueError(f"{key}: a variant cannot reference itself")
                if reference not in keys:
                    raise ValueError(f"{key}: unknown reference {reference!r}")
        self.generation_order()
        return self

    def variants(self) -> list[tuple[str, Variant]]:
        """All variants as ("view/variant", Variant) pairs, in file order."""
        return [
            (f"{view_name}/{name}", variant)
            for view_name, view in self.views.items()
            for name, variant in view.variants.items()
        ]

    def variant(self, key: str) -> Variant:
        view_name, name = key.split("/")
        return self.views[view_name].variants[name]

    def size(self, key: str) -> tuple[int, int]:
        return self.views[key.split("/")[0]].size

    def prompt(self, key: str, description: str | None = None) -> str:
        """Instruction, blocks, setting, the optional description, then the identity line."""
        variant = self.variant(key)
        parts = [variant.instruction, *(self.blocks[block] for block in variant.blocks)]
        if variant.setting:
            parts.append(variant.setting)
        if description:
            parts.append(description.strip())
        parts.append(self.identity)
        return " ".join(parts)

    def generation_order(self) -> list[str]:
        """Variant keys with each reference before the variants that use it, else file order."""
        pending = dict(self.variants())
        done: list[str] = []
        while pending:
            ready = next(
                (
                    key
                    for key, variant in pending.items()
                    if all(reference in done for reference in variant.references)
                ),
                None,
            )
            if ready is None:
                raise ValueError(f"references form a cycle among {sorted(pending)}")
            done.append(ready)
            del pending[ready]
        return done


def load_recipe(path: str | Path) -> Recipe:
    with open(path, encoding="utf-8") as f:
        return Recipe.model_validate(yaml.safe_load(f))
