"""What each pipeline stage receives and returns.

Stages are plain callables, so a stage can be a function or an object that keeps a loaded
model. Tests pass fakes with the same signatures, which keeps CI free of GPUs and weights.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image


class StageUnavailable(RuntimeError):
    """A stage cannot run on this machine: a package or the GPU it needs is missing."""


class Prepare(Protocol):
    def __call__(self, reference: Path) -> Image.Image:
        """Validate the reference file and return the prepared reference image."""


class Generate(Protocol):
    def __call__(
        self, prompt: str, images: Sequence[Image.Image], size: tuple[int, int], seed: int
    ) -> Image.Image:
        """Edit `images` (Picture 1, 2, ...) into a new image of `size` (width, height)."""


class Caption(Protocol):
    def __call__(self, image: Image.Image) -> str:
        """Describe the image for the training caption."""


@dataclass(frozen=True)
class Stages:
    prepare: Prepare
    generate: Generate
    # Without a caption stage, the results have no captions.
    caption: Caption | None = None
