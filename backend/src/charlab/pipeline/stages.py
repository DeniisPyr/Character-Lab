"""What each pipeline stage receives and returns.

Stages are plain callables, so a stage can be a function or an object that keeps a loaded
model. Tests pass fakes with the same signatures, which keeps CI free of GPUs and weights.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from PIL import Image

if TYPE_CHECKING:
    from charlab.pipeline.jobs import Check, Job, Repair


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


class Gate(Protocol):
    def start(self, reference: Image.Image) -> None:
        """Score the prepared reference, once per run."""

    def __call__(self, job: "Job", image: Image.Image) -> "Check":
        """Score one generated image against the reference."""


class FaceRepair(Protocol):
    def start(self, reference: Image.Image) -> None:
        """Prepare the face reference from the prepared reference, once per run."""

    def __call__(
        self, job: "Job", image: Image.Image, check: "Check", seed: int
    ) -> tuple["Repair", Image.Image] | None:
        """Repair the image's face if it needs it: the repair and the image to use."""


class Caption(Protocol):
    def __call__(self, image: Image.Image) -> str:
        """Describe the image for the training caption."""


@dataclass(frozen=True)
class Stages:
    prepare: Prepare
    generate: Generate
    # Without a caption stage, the results have no captions.
    caption: Caption | None = None
    # Without a quality gate, every image is kept as generated.
    check: Gate | None = None
    # Without a repair stage, faces are never redrawn.
    repair: FaceRepair | None = None
