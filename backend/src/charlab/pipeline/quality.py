"""Quality gate: score each generated image against the reference character.

Checks, with limits from the settings:
- identity: CCIP difference from the reference (every image, faces or not)
- face: one found, and big enough (shots that should show a face)
- sharpness: the face's, relative to the reference face's
"""

import logging
from collections.abc import Callable

import numpy as np
from PIL import Image

from charlab.config import QualitySettings
from charlab.pipeline.faces import Face, FaceDetector
from charlab.pipeline.identity import Ccip
from charlab.pipeline.jobs import Check, Job

log = logging.getLogger(__name__)

# Margin around the face box for the sharpness crop, and the size it is measured at, so faces
# of different sizes are comparable.
FACE_MARGIN = 0.25
SHARPNESS_SIZE = 256


def sharpness(image: Image.Image, face: Face) -> float:
    """Variance of the Laplacian over the face: high for crisp edges, low for blur."""
    left, top, right, bottom = face.box
    margin = int(FACE_MARGIN * max(right - left, bottom - top))
    crop = image.crop(
        (
            max(0, left - margin),
            max(0, top - margin),
            min(image.width, right + margin),
            min(image.height, bottom + margin),
        )
    )
    grey = np.asarray(crop.convert("L").resize((SHARPNESS_SIZE,) * 2), dtype=np.float32)
    laplacian = (
        grey[:-2, 1:-1] + grey[2:, 1:-1] + grey[1:-1, :-2] + grey[1:-1, 2:] - 4 * grey[1:-1, 1:-1]
    )
    return float(laplacian.var())


class QualityGate:
    """The check stage. `start` scores the reference once; each call scores one image."""

    def __init__(
        self,
        settings: QualitySettings,
        detect: Callable[[Image.Image], list[Face]] | None = None,
        ccip: Ccip | None = None,
    ):
        self.settings = settings
        self.detect = detect or FaceDetector(settings)
        self.ccip = ccip or Ccip(settings)
        self._reference = None
        self._reference_sharpness: float | None = None

    def start(self, reference: Image.Image) -> None:
        self._reference = self.ccip.features(reference)
        faces = self.detect(reference)
        if faces:
            self._reference_sharpness = sharpness(reference, faces[0])
        else:
            log.warning("no face found in the reference; face sharpness will not be checked")

    def __call__(self, job: Job, image: Image.Image) -> Check:
        if self._reference is None:
            raise RuntimeError("call start(reference) before checking images")
        limits = self.settings
        identity = self.ccip.difference(self._reference, self.ccip.features(image))
        failures = []
        if identity >= limits.same_character_below:
            failures.append(f"CCIP difference {identity:.3f}: not the same character")
        if not job.face:
            return Check(identity, failures=tuple(failures))

        faces = self.detect(image)
        if not faces:
            return Check(identity, failures=(*failures, "no face found"))
        face, relative = faces[0], None
        if face.size < limits.min_face_px:
            failures.append(f"face {face.size} px, smaller than {limits.min_face_px} px")
        if self._reference_sharpness:
            relative = sharpness(image, face) / self._reference_sharpness
            if relative < limits.min_sharpness:
                failures.append(f"face sharpness {relative:.2f}x the reference's, too blurry")
        return Check(identity, face, relative, tuple(failures))
