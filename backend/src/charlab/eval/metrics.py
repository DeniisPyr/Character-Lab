"""Score a finished dataset against the reference: is it one character, and are the faces good
and varied?

It uses the quality gate's models (CCIP and the anime face detector) on the CPU, and works on
any folder of image/caption pairs, including one made by another tool.

Besides the gate's whole-image CCIP, each face is compared with the reference's face. The
whole image barely reacts when only the face changes: a heavily blurred face moved the
whole-image difference from 0.008 to 0.042, and the face's from 0.006 to 0.202.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image

from charlab.config import QualitySettings
from charlab.eval.dataset import Dataset
from charlab.pipeline.faces import Face, FaceDetector
from charlab.pipeline.identity import Ccip
from charlab.pipeline.quality import face_crop, fingerprint, sharpness

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageScore:
    name: str
    size: tuple[int, int]
    identity: float  # CCIP difference from the reference
    face: Face | None
    face_identity: float | None  # CCIP difference between the face and the reference's face
    sharpness: float | None  # the face's, as a share of the reference face's
    caption: str | None


@dataclass(frozen=True)
class Scores:
    images: tuple[ImageScore, ...]
    consistency: np.ndarray  # CCIP differences between the dataset's images, (N, N)
    different_bits: np.ndarray  # how many of the 64 fingerprint bits differ, (N, N)
    orphans: tuple[str, ...]  # caption files without an image


class Scorer:
    """Scores a dataset; the models load on first use."""

    def __init__(
        self,
        settings: QualitySettings,
        detect: Callable[[Image.Image], list[Face]] | None = None,
        ccip: Ccip | None = None,
    ):
        self.detect = detect or FaceDetector(settings)
        self.ccip = ccip or Ccip(settings)

    def __call__(self, reference: Image.Image, dataset: Dataset) -> Scores:
        entries = dataset.entries
        features = [self.ccip.features(entry.image) for entry in entries]
        whole = self.ccip.differences([self.ccip.features(reference), *features])
        faces = [next(iter(self.detect(entry.image)), None) for entry in entries]
        face_identity, relative = self._faces(reference, [e.image for e in entries], faces)

        prints = np.array([fingerprint(entry.image) for entry in entries])
        images = tuple(
            ImageScore(
                entry.name,
                entry.image.size,
                float(whole[0, i + 1]),
                faces[i],
                face_identity[i],
                relative[i],
                entry.caption,
            )
            for i, entry in enumerate(entries)
        )
        return Scores(
            images,
            consistency=whole[1:, 1:],
            different_bits=(prints[:, None] != prints[None, :]).sum(axis=2),
            orphans=dataset.orphans,
        )

    def _faces(
        self, reference: Image.Image, images: list[Image.Image], faces: list[Face | None]
    ) -> tuple[list[float | None], list[float | None]]:
        """Each face's CCIP difference from the reference's face, and its relative sharpness."""
        identity: list[float | None] = [None] * len(images)
        relative: list[float | None] = [None] * len(images)
        reference_faces = self.detect(reference)
        if not reference_faces:
            log.warning("no face found in the reference; faces are not compared with it")
            return identity, relative

        reference_face = reference_faces[0]
        found = [i for i, face in enumerate(faces) if face]
        crops = [face_crop(images[i], faces[i]) for i in found]
        matrix = self.ccip.differences(
            [self.ccip.features(face_crop(reference, reference_face))]
            + [self.ccip.features(crop) for crop in crops]
        )
        reference_sharpness = sharpness(reference, reference_face)
        for k, i in enumerate(found):
            identity[i] = float(matrix[0, k + 1])
            if reference_sharpness:
                relative[i] = sharpness(images[i], faces[i]) / reference_sharpness
        return identity, relative
