"""Score a finished dataset: is it one character, are the faces good, is it varied, and are the
captions ready for training?

It uses the quality gate's models (CCIP and the anime face detector) on the CPU, and works on
any folder of image/caption pairs, including one made by another tool.

Besides the gate's whole-image CCIP, each face is compared with the reference's face. The
whole image barely reacts when only the face changes: a heavily blurred face moved the
whole-image difference from 0.008 to 0.042, and the face's from 0.006 to 0.202.
"""

import logging
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from PIL import Image

from charlab.config import QualitySettings
from charlab.eval.dataset import Dataset
from charlab.pipeline.caption import needs_cleaning
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


def summarise(scores: Scores, same_character_below: float) -> dict[str, Any]:
    """The dataset in numbers: each score's mean and its worst image or pair, and counts.

    None stands for a score with nothing to measure, such as face sizes when no face was found.
    """
    images = scores.images
    names = [image.name for image in images]
    faces = {image.name: image.face.size for image in images if image.face}
    return {
        "images": len(images),
        "identity": {
            **_spread({image.name: image.identity for image in images}, max),
            "same_character": sum(image.identity < same_character_below for image in images),
        },
        "face_identity": _spread(_known(images, "face_identity"), max),
        "consistency": _pairs(scores.consistency, names, max),
        "faces": {"found": len(faces), "missing": [name for name in names if name not in faces]},
        "face_size": _spread(faces, min),
        "sharpness": _spread(_known(images, "sharpness"), min),
        # Different fingerprint bits: the closest pair is the most alike.
        "diversity": _pairs(scores.different_bits, names, min),
        "sizes": dict(Counter(f"{width}x{height}" for width, height in (i.size for i in images))),
        "captions": _captions(images, scores.orphans),
    }


def _known(images: tuple[ImageScore, ...], score: str) -> dict[str, float]:
    return {i.name: getattr(i, score) for i in images if getattr(i, score) is not None}


def _spread(
    values: dict[Any, float], worst: Callable, label: str = "worst_image"
) -> dict[str, Any] | None:
    """The mean, and the worst value with its image or pair; None when there are no values."""
    if not values:
        return None
    key = worst(values, key=values.__getitem__)
    mean = sum(values.values()) / len(values)
    return {"mean": round(mean, 4), "worst": round(values[key], 4), label: key}


def _pairs(matrix: np.ndarray, names: list[str], worst: Callable) -> dict[str, Any] | None:
    """`_spread` over every pair of images."""
    pairs = {(a, b): matrix[i, j].item() for (i, a), (j, b) in combinations(enumerate(names), 2)}
    spread = _spread(pairs, worst, label="worst_pair")
    return spread and {**spread, "worst_pair": list(spread["worst_pair"])}


def _captions(images: tuple[ImageScore, ...], orphans: tuple[str, ...]) -> dict[str, Any]:
    """What a trainer gets: is every image captioned, and does every caption start with the
    same trigger word, stay short enough and leave out the character's fixed look?"""
    captions = {image.name: image.caption for image in images if image.caption}
    first_words = Counter(re.split(r"[\s,.]", text, maxsplit=1)[0] for text in captions.values())
    trigger, with_trigger = first_words.most_common(1)[0] if first_words else (None, 0)
    words = [len(text.split()) for text in captions.values()]
    return {
        "captioned": len(captions),
        "missing": [image.name for image in images if not image.caption],
        "orphans": list(orphans),
        "trigger": trigger,
        "with_trigger": with_trigger,
        "words": {"min": min(words), "mean": round(sum(words) / len(words), 1), "max": max(words)}
        if words
        else None,
        "distinct": len(set(captions.values())),
        "need_cleaning": [name for name, text in captions.items() if needs_cleaning(text)],
    }
