import numpy as np
import pytest
from PIL import Image, ImageFilter

from charlab.config import load_settings
from charlab.pipeline.faces import Face
from charlab.pipeline.jobs import Job
from charlab.pipeline.quality import QualityGate, duplicates, fingerprint, sharpness

SETTINGS = load_settings().quality
FACE = Face((100, 100, 228, 228), 0.85)  # 128 px
JOB = Job("portrait/front", "A.", (512, 512), 1, ())
BACK = Job("turnaround/back", "B.", (512, 512), 2, (), face=False)


def checkerboard(size=512, cell=4):
    """Fine detail everywhere, so the face crop is sharp."""
    grid = (np.indices((size, size)).sum(axis=0) // cell) % 2
    return Image.fromarray((grid * 255).astype(np.uint8)).convert("RGB")


SHARP = checkerboard()
BLURRY = SHARP.filter(ImageFilter.GaussianBlur(6))


class FakeCcip:
    """Features are the image's mean brightness; the difference is how far apart they are."""

    def features(self, image):
        return float(np.asarray(image, dtype=np.float32).mean()) / 255

    def difference(self, a, b):
        return abs(a - b)


def gate(faces=(FACE,), reference_faces=(FACE,)):
    found = {"reference": list(reference_faces), "image": list(faces)}
    calls = []

    def detect(image):
        calls.append(image)
        return found["reference" if len(calls) == 1 else "image"]

    checker = QualityGate(SETTINGS, detect=detect, ccip=FakeCcip())
    checker.start(SHARP)
    return checker


def test_sharpness_is_higher_for_a_crisp_face():
    assert sharpness(SHARP, FACE) > 10 * sharpness(BLURRY, FACE)


def test_a_good_image_passes_with_its_scores():
    check = gate()(JOB, SHARP)

    assert check.passed
    assert check.identity == 0
    assert check.face == FACE
    assert check.sharpness == pytest.approx(1.0)


def test_another_character_fails_the_identity_check():
    check = gate()(JOB, Image.new("RGB", (512, 512), "black"))

    assert not check.passed
    assert "not the same character" in check.failures[0]


@pytest.mark.parametrize(
    ("faces", "image", "failure"),
    [
        ((), SHARP, "no face found"),
        ((Face((100, 100, 140, 140), 0.85),), SHARP, "face 40 px, smaller than 64 px"),
        ((FACE,), BLURRY, "too blurry"),
    ],
    ids=["no face", "small face", "blurry face"],
)
def test_face_checks(faces, image, failure):
    check = gate(faces=faces)(JOB, image)

    assert any(failure in reason for reason in check.failures)


def test_a_shot_from_behind_skips_the_face_checks():
    check = gate(faces=())(BACK, SHARP)

    assert check.passed
    assert check.face is None


def test_without_a_face_in_the_reference_sharpness_is_not_checked():
    check = gate(reference_faces=())(JOB, BLURRY)

    assert check.sharpness is None
    assert not any("blurry" in reason for reason in check.failures)


def test_images_cannot_be_checked_before_the_reference():
    checker = QualityGate(SETTINGS, detect=lambda image: [FACE], ccip=FakeCcip())

    with pytest.raises(RuntimeError, match="call start"):
        checker(JOB, SHARP)


def waves(size=256):
    """A smooth pattern with structure in both directions."""
    y, x = np.indices((size, size)) / size
    pixels = 127 + 127 * np.sin(6 * x) * np.cos(4 * y)
    return Image.fromarray(pixels.astype(np.uint8)).convert("RGB")


def test_fingerprint_is_64_bits_and_survives_resizing():
    image = waves()

    assert fingerprint(image).shape == (64,)
    assert (fingerprint(image) != fingerprint(image.resize((97, 97)))).sum() <= 2


def test_only_near_exact_copies_count_as_duplicates():
    pattern = waves()
    images = {"a": pattern, "b": pattern.copy(), "c": pattern.transpose(Image.FLIP_LEFT_RIGHT)}

    assert duplicates(images, max_bits=2) == [("a", "b", 0)]
