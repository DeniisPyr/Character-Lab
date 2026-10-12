import logging

import numpy as np
import pytest
from PIL import Image, ImageFilter

from charlab.config import load_settings
from charlab.eval.dataset import InvalidDataset, load_dataset
from charlab.eval.metrics import ImageScore, Scorer, Scores, summarise
from charlab.pipeline.faces import Face

SETTINGS = load_settings().quality
FACE = Face((100, 100, 228, 228), 0.85)


def checkerboard(size=512, cell=4):
    grid = (np.indices((size, size)).sum(axis=0) // cell) % 2
    return Image.fromarray((grid * 255).astype(np.uint8)).convert("RGB")


SHARP = checkerboard()


class FakeCcip:
    """Features are the image's mean brightness; differences are how far apart they are."""

    def __init__(self):
        self.seen = []

    def features(self, image):
        self.seen.append(image.size)
        return np.array([np.asarray(image, dtype=np.float32).mean() / 255])

    def differences(self, features):
        values = np.array(features)[:, 0]
        return np.abs(values[:, None] - values[None, :])


def write(folder, name, image, caption=None):
    image.save(folder / name)
    if caption is not None:
        (folder / name).with_suffix(".txt").write_text(caption, encoding="utf-8")


def test_images_are_paired_with_their_captions(tmp_path):
    write(tmp_path, "002_b.png", Image.new("RGB", (64, 32)), "mychar, side view.\n")
    write(tmp_path, "001_a.jpg", Image.new("L", (32, 32)), "mychar, front view.")
    write(tmp_path, "003_c.png", Image.new("RGB", (32, 32)), "")
    write(tmp_path, "004_d.webp", Image.new("RGB", (32, 32)))
    (tmp_path / "005_e.txt").write_text("a caption without an image", encoding="utf-8")
    (tmp_path / "._001_a.jpg").write_bytes(b"macOS metadata")
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")

    dataset = load_dataset(tmp_path)

    names = [entry.name for entry in dataset.entries]
    assert names == ["001_a.jpg", "002_b.png", "003_c.png", "004_d.webp"]
    assert [e.caption for e in dataset.entries] == [
        "mychar, front view.",
        "mychar, side view.",
        None,
        None,
    ]
    assert dataset.entries[0].image.mode == "RGB"
    assert dataset.entries[1].image.size == (64, 32)
    assert dataset.orphans == ("005_e.txt",)


def test_unusable_datasets_are_rejected(tmp_path):
    with pytest.raises(InvalidDataset, match="not a folder"):
        load_dataset(tmp_path / "missing")
    with pytest.raises(InvalidDataset, match="no images found"):
        load_dataset(tmp_path)
    (tmp_path / "a.png").write_bytes(b"not a png")
    with pytest.raises(InvalidDataset, match="cannot be read"):
        load_dataset(tmp_path)


def score(tmp_path, images, faces, reference_faces=(FACE,)):
    """Score `images` (name: image) against SHARP, finding `faces` (name: face or None)."""
    for name, image in images.items():
        write(tmp_path, name, image, f"mychar, {name}.")
    found = {id(SHARP): list(reference_faces)}

    def detect(image):
        if image is SHARP:
            return found[id(SHARP)]
        name = next(n for n, e in zip(images, dataset.entries, strict=True) if e.image is image)
        return [faces[name]] if faces[name] else []

    dataset = load_dataset(tmp_path)
    return Scorer(SETTINGS, detect=detect, ccip=FakeCcip())(SHARP, dataset)


def test_each_image_is_scored_against_the_reference(tmp_path):
    blurry = SHARP.filter(ImageFilter.GaussianBlur(6))
    images = {"a.png": SHARP, "b.png": blurry, "c.png": Image.new("RGB", (512, 512), "white")}

    scores = score(tmp_path, images, {"a.png": FACE, "b.png": FACE, "c.png": None})

    a, b, c = scores.images
    assert (a.name, a.size, a.caption) == ("a.png", (512, 512), "mychar, a.png.")
    assert a.identity == 0 and a.face_identity == 0
    assert a.face == FACE and a.sharpness == pytest.approx(1.0)
    assert b.sharpness < 0.1
    assert c.identity == pytest.approx(0.5, abs=0.01)
    assert c.face is c.face_identity is c.sharpness is None


def test_pairs_of_images_are_compared_with_each_other(tmp_path):
    images = {
        "a.png": SHARP,
        "b.png": SHARP.copy(),
        "c.png": Image.new("RGB", (512, 512), "white"),
    }

    scores = score(tmp_path, images, dict.fromkeys(images, FACE))

    assert scores.consistency.shape == scores.different_bits.shape == (3, 3)
    assert scores.consistency[0, 1] == 0
    assert scores.consistency[0, 2] == pytest.approx(0.5, abs=0.01)
    assert scores.different_bits[0, 1] == 0
    assert np.all(np.diag(scores.different_bits) == 0)
    assert scores.different_bits[0, 2] == scores.different_bits[2, 0] > 0


def test_faces_are_compared_by_their_crops(tmp_path):
    # The same image, but only the face region is white: the whole image barely changes,
    # the face crop does.
    face_changed = SHARP.copy()
    face_changed.paste((255, 255, 255), FACE.box)

    scores = score(tmp_path, {"a.png": face_changed}, {"a.png": FACE})

    (image,) = scores.images
    assert image.face_identity > 5 * image.identity


def test_without_a_face_in_the_reference_faces_are_not_compared(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        scores = score(tmp_path, {"a.png": SHARP}, {"a.png": FACE}, reference_faces=())

    (image,) = scores.images
    assert image.face == FACE
    assert image.face_identity is image.sharpness is None
    assert "no face found in the reference" in caplog.text


def image(
    name, identity, face=None, face_identity=None, sharpness=None, caption=None, size=(64, 64)
):
    return ImageScore(name, size, identity, face, face_identity, sharpness, caption)


def test_summary_gives_each_score_its_mean_and_worst_image():
    small = Face((0, 0, 40, 50), 0.9)
    images = (
        image("a.png", 0.01, FACE, 0.02, 1.0, "mychar, front view. She smiles."),
        image("b.png", 0.05, small, 0.06, 0.2, "mychar, side view. She has long dark hair."),
        image("c.png", 0.30, size=(32, 64), caption="other, back view."),
    )
    consistency = np.array([[0, 0.02, 0.4], [0.02, 0, 0.3], [0.4, 0.3, 0]])
    bits = np.array([[0, 5, 30], [5, 0, 20], [30, 20, 0]])

    summary = summarise(Scores(images, consistency, bits, ("d.txt",)), same_character_below=0.18)

    assert summary["images"] == 3
    assert summary["identity"] == {
        "mean": 0.12,
        "worst": 0.3,
        "worst_image": "c.png",
        "same_character": 2,
    }
    assert summary["face_identity"] == {"mean": 0.04, "worst": 0.06, "worst_image": "b.png"}
    assert summary["faces"] == {"found": 2, "missing": ["c.png"]}
    assert summary["face_size"] == {"mean": 84.0, "worst": 40, "worst_image": "b.png"}
    assert summary["sharpness"]["worst_image"] == "b.png"
    assert summary["consistency"] == {"mean": 0.24, "worst": 0.4, "worst_pair": ["a.png", "c.png"]}
    assert summary["diversity"] == {"mean": 18.3333, "worst": 5, "worst_pair": ["a.png", "b.png"]}
    assert summary["sizes"] == {"64x64": 2, "32x64": 1}


def test_summary_checks_the_captions_a_trainer_gets():
    images = (
        image("a.png", 0, caption="mychar, front view. She smiles."),
        image("b.png", 0, caption="mychar, side view. She has long dark hair."),
        image("c.png", 0, caption="mychar. She smiles."),
        image("d.png", 0, caption="other, back view."),
        image("e.png", 0),
    )
    scores = Scores(images, np.zeros((5, 5)), np.zeros((5, 5), dtype=int), ("f.txt",))

    captions = summarise(scores, same_character_below=0.18)["captions"]

    assert captions == {
        "captioned": 4,
        "missing": ["e.png"],
        "orphans": ["f.txt"],
        "trigger": "mychar",
        "with_trigger": 3,
        "words": {"min": 3, "mean": 4.8, "max": 8},
        "distinct": 4,
        "need_cleaning": ["b.png"],
    }


def test_a_single_image_without_a_face_or_caption_has_nothing_to_compare():
    scores = Scores((image("a.png", 0.01),), np.zeros((1, 1)), np.zeros((1, 1), dtype=int), ())

    summary = summarise(scores, same_character_below=0.18)

    assert summary["consistency"] is summary["diversity"] is None
    assert summary["face_identity"] is summary["face_size"] is summary["sharpness"] is None
    assert summary["captions"]["words"] is None
    assert summary["captions"]["trigger"] is None
