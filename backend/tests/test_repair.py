import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from charlab.config import load_settings
from charlab.pipeline import repair as repair_module
from charlab.pipeline.faces import Face
from charlab.pipeline.jobs import Check, Job
from charlab.pipeline.repair import (
    PROMPT,
    REFERENCE_PROMPT,
    REPAIR_SEED_STEP,
    REPAIR_SIZE,
    FaceRepairer,
    FaceRepairStage,
    blend,
    crop_box,
    keep_repair,
    layout_similarity,
    oval_mask,
)

FACE = Face((400, 300, 500, 400), 0.85)  # 100 px, centred at (450, 350)
IMAGE = Image.new("RGB", (1280, 720), "white")


def test_crop_is_a_square_around_the_face():
    box = crop_box(FACE, IMAGE)

    assert box == (340, 240, 560, 460)  # 2.2 x 100 px, same centre


@pytest.mark.parametrize(
    ("face", "box"),
    [
        (Face((0, 0, 100, 100), 0.9), (0, 0, 220, 220)),  # moved inside at the top left
        (Face((1240, 680, 1280, 720), 0.9), (1152, 592, 1280, 720)),  # bottom right, 128 minimum
        (Face((100, 0, 900, 700), 0.9), (140, 0, 860, 720)),  # no larger than the image
    ],
    ids=["top-left corner", "small face at the edge", "face bigger than the crop"],
)
def test_crop_stays_inside_the_image(face, box):
    assert crop_box(face, IMAGE) == box


def test_oval_covers_the_face_and_fades_to_nothing():
    box = crop_box(FACE, IMAGE)
    mask = oval_mask(FACE, box, grow=0.35, feather=0.04)

    assert mask.size == (220, 220)
    assert mask.getpixel((110, 110)) == 255
    assert mask.getpixel((0, 0)) == 0


def test_blend_changes_the_face_and_nothing_else():
    box = crop_box(FACE, IMAGE)

    out = blend(IMAGE, FACE, box, Image.new("RGB", (REPAIR_SIZE, REPAIR_SIZE), "red"))

    assert out.size == IMAGE.size
    assert out.getpixel((450, 350)) == (255, 0, 0)
    assert out.getpixel((box[0] + 1, box[1] + 1)) == (255, 255, 255)
    assert out.getpixel((10, 10)) == (255, 255, 255)


class FakeGenerate:
    def __init__(self):
        self.calls = []

    def __call__(self, prompt, images, size, seed):
        self.calls.append((prompt, [image.size for image in images], size, seed))
        return Image.new("RGB", size, "red")


def test_repair_redraws_the_enlarged_crop_and_blends_it_back():
    generate = FakeGenerate()

    out = FaceRepairer(generate)(IMAGE, FACE, seed=1500)

    assert generate.calls == [(PROMPT, [(1024, 1024)], (1024, 1024), 1500)]
    assert out.getpixel((450, 350)) == (255, 0, 0)
    assert out.getpixel((10, 10)) == (255, 255, 255)


def test_a_face_reference_becomes_picture_2():
    generate = FakeGenerate()
    reference = Image.new("RGB", (400, 400))

    FaceRepairer(generate)(IMAGE, FACE, seed=1500, reference=reference)

    prompt, sizes, _, _ = generate.calls[0]
    assert prompt == PROMPT + REFERENCE_PROMPT
    assert sizes == [(1024, 1024), (400, 400)]


# Real repairs from the GPU experiment (2026-10-11), cropped around the face: a side profile
# that the repair turned into a front face, and a three-quarter view it sharpened.
FIXTURES = Path(__file__).parent / "fixtures" / "repair"
FACES = json.loads((FIXTURES / "faces.json").read_text("utf-8"))
LIMITS = load_settings().face_repair


def real_repair(case):
    data = FACES[case]
    images, checks = [], []
    for label, scores in (("before", "before"), ("after", "B")):
        images.append(Image.open(FIXTURES / f"{case}_{label}.png").convert("RGB"))
        face = Face(tuple(data[label]["face"]), data[label]["confidence"])
        score = data["scores"][scores]
        checks.append(Check(score["ccip_image"], face, score["sharpness"]))
    return (*images, *checks)


def test_layout_ignores_fine_detail_but_not_the_layout():
    before, after, *_ = real_repair("three_quarter")
    noise = np.random.default_rng(0).integers(-20, 21, (*before.size[::-1], 3))
    noisy = Image.fromarray(np.clip(np.asarray(before, int) + noise, 0, 255).astype(np.uint8))

    assert layout_similarity(before, before) == pytest.approx(1.0)
    assert layout_similarity(before, noisy) > 0.99
    assert layout_similarity(before, before.transpose(Image.FLIP_TOP_BOTTOM)) < 0.5


def test_a_front_face_drawn_over_a_side_profile_is_rejected():
    # Its identity score even improved: only the structure check catches it.
    keep, reason = keep_repair(*real_repair("side"), LIMITS)

    assert not keep
    assert reason.startswith("layout changed: 0.7")


def test_a_sharper_face_in_the_same_pose_is_kept():
    keep, reason = keep_repair(*real_repair("three_quarter"), LIMITS)

    assert keep
    assert reason == "kept: layout 1.00, face moved 0.00x, sharpness 0.24 -> 0.42"


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"face": None}, "no face found after the repair"),
        ({"face": Face((118, 47, 226, 167), 0.86)}, "face moved 0.37x its size"),
        ({"sharpness": 0.2}, "not sharper: 0.20 <= 0.24"),
        ({"identity": 0.03}, "less like the character: CCIP 0.030 vs 0.008"),
        ({"failures": ("face 40 px, smaller than 64 px",)}, "more checks failed"),
    ],
    ids=["no face", "face moved", "not sharper", "identity worse", "more failures"],
)
def test_a_repair_that_is_not_better_is_rejected(change, reason):
    before, after, before_check, after_check = real_repair("three_quarter")

    keep, why = keep_repair(before, after, before_check, replace(after_check, **change), LIMITS)

    assert not keep
    assert why.startswith(reason)


FRONT = Job("turnaround/front", "A.", (1280, 720), 42, ())
BACK = Job("turnaround/back", "B.", (1280, 720), 43, (), face=False)
GOOD = Check(0.01, FACE, 0.3)  # FACE is 100 px


class FakeGate:
    def __init__(self, faces=(FACE,)):
        self.faces, self.checked = list(faces), []

    def detect(self, image):
        return self.faces

    def __call__(self, job, image):
        self.checked.append(image)
        return Check(0.01, FACE, 0.5)


def stage(mode="failed", use_reference=True, gate=None):
    settings = LIMITS.model_copy(update={"mode": mode, "use_reference": use_reference})
    return FaceRepairStage(settings, FakeGenerate(), gate or FakeGate())


@pytest.mark.parametrize(
    ("mode", "job", "check", "repair"),
    [
        ("failed", FRONT, GOOD, False),
        ("failed", FRONT, replace(GOOD, failures=("too blurry",)), True),
        ("failed", FRONT, Check(0.5, None, None, ("no face found",)), False),
        ("failed", BACK, replace(GOOD, failures=("x",)), False),
        ("small", FRONT, GOOD, True),  # 100 px < 128
        ("small", FRONT, replace(GOOD, face=Face((0, 0, 200, 200), 0.9)), False),
        ("off", FRONT, replace(GOOD, failures=("too blurry",)), False),
    ],
    ids=["passed", "failed", "no face to repair", "shot from behind", "small face",
         "big face", "off"],
)  # fmt: skip
def test_which_faces_get_repaired(mode, job, check, repair):
    assert stage(mode).needs_repair(job, check) is repair


def test_the_face_reference_is_cut_from_the_reference():
    with_reference, without = stage(), stage(use_reference=False)

    with_reference.start(IMAGE)
    without.start(IMAGE)

    assert with_reference._reference.size == (220, 220)
    assert without._reference is None


@pytest.mark.parametrize("keep", [True, False])
def test_a_repair_is_checked_and_kept_only_if_the_guard_agrees(monkeypatch, keep):
    monkeypatch.setattr(repair_module, "keep_repair", lambda *args: (keep, "because"))
    gate = FakeGate()
    repairer = stage(gate=gate)
    repairer.start(IMAGE)

    repair, image = repairer(FRONT, IMAGE, replace(GOOD, failures=("too blurry",)), seed=42)

    assert (repair.seed, repair.kept, repair.reason) == (42 + REPAIR_SEED_STEP, keep, "because")
    assert repair.check.sharpness == 0.5
    assert len(gate.checked) == 1
    assert (image is IMAGE) is not keep
    assert repairer.repairer.generate.calls[0][1] == [(1024, 1024), (220, 220)]
