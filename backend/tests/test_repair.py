import pytest
from PIL import Image

from charlab.pipeline.faces import Face
from charlab.pipeline.repair import (
    PROMPT,
    REFERENCE_PROMPT,
    REPAIR_SIZE,
    FaceRepairer,
    blend,
    crop_box,
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
