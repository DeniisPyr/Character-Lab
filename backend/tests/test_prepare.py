import sys

import numpy as np
import pytest
from PIL import Image

from charlab.config import PrepareSettings
from charlab.pipeline import prepare
from charlab.pipeline.prepare import InvalidReference, load_reference, prepare_reference, rembg_mask
from charlab.pipeline.stages import StageUnavailable

SETTINGS = PrepareSettings(
    max_file_mb=20,
    max_pixels=25_000_000,
    min_side=512,
    background="auto",
    background_model="birefnet-general",
    crop_margin=0.05,
)
WHITE, GREY, RED = (255, 255, 255), (128, 128, 128), (200, 0, 0)
# A 100x200 subject: with a 5% margin of its longer side, the crop is 120x220.
SUBJECT = (200, 300, 300, 500)
CROPPED = (120, 220)


def save(tmp_path, name, size=(512, 768), fmt=None, **params):
    path = tmp_path / name
    Image.new("RGB", size, "white").save(path, format=fmt, **params)
    return path


def figure(tmp_path, background=WHITE, box=SUBJECT, mode="RGB", name="ref.png", **params):
    if background == "noise":
        pixels = np.random.default_rng(0).integers(0, 256, (768, 512, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
    else:
        image = Image.new(mode, (512, 768), background)
    image.paste(RED, box)
    path = tmp_path / name
    image.save(path, **params)
    return path


def mask_of(box):
    def remove(image):
        calls.append(image)
        mask = Image.new("L", image.size, 0)
        mask.paste(255, box)
        return mask

    calls = []
    remove.calls = calls
    return remove


def no_model(image):
    raise AssertionError("the model should not run")


@pytest.mark.parametrize("name", ["ref.png", "ref.jpg", "ref.JPEG", "ref.webp"])
def test_supported_formats_load(tmp_path, name):
    image = load_reference(save(tmp_path, name), SETTINGS)

    assert image.size == (512, 768)


def test_exif_rotation_is_applied(tmp_path):
    exif = Image.Exif()
    exif[0x0112] = 6  # orientation: rotate 90 degrees clockwise
    path = save(tmp_path, "photo.jpg", size=(512, 768), exif=exif)

    assert load_reference(path, SETTINGS).size == (768, 512)


def test_missing_file(tmp_path):
    with pytest.raises(InvalidReference, match="not found"):
        load_reference(tmp_path / "nope.png", SETTINGS)


def test_empty_file(tmp_path):
    path = tmp_path / "empty.png"
    path.touch()

    with pytest.raises(InvalidReference, match="the file is empty"):
        load_reference(path, SETTINGS)


def test_file_over_the_size_limit(tmp_path):
    settings = SETTINGS.model_copy(update={"max_file_mb": 0.0001})

    with pytest.raises(InvalidReference, match="over the limit of 0.0001 MB"):
        load_reference(save(tmp_path, "ref.png"), settings)


def test_not_an_image(tmp_path):
    path = tmp_path / "ref.png"
    path.write_text("hello", encoding="utf-8")

    with pytest.raises(InvalidReference, match="cannot be read"):
        load_reference(path, SETTINGS)


def test_truncated_image(tmp_path):
    path = save(tmp_path, "ref.png")
    path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])

    with pytest.raises(InvalidReference, match="cannot be read"):
        load_reference(path, SETTINGS)


def test_extension_must_match_content(tmp_path):
    path = save(tmp_path, "ref.jpg", fmt="PNG")

    with pytest.raises(InvalidReference, match="the file is PNG but named .jpg"):
        load_reference(path, SETTINGS)


def test_unsupported_format(tmp_path):
    with pytest.raises(InvalidReference, match="GIF images are not supported"):
        load_reference(save(tmp_path, "ref.gif"), SETTINGS)


def test_too_many_pixels(tmp_path):
    settings = SETTINGS.model_copy(update={"max_pixels": 100_000})

    with pytest.raises(InvalidReference, match="512x768 is over the limit of 100,000 pixels"):
        load_reference(save(tmp_path, "ref.png"), settings)


def test_decompression_bomb_is_rejected(tmp_path, monkeypatch):
    path = save(tmp_path, "ref.png")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)

    with pytest.raises(InvalidReference, match="too many pixels to load safely"):
        load_reference(path, SETTINGS)


def test_too_small(tmp_path):
    with pytest.raises(InvalidReference, match="the shorter side needs at least 512 px"):
        load_reference(save(tmp_path, "ref.png", size=(511, 2000)), SETTINGS)


def test_plain_background_is_cropped_without_the_model(tmp_path):
    image = prepare_reference(figure(tmp_path), SETTINGS, no_model)

    assert (image.mode, image.size) == ("RGB", CROPPED)
    assert image.getpixel((0, 0)) == WHITE
    assert image.getpixel((60, 110)) == RED


def test_crop_stays_inside_the_image(tmp_path):
    path = figure(tmp_path, background=GREY, box=(220, 568, 280, 768))

    image = prepare_reference(path, SETTINGS, no_model)

    assert image.size == (80, 210)
    assert image.getpixel((0, 0)) == GREY
    assert image.getpixel((40, 209)) == RED


def test_jpeg_noise_still_counts_as_plain(tmp_path):
    path = figure(tmp_path, name="ref.jpg", quality=85)

    image = prepare_reference(path, SETTINGS, no_model)

    assert image.width == pytest.approx(CROPPED[0], abs=4)
    assert image.height == pytest.approx(CROPPED[1], abs=4)


def test_transparency_is_used_as_the_background(tmp_path):
    path = figure(tmp_path, background=(0, 0, 0, 0), mode="RGBA")

    image = prepare_reference(path, SETTINGS, no_model)

    assert (image.mode, image.size) == ("RGB", CROPPED)
    assert image.getpixel((0, 0)) == WHITE


def test_busy_background_is_removed_with_the_model(tmp_path):
    remover = mask_of(SUBJECT)

    image = prepare_reference(figure(tmp_path, background="noise"), SETTINGS, remover)

    assert [call.size for call in remover.calls] == [(512, 768)]
    assert image.size == CROPPED
    assert image.getpixel((0, 0)) == WHITE
    assert image.getpixel((60, 110)) == RED


def test_remove_mode_uses_the_model_on_a_plain_background(tmp_path):
    remover = mask_of(SUBJECT)
    settings = SETTINGS.model_copy(update={"background": "remove"})

    assert prepare_reference(figure(tmp_path), settings, remover).size == CROPPED
    assert len(remover.calls) == 1


def test_keep_mode_only_flattens_transparency(tmp_path):
    path = figure(tmp_path, background=(0, 0, 0, 0), mode="RGBA")
    settings = SETTINGS.model_copy(update={"background": "keep"})

    image = prepare_reference(path, settings, no_model)

    assert (image.mode, image.size) == ("RGB", (512, 768))
    assert image.getpixel((0, 0)) == WHITE


@pytest.mark.parametrize("box", [(0, 0, 0, 0), (0, 0, 512, 768)], ids=["empty", "everything"])
def test_implausible_cutout_falls_back_to_the_image(tmp_path, caplog, box):
    path = figure(tmp_path, background="noise")

    image = prepare_reference(path, SETTINGS, mask_of(box))

    assert image.size == (512, 768)
    assert image.getpixel((0, 0)) == Image.open(path).getpixel((0, 0))
    assert "probably failed" in caplog.text


@pytest.mark.parametrize(("background", "mode"), [(WHITE, "RGB"), ((0, 0, 0, 0), "RGBA")])
def test_image_without_a_subject_is_rejected(tmp_path, background, mode):
    path = tmp_path / "blank.png"
    Image.new(mode, (512, 768), background).save(path)

    with pytest.raises(InvalidReference, match="no subject found"):
        prepare_reference(path, SETTINGS, no_model)


def test_default_remover_uses_the_configured_model(tmp_path, monkeypatch):
    models = []

    def fake_rembg(image, model):
        models.append(model)
        return mask_of(SUBJECT)(image)

    monkeypatch.setattr(prepare, "rembg_mask", fake_rembg)

    assert prepare_reference(figure(tmp_path, background="noise"), SETTINGS).size == CROPPED
    assert models == ["birefnet-general"]


def test_missing_rembg_is_reported(monkeypatch):
    monkeypatch.setitem(sys.modules, "rembg", None)

    with pytest.raises(StageUnavailable, match=r"needs rembg; .* \[gpu\] extra"):
        rembg_mask(Image.new("RGB", (512, 512)), "birefnet-general")
