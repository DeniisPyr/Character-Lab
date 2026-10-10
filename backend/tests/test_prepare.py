import pytest
from PIL import Image

from charlab.config import PrepareSettings
from charlab.pipeline.prepare import InvalidReference, load_reference

LIMITS = PrepareSettings(max_file_mb=20, max_pixels=25_000_000, min_side=512)


def save(tmp_path, name, size=(512, 768), fmt=None, **params):
    path = tmp_path / name
    Image.new("RGB", size, "white").save(path, format=fmt, **params)
    return path


@pytest.mark.parametrize("name", ["ref.png", "ref.jpg", "ref.JPEG", "ref.webp"])
def test_supported_formats_load(tmp_path, name):
    image = load_reference(save(tmp_path, name), LIMITS)

    assert image.size == (512, 768)


def test_exif_rotation_is_applied(tmp_path):
    exif = Image.Exif()
    exif[0x0112] = 6  # orientation: rotate 90 degrees clockwise
    path = save(tmp_path, "photo.jpg", size=(512, 768), exif=exif)

    assert load_reference(path, LIMITS).size == (768, 512)


def test_missing_file(tmp_path):
    with pytest.raises(InvalidReference, match="not found"):
        load_reference(tmp_path / "nope.png", LIMITS)


def test_empty_file(tmp_path):
    path = tmp_path / "empty.png"
    path.touch()

    with pytest.raises(InvalidReference, match="the file is empty"):
        load_reference(path, LIMITS)


def test_file_over_the_size_limit(tmp_path):
    limits = LIMITS.model_copy(update={"max_file_mb": 0.0001})

    with pytest.raises(InvalidReference, match="over the limit of 0.0001 MB"):
        load_reference(save(tmp_path, "ref.png"), limits)


def test_not_an_image(tmp_path):
    path = tmp_path / "ref.png"
    path.write_text("hello", encoding="utf-8")

    with pytest.raises(InvalidReference, match="cannot be read"):
        load_reference(path, LIMITS)


def test_truncated_image(tmp_path):
    path = save(tmp_path, "ref.png")
    path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])

    with pytest.raises(InvalidReference, match="cannot be read"):
        load_reference(path, LIMITS)


def test_extension_must_match_content(tmp_path):
    path = save(tmp_path, "ref.jpg", fmt="PNG")

    with pytest.raises(InvalidReference, match="the file is PNG but named .jpg"):
        load_reference(path, LIMITS)


def test_unsupported_format(tmp_path):
    with pytest.raises(InvalidReference, match="GIF images are not supported"):
        load_reference(save(tmp_path, "ref.gif"), LIMITS)


def test_too_many_pixels(tmp_path):
    limits = LIMITS.model_copy(update={"max_pixels": 100_000})

    with pytest.raises(InvalidReference, match="512x768 is over the limit of 100,000 pixels"):
        load_reference(save(tmp_path, "ref.png"), limits)


def test_decompression_bomb_is_rejected(tmp_path, monkeypatch):
    path = save(tmp_path, "ref.png")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)

    with pytest.raises(InvalidReference, match="too many pixels to load safely"):
        load_reference(path, LIMITS)


def test_too_small(tmp_path):
    with pytest.raises(InvalidReference, match="the shorter side needs at least 512 px"):
        load_reference(save(tmp_path, "ref.png", size=(511, 2000)), LIMITS)
