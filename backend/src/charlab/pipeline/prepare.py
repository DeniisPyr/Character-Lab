"""Prepare stage: load and check the reference image."""

from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from charlab.config import PrepareSettings

# Pillow format name -> accepted file extensions
FORMATS = {"PNG": (".png",), "JPEG": (".jpg", ".jpeg"), "WEBP": (".webp",)}


class InvalidReference(Exception):
    """The reference image cannot be used; the message says why."""


def _check_header(path: Path, image: Image.Image, limits: PrepareSettings) -> None:
    if image.format not in FORMATS:
        raise InvalidReference(
            f"{path}: {image.format or 'unknown'} images are not supported; use PNG, JPEG or WebP"
        )
    if path.suffix.lower() not in FORMATS[image.format]:
        raise InvalidReference(
            f"{path}: the file is {image.format} but named {path.suffix or 'without an extension'}"
        )
    width, height = image.size
    if width * height > limits.max_pixels:
        raise InvalidReference(
            f"{path}: {width}x{height} is over the limit of {limits.max_pixels:,} pixels"
        )
    if min(width, height) < limits.min_side:
        raise InvalidReference(
            f"{path}: {width}x{height} is too small; "
            f"the shorter side needs at least {limits.min_side} px"
        )


def load_reference(path: Path, limits: PrepareSettings) -> Image.Image:
    """Check the reference file and return it decoded and upright (EXIF rotation applied).

    Size, format and dimensions are checked from the file and its header before any pixels
    are decoded, so oversized or mislabelled files are rejected without loading them.
    """
    if not path.is_file():
        raise InvalidReference(f"{path}: not found")
    size = path.stat().st_size
    if size == 0:
        raise InvalidReference(f"{path}: the file is empty")
    if size > limits.max_file_mb * 1024**2:
        raise InvalidReference(
            f"{path}: {size / 1024**2:.1f} MB is over the limit of {limits.max_file_mb} MB"
        )
    try:
        with Image.open(path) as header:
            _check_header(path, header, limits)
            header.verify()
        image = Image.open(path)
        image.load()
    except Image.DecompressionBombError as error:
        raise InvalidReference(f"{path}: too many pixels to load safely") from error
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
        raise InvalidReference(f"{path}: cannot be read; it is damaged or not an image") from error
    return ImageOps.exif_transpose(image)
