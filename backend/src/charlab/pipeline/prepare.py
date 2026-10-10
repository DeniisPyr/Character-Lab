"""Prepare stage: check the reference image, separate the subject and crop to it."""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from charlab.config import PrepareSettings
from charlab.pipeline.stages import StageUnavailable

log = logging.getLogger(__name__)

# Pillow format name -> accepted file extensions
FORMATS = {"PNG": (".png",), "JPEG": (".jpg", ".jpeg"), "WEBP": (".webp",)}

# Measured on an illustrated reference on white: border pixels lie within 2 of their median
# colour, and a colour distance over 25 outlines the subject as closely as the model does.
# At 15, faint marks in the background count as subject.
PLAIN_BORDER = 10
SUBJECT_DISTANCE = 25
# A cutout this small or this large means the model did not find the subject.
MIN_COVERAGE = 0.005
MAX_COVERAGE = 0.99

WHITE = (255, 255, 255)

# Takes an RGB image and returns an "L" mask of the same size: 255 subject, 0 background.
Remover = Callable[[Image.Image], Image.Image]


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


def rembg_mask(image: Image.Image, model: str) -> Image.Image:
    """Cut the subject out with a rembg model and return its mask.

    The model is downloaded on first use, to the folder in REMBG_HOME (default ~/.u2net).
    """
    try:
        # Without onnxruntime, importing rembg exits the process instead of raising.
        import rembg
    except (ImportError, SystemExit) as error:
        raise StageUnavailable(
            "background removal needs rembg; install the package with the [gpu] extra"
        ) from error
    return rembg.remove(image, session=rembg.new_session(model), only_mask=True)


def _transparency(image: Image.Image) -> Image.Image | None:
    """Return the alpha channel if any pixel is transparent."""
    if image.mode not in ("RGBA", "LA", "PA") and "transparency" not in image.info:
        return None
    alpha = image.convert("RGBA").getchannel("A")
    return alpha if alpha.getextrema()[0] < 255 else None


def _on_white(image: Image.Image, mask: Image.Image | None) -> Image.Image:
    white = Image.new("RGB", image.size, WHITE)
    white.paste(image.convert("RGB"), mask=mask)
    return white


def _plain_colour(rgb: np.ndarray) -> tuple[int, int, int] | None:
    """Return the background colour if the image border is one plain colour."""
    height, width, _ = rgb.shape
    side = max(4, min(height, width) // 50)
    parts = (rgb[:side], rgb[-side:], rgb[:, :side], rgb[:, -side:])
    border = np.concatenate([part.reshape(-1, 3) for part in parts]).astype(np.float32)
    colour = np.median(border, axis=0)
    # A high percentile, not the maximum: the subject may touch the edge, like feet or hair.
    if np.percentile(np.linalg.norm(border - colour, axis=1), 95) >= PLAIN_BORDER:
        return None
    return tuple(int(round(c)) for c in colour)


def _colour_mask(rgb: np.ndarray, colour: tuple[int, int, int]) -> Image.Image:
    """Mask the pixels that differ from the background colour."""
    distance = np.linalg.norm(rgb.astype(np.float32) - np.asarray(colour, np.float32), axis=2)
    return Image.fromarray(np.where(distance > SUBJECT_DISTANCE, 255, 0).astype(np.uint8))


def _crop(image: Image.Image, mask: Image.Image, margin: float) -> Image.Image:
    """Crop to the subject with a margin, staying inside the image.

    No padding past the edge: where the frame cuts the subject off (a bust, say), padding
    would leave a hard cut floating on the background.
    """
    left, top, right, bottom = mask.point(lambda v: 255 if v > 127 else 0).getbbox()
    pad = round(margin * max(right - left, bottom - top))
    return image.crop(
        (
            max(left - pad, 0),
            max(top - pad, 0),
            min(right + pad, image.width),
            min(bottom + pad, image.height),
        )
    )


def prepare_reference(
    path: Path, settings: PrepareSettings, remover: Remover | None = None
) -> Image.Image:
    """Load the reference and return it as RGB, cropped to the subject.

    Transparent images and images on a plain colour need no model. Otherwise the subject is
    cut out with `remover` (default: the rembg model from settings) and put on white. If the
    cutout is implausible, the image is used as it is and a warning is logged.
    """
    image = load_reference(path, settings)
    alpha = _transparency(image)
    flat = _on_white(image, alpha)
    if settings.background == "keep":
        return flat

    rgb = np.asarray(flat)
    if alpha is not None:
        log.info("%s: using the transparency as the background", path)
        mask = alpha
    elif settings.background == "auto" and (colour := _plain_colour(rgb)) is not None:
        log.info("%s: plain background %s, no model needed", path, colour)
        mask = _colour_mask(rgb, colour)
    else:
        log.info("%s: removing the background with %s", path, settings.background_model)
        mask = remover(flat) if remover else rembg_mask(flat, settings.background_model)
        coverage = (np.asarray(mask) > 127).mean()
        if not MIN_COVERAGE <= coverage <= MAX_COVERAGE:
            log.warning(
                "%s: the subject cutout covers %.1f%% of the image, so it probably failed; "
                "using the image as it is",
                path,
                coverage * 100,
            )
            return flat
        flat = _on_white(flat, mask)

    if mask.getextrema()[1] <= 127:
        raise InvalidReference(f"{path}: no subject found; the image is empty or one colour")
    return _crop(flat, mask, settings.crop_margin)
