"""Read a finished dataset the way LoRA trainers do: each image with the .txt caption of the
same name."""

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

# Image types that LoRA trainers such as ai-toolkit and kohya pick up.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


class InvalidDataset(Exception):
    """The folder is missing, has no images, or has an image that cannot be read."""


@dataclass(frozen=True)
class Entry:
    name: str  # the image's file name
    image: Image.Image
    caption: str | None  # None when the image has no caption file, or an empty one


@dataclass(frozen=True)
class Dataset:
    entries: tuple[Entry, ...]
    orphans: tuple[str, ...]  # caption files without an image


def load_dataset(folder: Path) -> Dataset:
    """Every image in `folder`, in file name order, with its caption."""
    if not folder.is_dir():
        raise InvalidDataset(f"{folder}: not a folder")
    files = sorted(path for path in folder.iterdir() if not path.name.startswith("."))
    images = [path for path in files if path.suffix.lower() in IMAGE_SUFFIXES]
    if not images:
        raise InvalidDataset(f"{folder}: no images found")

    entries = []
    for path in images:
        try:
            with Image.open(path) as image:
                pixels = image.convert("RGB")
        except (UnidentifiedImageError, OSError) as error:
            message = f"{path}: cannot be read; it is damaged or not an image"
            raise InvalidDataset(message) from error
        text = path.with_suffix(".txt")
        caption = text.read_text(encoding="utf-8").strip() if text.is_file() else ""
        entries.append(Entry(path.name, pixels, caption or None))
    stems = {path.stem for path in images}
    orphans = tuple(path.name for path in files if path.suffix == ".txt" and path.stem not in stems)
    return Dataset(tuple(entries), orphans)
