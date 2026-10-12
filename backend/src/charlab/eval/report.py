"""Write a dataset's evaluation next to it.

eval.json   every number: the summary, then each image's scores
eval.md     the same as tables, to read or paste into a README
"""

import json
from pathlib import Path
from typing import Any

from charlab.eval.metrics import ImageScore

# Summary rows: key in the summary, label, which way is better, number format.
ROWS = (
    ("identity", "Identity: CCIP difference from the reference", "lower", "{:.3f}"),
    ("face_identity", "Face identity: CCIP difference from the reference face", "lower", "{:.3f}"),
    ("consistency", "Consistency: CCIP difference between two images", "lower", "{:.3f}"),
    ("face_size", "Face size: shorter side in px", "higher", "{:.0f}"),
    ("sharpness", "Face sharpness: × the reference face's", "higher", "{:.2f}"),
    ("diversity", "Diversity: fingerprint bits that differ, of 64", "higher", "{:.3g}"),
)
NONE = "–"


def write_eval(
    out: Path, images: tuple[ImageScore, ...], summary: dict[str, Any], run: dict[str, Any]
) -> tuple[Path, Path]:
    """Write eval.json and eval.md into `out`; return their paths.

    `run` says what was evaluated: the charlab version, the dataset and reference names, and
    the CCIP limit for the same character.
    """
    report = {**run, "summary": summary, "images": [_image(image) for image in images]}
    data = out / "eval.json"
    data.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    text = out / "eval.md"
    text.write_text(render(report), encoding="utf-8")
    return data, text


def _image(image: ImageScore) -> dict[str, Any]:
    face = image.face
    return {
        "image": image.name,
        "size": list(image.size),
        "identity": round(image.identity, 4),
        "face": None
        if face is None
        else {"box": list(face.box), "size": face.size, "confidence": round(face.confidence, 3)},
        "face_identity": _round(image.face_identity, 4),
        "sharpness": _round(image.sharpness, 3),
        "caption_words": len(image.caption.split()) if image.caption else None,
    }


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def render(report: dict[str, Any]) -> str:
    """The report as markdown: a summary table, the counts, then a table of every image."""
    summary = report["summary"]
    lines = [
        f"# Evaluation of {report['dataset']}",
        "",
        f"{summary['images']} images against the reference `{report['reference']}`, "
        f"scored by charlab {report['charlab']}.",
        "",
        "| Score | Better | Mean | Worst | Worst image or pair |",
        "|---|---|---|---|---|",
    ]
    for key, label, better, number in ROWS:
        spread = summary[key]
        if spread is None:
            lines.append(f"| {label} | {better} | {NONE} | {NONE} | {NONE} |")
            continue
        where = spread.get("worst_image") or " and ".join(spread["worst_pair"])
        mean, worst = number.format(spread["mean"]), number.format(spread["worst"])
        lines.append(f"| {label} | {better} | {mean} | {worst} | {where} |")

    limit = report["same_character_below"]
    lines += ["", f"CCIP sees the same character below a difference of {limit}.", ""]
    lines += _counts(summary)
    lines += [
        "",
        "## Images",
        "",
        "| Image | Size | Identity | Face identity | Face px | Sharpness | Caption words |",
        "|---|---|---|---|---|---|---|",
    ]
    for image in report["images"]:
        width, height = image["size"]
        cells = [
            image["image"],
            f"{width}×{height}",
            f"{image['identity']:.3f}",
            _cell(image["face_identity"], "{:.3f}"),
            _cell(image["face"] and image["face"]["size"], "{}"),
            _cell(image["sharpness"], "{:.2f}"),
            _cell(image["caption_words"], "{}"),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _cell(value: Any, number: str) -> str:
    return NONE if value is None else number.format(value)


def _counts(summary: dict[str, Any]) -> list[str]:
    images, faces, captions = summary["images"], summary["faces"], summary["captions"]
    sizes = ", ".join(
        f"{count} × {size.replace('x', '×')}" for size, count in summary["sizes"].items()
    )
    lines = [
        f"- **Same character:** {summary['identity']['same_character']} of {images} images",
        f"- **Faces found:** {faces['found']} of {images}"
        + _names(" (none in {})", faces["missing"]),
        f"- **Sizes:** {sizes}",
        f"- **Captions:** {captions['captioned']} of {images} images",
    ]
    if captions["captioned"]:
        words = captions["words"]
        lines += [
            f"  - {captions['with_trigger']} start with `{captions['trigger']}`",
            f"  - {words['min']}–{words['max']} words, {words['mean']} on average",
            f"  - {captions['distinct']} distinct",
            f"  - {len(captions['need_cleaning'])} describe the character's fixed look or have "
            "filler" + _names(": {}", captions["need_cleaning"]),
        ]
    lines += [
        f"  - {label}: {', '.join(names)}"
        for label, names in (
            ("No caption", captions["missing"]),
            ("Captions without an image", captions["orphans"]),
        )
        if names
    ]
    return lines


def _names(template: str, names: list[str]) -> str:
    return template.format(", ".join(names)) if names else ""
