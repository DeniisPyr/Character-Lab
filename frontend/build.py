"""Build the showcase page: copy the page files, and turn the example dataset into web images and
one data file, so the page shows a real run without running anything.

    python frontend/build.py           # writes site/
    python -m http.server -d site      # then open http://localhost:8000

Needs only Pillow.
"""

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "diana"
REFERENCE = ROOT / "examples" / "reference.png"
PAGE = ("index.html", "styles.css", "app.js")
# The datasets' PNGs are about 1 MB each; WebP at this size keeps the page under 1 MB.
LONG_SIDE = 1280
QUALITY = 82
# The scores table: key in eval.json's summary, name, what it measures, and the formats of the
# mean and the worst value (as in eval.md, so the page shows the same numbers).
SCORES = (
    ("identity", "Identity", "CCIP difference from the reference; lower is better", ".3f", ".3f"),
    ("face_identity", "Face identity", "CCIP difference from the reference face; lower is better",
     ".3f", ".3f"),
    ("consistency", "Consistency", "CCIP difference between two images; lower is better",
     ".3f", ".3f"),
    ("face_size", "Face size", "shorter side in px; higher is better", ".0f", ".0f"),
    ("sharpness", "Face sharpness", "× the reference face's; higher is better", ".2f", ".2f"),
    ("diversity", "Diversity", "fingerprint bits that differ, of 64; higher is better",
     ".1f", ".0f"),
)  # fmt: skip


def web_image(source: Path, target: Path) -> None:
    with Image.open(source) as image:
        image = image.convert("RGB")
        image.thumbnail((LONG_SIDE, LONG_SIDE), Image.Resampling.LANCZOS)
        image.save(target, "WEBP", quality=QUALITY, method=6)


def build(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    for name in PAGE:
        shutil.copy2(ROOT / "frontend" / name, out / name)

    manifest = json.loads((EXAMPLE / "manifest.json").read_text("utf-8"))
    report = json.loads((EXAMPLE / "report.json").read_text("utf-8"))
    evaluation = json.loads((EXAMPLE / "eval.json").read_text("utf-8"))
    checks = {entry["image"]: entry for entry in report["images"]}
    scores = {entry["image"]: entry for entry in evaluation["images"]}

    web_image(REFERENCE, out / "images" / "reference.webp")
    images = []
    for entry in manifest["images"]:
        name = entry["image"]
        web_image(EXAMPLE / name, out / "images" / f"{Path(name).stem}.webp")
        check, score = checks[name], scores[name]
        images.append(
            {
                "src": f"images/{Path(name).stem}.webp",
                "file": name,
                "view": entry["view"],
                "variant": entry["variant"],
                "size": entry["size"],
                "seed": entry["seed"],
                "prompt": entry["prompt"],
                "caption": entry["caption"],
                "passed": check["passed"],
                "tries": len(check["attempts"]),
                "identity": f"{score['identity']:.3f}",
                "face": score["face"] and score["face"]["size"],
                "sharpness": None if score["sharpness"] is None else f"{score['sharpness']:.2f}",
            }
        )

    summary = evaluation["summary"]
    scores = []
    for key, name, description, mean, worst in SCORES:
        spread = summary[key]
        if spread is None:
            continue
        where = spread.get("worst_image") or " and ".join(spread["worst_pair"])
        scores.append(
            {
                "name": name,
                "description": description,
                "mean": format(spread["mean"], mean),
                "worst": format(spread["worst"], worst),
                "where": where,
            }
        )
    generation = manifest["generation"]
    data = {
        "trigger": manifest["trigger"],
        "made_by": manifest["charlab"],
        "scored_by": evaluation["charlab"],
        "model": generation["model"],
        "steps": generation["steps"],
        "same_character_below": evaluation["same_character_below"],
        "summary": summary,
        "scores": scores,
        "images": images,
    }
    (out / "data.json").write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "site", help="default: site/")
    args = parser.parse_args()
    build(args.out)
    size = sum(path.stat().st_size for path in args.out.rglob("*") if path.is_file())
    print(f"built {args.out} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
