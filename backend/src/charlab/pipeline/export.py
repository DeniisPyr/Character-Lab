"""Export stage: write the dataset as numbered image/caption pairs with a manifest.

    001_turnaround_front.png   the image
    001_turnaround_front.txt   its training caption
    manifest.json              the run's settings, then every image with its seed and prompt
    report.json                the quality gate's scores for every attempt at every image, and
                               every face repair

The number keeps files in generation order. LoRA trainers pair each image with the .txt of
the same name.
"""

import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from charlab.config import QualitySettings
from charlab.pipeline.caption import build_caption
from charlab.pipeline.jobs import Attempt, Check, Job, Repair, Result
from charlab.pipeline.quality import duplicates


class Export:
    """Writes one run's dataset into `out`."""

    def __init__(self, out: Path, jobs: Sequence[Job]):
        self.out = out
        self._stems = {
            job.key: f"{number:03d}_{job.view}_{job.variant}"
            for number, job in enumerate(jobs, start=1)
        }

    def image_name(self, job: Job) -> str:
        """The image's file name, e.g. 001_turnaround_front.png."""
        return f"{self._stems[job.key]}.png"

    def save_image(self, job: Job, image: Image.Image) -> None:
        """Save one image as soon as it is generated."""
        self.out.mkdir(parents=True, exist_ok=True)
        image.save(self.out / self.image_name(job))

    def write_captions(self, results: Sequence[Result], trigger: str, run: dict[str, Any]) -> Path:
        """Write a caption file per image and the manifest; return the manifest's path.

        `run` describes the run (settings, models, version) and goes into the manifest as is.
        """
        self.out.mkdir(parents=True, exist_ok=True)
        images = []
        for result in results:
            job, stem = result.job, self._stems[result.job.key]
            caption = build_caption(trigger, job.caption, result.description)
            (self.out / f"{stem}.txt").write_text(caption + "\n", encoding="utf-8")
            images.append(
                {
                    "image": f"{stem}.png",
                    "caption_file": f"{stem}.txt",
                    "view": job.view,
                    "variant": job.variant,
                    "size": list(job.size),
                    "seed": result.seed,
                    "prompt": job.prompt,
                    "caption": caption,
                }
            )
        manifest = self.out / "manifest.json"
        content = {**run, "trigger": trigger, "images": images}
        manifest.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
        return manifest

    def write_report(self, results: Sequence[Result], limits: QualitySettings) -> Path:
        """Write the quality gate's scores for every attempt, each face repair, and the
        near-duplicate pairs."""
        self.out.mkdir(parents=True, exist_ok=True)
        images = [
            {
                "image": self.image_name(result.job),
                "passed": None if result.check is None else result.check.passed,
                "kept_attempt": result.kept + 1,
                "attempts": [_attempt(attempt) for attempt in result.attempts],
                "repair": None if result.repair is None else _repair(result.repair),
            }
            for result in results
        ]
        pairs = duplicates(
            {self.image_name(r.job): r.image for r in results}, limits.duplicate_max_bits
        )
        report = {
            "summary": {
                "images": len(images),
                "passed": sum(image["passed"] is not False for image in images),
                "retried": [i["image"] for i in images if len(i["attempts"]) > 1],
                "failed": [i["image"] for i in images if i["passed"] is False],
                "repaired": [i["image"] for i in images if i["repair"] and i["repair"]["kept"]],
                "repairs_rejected": [
                    i["image"] for i in images if i["repair"] and not i["repair"]["kept"]
                ],
                "near_duplicates": len(pairs),
            },
            "limits": limits.model_dump(include=_LIMITS),
            "images": images,
            "near_duplicates": [{"a": a, "b": b, "different_bits": bits} for a, b, bits in pairs],
        }
        path = self.out / "report.json"
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return path

    def zip(self) -> Path:
        """Pack the dataset into <out>.zip, next to the folder."""
        shutil.make_archive(str(self.out), "zip", root_dir=self.out)
        return self.out.with_name(f"{self.out.name}.zip")


_LIMITS = {
    "max_attempts",
    "same_character_below",
    "min_face_px",
    "min_sharpness",
    "duplicate_max_bits",
}


def _attempt(attempt: Attempt) -> dict[str, Any]:
    if attempt.check is None:
        return {"seed": attempt.seed, "passed": None}
    return {"seed": attempt.seed, **_check(attempt.check)}


def _repair(repair: Repair) -> dict[str, Any]:
    return {
        "seed": repair.seed,
        "kept": repair.kept,
        "reason": repair.reason,
        **_check(repair.check),
    }


def _check(check: Check) -> dict[str, Any]:
    face = check.face
    return {
        "passed": check.passed,
        "identity": round(check.identity, 4),
        "face": None
        if face is None
        else {"box": list(face.box), "size": face.size, "confidence": round(face.confidence, 3)},
        "sharpness": None if check.sharpness is None else round(check.sharpness, 3),
        "failures": list(check.failures),
    }
