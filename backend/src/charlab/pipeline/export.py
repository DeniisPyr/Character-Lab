"""Export stage: write the dataset as numbered image/caption pairs with a manifest.

    001_turnaround_front.png   the image
    001_turnaround_front.txt   its training caption
    manifest.json              the run's settings, then every image with its seed and prompt

The number keeps files in generation order. LoRA trainers pair each image with the .txt of
the same name.
"""

import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from charlab.pipeline.caption import build_caption
from charlab.pipeline.jobs import Job, Result


class Export:
    """Writes one run's dataset into `out`."""

    def __init__(self, out: Path, jobs: Sequence[Job]):
        self.out = out
        self._stems = {
            job.key: f"{number:03d}_{job.view}_{job.variant}"
            for number, job in enumerate(jobs, start=1)
        }

    def save_image(self, job: Job, image: Image.Image) -> None:
        """Save one image as soon as it is generated."""
        self.out.mkdir(parents=True, exist_ok=True)
        image.save(self.out / f"{self._stems[job.key]}.png")

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
                    "seed": job.seed,
                    "prompt": job.prompt,
                    "caption": caption,
                }
            )
        manifest = self.out / "manifest.json"
        content = {**run, "trigger": trigger, "images": images}
        manifest.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
        return manifest

    def zip(self) -> Path:
        """Pack the dataset into <out>.zip, next to the folder."""
        return Path(shutil.make_archive(str(self.out), "zip", root_dir=self.out))
