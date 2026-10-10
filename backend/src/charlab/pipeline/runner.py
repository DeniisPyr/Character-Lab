"""Runs the stages over a plan of jobs."""

import logging
from collections.abc import Sequence
from pathlib import Path

from charlab.pipeline.jobs import Job, Result
from charlab.pipeline.stages import Stages

log = logging.getLogger(__name__)


def run(reference: Path, jobs: Sequence[Job], stages: Stages) -> list[Result]:
    """Prepare the reference once, generate every job in order, then caption all images.

    Picture 1 is always the prepared reference; a job's references follow as Picture 2, 3.
    Captioning starts only after the last image is generated, so only one model needs to be
    in GPU memory at a time.
    """
    picture_1 = stages.prepare(reference)

    images = {}
    for number, job in enumerate(jobs, start=1):
        missing = [key for key in job.references if key not in images]
        if missing:
            raise ValueError(f"{job.key} needs {missing}, which are not generated yet")
        inputs = [picture_1, *(images[key] for key in job.references)]
        image = stages.generate(job.prompt, inputs, job.size, job.seed)
        if image.size != job.size:
            raise ValueError(f"{job.key}: generator returned {image.size}, expected {job.size}")
        images[job.key] = image
        log.info("generated %s (%d/%d)", job.key, number, len(jobs))

    results = []
    for number, job in enumerate(jobs, start=1):
        results.append(Result(job, images[job.key], stages.caption(images[job.key])))
        log.info("captioned %s (%d/%d)", job.key, number, len(jobs))
    return results
