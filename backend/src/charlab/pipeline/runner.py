"""Runs the stages over a plan of jobs."""

import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from PIL import Image

from charlab.pipeline.jobs import Attempt, Job, Result
from charlab.pipeline.stages import Stages

log = logging.getLogger(__name__)

# A retry's seed is the job's seed plus this times the retry number. Recipe positions stay far
# below it, so a retry never reuses another job's seed.
RETRY_SEED_STEP = 1000


def _kept(attempts: Sequence[Attempt]) -> int:
    """The first attempt that passed; if none did, the one with the fewest failures and then
    the lowest CCIP difference."""
    for index, attempt in enumerate(attempts):
        if attempt.check is None or attempt.check.passed:
            return index
    return min(
        range(len(attempts)),
        key=lambda i: (len(attempts[i].check.failures), attempts[i].check.identity),
    )


def run(
    reference: Path,
    jobs: Sequence[Job],
    stages: Stages,
    on_image: Callable[[Job, Image.Image], None] | None = None,
    after_generate: Callable[[], None] | None = None,
    max_attempts: int = 1,
) -> list[Result]:
    """Prepare the reference once, generate every job in order, then caption all images
    (if there is a caption stage).

    Picture 1 is always the prepared reference; a job's references follow as Picture 2, 3.
    Captioning starts only after the last image is generated, so only one model needs to be
    in GPU memory at a time.

    With a quality gate, each image is checked as soon as it is made. A failed image is made
    again with a new seed, up to `max_attempts` in all, before the next job starts, so later
    jobs get the kept image as Picture 2.

    `on_image` gets each kept image right away, so the images already made survive a later
    crash. `after_generate` runs between generating and captioning, to free the GPU for the
    caption model.
    """
    picture_1 = stages.prepare(reference)
    if stages.check:
        stages.check.start(picture_1)

    images, attempts_by_key, kept_by_key = {}, {}, {}
    for number, job in enumerate(jobs, start=1):
        missing = [key for key in job.references if key not in images]
        if missing:
            raise ValueError(f"{job.key} needs {missing}, which are not generated yet")
        inputs = [picture_1, *(images[key] for key in job.references)]

        attempts, made = [], []
        for retry in range(max_attempts if stages.check else 1):
            seed = job.seed + RETRY_SEED_STEP * retry
            image = stages.generate(job.prompt, inputs, job.size, seed)
            if image.size != job.size:
                raise ValueError(f"{job.key}: generator returned {image.size}, expected {job.size}")
            check = stages.check(job, image) if stages.check else None
            attempts.append(Attempt(seed, check))
            made.append(image)
            if check is None or check.passed:
                break
            log.info("%s, seed %d: %s", job.key, seed, "; ".join(check.failures))

        kept = _kept(attempts)
        if attempts[kept].check and not attempts[kept].check.passed:
            log.warning(
                "%s failed all %d attempts; keeping seed %d",
                job.key,
                len(attempts),
                attempts[kept].seed,
            )
        images[job.key] = made[kept]
        attempts_by_key[job.key] = attempts
        kept_by_key[job.key] = kept
        log.info("generated %s (%d/%d)", job.key, number, len(jobs))
        if on_image:
            on_image(job, made[kept])
    if after_generate:
        after_generate()

    def result(job: Job, description: str | None) -> Result:
        attempts = tuple(attempts_by_key[job.key])
        return Result(job, images[job.key], description, attempts, kept_by_key[job.key])

    if stages.caption is None:
        return [result(job, None) for job in jobs]
    results = []
    for number, job in enumerate(jobs, start=1):
        results.append(result(job, stages.caption(images[job.key])))
        log.info("captioned %s (%d/%d)", job.key, number, len(jobs))
    return results
