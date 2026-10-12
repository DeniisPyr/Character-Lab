"""Make one dataset: prepare the reference, generate and check every image, caption it, and
export the image/caption pairs with manifest.json and report.json."""

from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

from charlab import __version__
from charlab.config import Settings
from charlab.pipeline.export import Export
from charlab.pipeline.generate import QwenGenerator
from charlab.pipeline.jobs import Job, Result
from charlab.pipeline.prepare import prepare_reference
from charlab.pipeline.quality import QualityGate
from charlab.pipeline.repair import FaceRepairStage
from charlab.pipeline.runner import run
from charlab.pipeline.stages import Caption, Stages
from charlab.recipe import Recipe


def run_record(
    reference: Path, description: str | None, settings: Settings, recipe: Recipe
) -> dict[str, Any]:
    """What the manifest records about the run, enough to repeat it."""
    return {
        "charlab": __version__,
        "reference": reference.name,
        "recipe": recipe.name,
        "seed": settings.seed,
        "description": description,
        "prepare": settings.prepare.model_dump(),
        "style": settings.style,
        "generation": settings.generation.model_dump(),
        "quality": settings.quality.model_dump(),
        "face_repair": settings.face_repair.model_dump(),
        "captioning": settings.caption.model_dump(),
    }


def make_dataset(
    reference: Path,
    jobs: Sequence[Job],
    out: Path,
    trigger: str,
    settings: Settings,
    generator: QwenGenerator,
    captioner: Caption,
    record: dict[str, Any],
    free_gpu: bool = True,
) -> list[Result]:
    """Make the dataset in `out` and return the results.

    `record` describes the run for the manifest. With `free_gpu`, the generator unloads before
    captioning so the caption model fits; without it, the generator stays loaded for the next
    dataset, which needs a GPU with room for both (80 GB).
    """
    gate = QualityGate(settings.quality)
    repair = None
    if settings.face_repair.mode != "off":
        repair = FaceRepairStage(settings.face_repair, generator, gate)
    stages = Stages(
        prepare=partial(prepare_reference, settings=settings.prepare),
        generate=generator,
        caption=captioner,
        check=gate,
        repair=repair,
    )
    export = Export(out, jobs)
    results = run(
        reference,
        jobs,
        stages,
        on_image=export.save_image,
        after_generate=generator.unload if free_gpu else None,
        max_attempts=settings.quality.max_attempts,
    )
    export.write_captions(results, trigger, record)
    export.write_report(results, settings.quality)
    return results
