"""Compare settings: make a dataset for each character and variant in a plan, score each like
`charlab eval`, and put the scores side by side in ablation.md and ablation.json.

A plan (YAML) names the variants, each a change to the settings, and which characters get
which variants. The first variant is the baseline that the others are compared with:

    variants:
      baseline: {}
      gguf:
        settings: {generation: {transformer: gguf}}
      no_chain:
        chain: false   # no earlier images as Picture 2
    runs:
      - characters: all
        variants: [baseline]
      - characters: [mychar]
        variants: [gguf, no_chain]

Characters are the reference images in one folder, named by file name without the extension;
the name is also the trigger word.

Runs with the same generation settings share one loaded model, and the caption model stays
loaded next to it, which needs an 80 GB GPU for bf16. A dataset that is complete is not made
again, so a stopped session can continue. Every dataset is scored again on each call, so the
tables follow changes to the metrics.
"""

import json
import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import yaml
from PIL import Image
from pydantic import BaseModel, ConfigDict

from charlab import __version__
from charlab.config import Settings, load_settings
from charlab.eval.dataset import IMAGE_SUFFIXES, load_dataset
from charlab.eval.metrics import Scorer, summarise
from charlab.eval.report import write_eval
from charlab.make import make_dataset, run_record
from charlab.pipeline.caption import TRIGGER, FlorenceCaptioner
from charlab.pipeline.generate import QwenGenerator
from charlab.pipeline.jobs import plan
from charlab.pipeline.prepare import prepare_reference
from charlab.recipe import load_recipe

log = logging.getLogger(__name__)


class InvalidPlan(Exception):
    """The plan names an unknown variant or character, or a character name can't be a
    trigger word."""


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _VariantSpec(_Spec):
    settings: dict[str, Any] = {}
    chain: bool = True


class _RunSpec(_Spec):
    characters: Literal["all"] | list[str]
    variants: list[str]


class _PlanSpec(_Spec):
    variants: dict[str, _VariantSpec | None]
    runs: list[_RunSpec]


@dataclass(frozen=True)
class Variant:
    name: str
    settings: dict[str, Any]  # changes to the settings
    chain: bool = True  # earlier images as Picture 2, as the recipe says


@dataclass(frozen=True)
class Run:
    character: str
    reference: Path
    variant: Variant


@dataclass(frozen=True)
class Plan:
    baseline: str  # the variant the others are compared with
    runs: tuple[Run, ...]  # grouped by generation settings, so each model loads once


def load_plan(path: Path, references: Path, config: Path | None = None) -> Plan:
    """Read the plan and check it, including every variant's settings, before anything runs."""
    with open(path, encoding="utf-8") as file:
        spec = _PlanSpec.model_validate(yaml.safe_load(file) or {})
    if not spec.variants:
        raise InvalidPlan(f"{path}: no variants")
    variants = {
        name: Variant(name, (given or _VariantSpec()).settings, (given or _VariantSpec()).chain)
        for name, given in spec.variants.items()
    }
    # Raises on a bad setting; the generation settings decide which runs share a model.
    generation = {
        name: load_settings(config, changes=v.settings).generation.model_dump_json()
        for name, v in variants.items()
    }

    characters = {
        image.stem: image
        for image in sorted(references.iterdir())
        if image.suffix.lower() in IMAGE_SUFFIXES and not image.name.startswith(".")
    }
    if not characters:
        raise InvalidPlan(f"{references}: no reference images")
    for name in characters:
        if not TRIGGER.match(name):
            raise InvalidPlan(
                f"{name}: a character name must be one word of letters, digits, _ or -"
            )

    runs: dict[tuple[str, str], Run] = {}
    for entry in spec.runs:
        names = list(characters) if entry.characters == "all" else entry.characters
        for variant in entry.variants:
            if variant not in variants:
                raise InvalidPlan(f"{path}: unknown variant {variant!r}")
            for name in names:
                if name not in characters:
                    raise InvalidPlan(f"{path}: no reference image for {name!r} in {references}")
                runs.setdefault((name, variant), Run(name, characters[name], variants[variant]))

    modes = list(dict.fromkeys(generation[variant] for _, variant in runs))
    order = list(variants)
    grouped = sorted(
        runs.values(),
        key=lambda run: (modes.index(generation[run.variant.name]), order.index(run.variant.name)),
    )
    return Plan(baseline=order[0], runs=tuple(grouped))


def run_plan(plan_: Plan, out: Path, config: Path | None = None) -> tuple[Path, Path]:
    """Make and score every run's dataset in out/<character>/<variant>/, then write
    ablation.json and ablation.md into `out`; return their paths."""
    base = load_settings(config)
    scorer = Scorer(base.quality)
    generator: QwenGenerator | None = None
    captioners: dict[str, FlorenceCaptioner] = {}
    # Every variant is scored against the reference prepared with the base settings, so the
    # scores compare.
    references: dict[str, Image.Image] = {}
    rows = []
    for run in plan_.runs:
        folder = out / run.character / run.variant.name
        settings = load_settings(config, changes=run.variant.settings)
        if not (folder / "report.json").is_file():
            if generator is None or generator.settings != settings.generation:
                if generator is not None:
                    generator.unload()
                generator = QwenGenerator(settings.generation)
            key = settings.caption.model_dump_json()
            if key not in captioners:
                captioners[key] = FlorenceCaptioner(settings.caption)
            _make(run, folder, settings, generator, captioners[key])

        if run.character not in references:
            references[run.character] = prepare_reference(run.reference, base.prepare)
        scores = scorer(references[run.character], load_dataset(folder))
        summary = summarise(scores, base.quality.same_character_below)
        write_eval(folder, scores.images, summary, _eval_record(run, base))
        timing = folder / "run.json"
        seconds = json.loads(timing.read_text("utf-8"))["seconds"] if timing.is_file() else None
        rows.append(
            {
                "character": run.character,
                "variant": run.variant.name,
                "chain": run.variant.chain,
                "settings": run.variant.settings,
                "seconds": seconds,
                "summary": summary,
            }
        )
    if generator is not None:
        generator.unload()

    report = {"charlab": __version__, "baseline": plan_.baseline, "runs": rows}
    data = out / "ablation.json"
    data.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    text = out / "ablation.md"
    text.write_text(render(report), encoding="utf-8")
    return data, text


def _make(
    run: Run,
    folder: Path,
    settings: Settings,
    generator: QwenGenerator,
    captioner: FlorenceCaptioner,
) -> None:
    recipe = load_recipe(settings.recipe)
    jobs = plan(recipe, settings.seed)
    if not run.variant.chain:
        jobs = [replace(job, references=()) for job in jobs]
    record = {
        **run_record(run.reference, None, settings, recipe),
        "ablation": {"variant": run.variant.name, "chain": run.variant.chain},
    }
    log.info("making %s/%s", run.character, run.variant.name)
    start = time.perf_counter()
    make_dataset(
        run.reference, jobs, folder, run.character, settings, generator, captioner, record,
        free_gpu=False,
    )  # fmt: skip
    seconds = round(time.perf_counter() - start)
    (folder / "run.json").write_text(json.dumps({"seconds": seconds}) + "\n", encoding="utf-8")


def _eval_record(run: Run, base: Settings) -> dict[str, Any]:
    return {
        "charlab": __version__,
        "dataset": f"{run.character}/{run.variant.name}",
        "reference": run.reference.name,
        "same_character_below": base.quality.same_character_below,
    }


# Columns: label, how to read the number from a run's summary, and its format.
COLUMNS = (
    ("Identity", lambda s: s["identity"]["mean"], "{:.3f}"),
    ("Worst identity", lambda s: s["identity"]["worst"], "{:.3f}"),
    ("Face identity", lambda s: s["face_identity"] and s["face_identity"]["mean"], "{:.3f}"),
    ("Worst face", lambda s: s["face_identity"] and s["face_identity"]["worst"], "{:.3f}"),
    ("Consistency", lambda s: s["consistency"] and s["consistency"]["mean"], "{:.3f}"),
    ("Diversity, bits", lambda s: s["diversity"] and s["diversity"]["mean"], "{:.1f}"),
)
NONE = "–"


def render(report: dict[str, Any]) -> str:
    """The comparison as markdown: each variant against the baseline, then every dataset."""
    rows, baseline = report["runs"], report["baseline"]
    lines = [
        "# Ablation",
        "",
        f"{len(rows)} datasets, each scored like `charlab eval` against its character's "
        f"reference, by charlab {report['charlab']}. CCIP differences: lower is closer.",
        "",
        "## Variants",
        "",
        f"Each variant against `{baseline}` on the same characters: the mean change. A variant "
        "that changes only the seed shows how much the scores move by chance.",
        "",
        "| Variant | Characters | " + " | ".join(f"Δ {label}" for label, _, _ in COLUMNS) + " |",
        "|---|---|" + "---|" * len(COLUMNS),
    ]
    base = {row["character"]: row["summary"] for row in rows if row["variant"] == baseline}
    for variant in dict.fromkeys(row["variant"] for row in rows if row["variant"] != baseline):
        paired = [
            (base[row["character"]], row["summary"])
            for row in rows
            if row["variant"] == variant and row["character"] in base
        ]
        cells = []
        for _, read, number in COLUMNS:
            changes = [
                read(after) - read(before)
                for before, after in paired
                if read(before) is not None and read(after) is not None
            ]
            signed = number.replace("{:", "{:+")
            cells.append(signed.format(sum(changes) / len(changes)) if changes else NONE)
        lines.append(f"| {variant} | {len(paired)} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Datasets",
        "",
        "Minutes include loading the models for the first dataset of each generation mode.",
        "",
        "| Character | Variant | Same character | "
        + " | ".join(label for label, _, _ in COLUMNS)
        + " | Faces | Minutes |",
        "|---|---|---|" + "---|" * len(COLUMNS) + "---|---|",
    ]
    for row in rows:
        summary = row["summary"]
        images = summary["images"]
        cells = [
            row["character"],
            row["variant"],
            f"{summary['identity']['same_character']}/{images}",
            *(
                NONE if (value := read(summary)) is None else number.format(value)
                for _, read, number in COLUMNS
            ),
            f"{summary['faces']['found']}/{images}",
            NONE if row["seconds"] is None else f"{row['seconds'] / 60:.1f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
