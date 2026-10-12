"""Command-line interface for the ``charlab`` command."""

import argparse
import json
import logging
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic import ValidationError

from charlab import __version__
from charlab.ablation import InvalidPlan, load_plan, run_plan
from charlab.config import load_settings
from charlab.eval.dataset import InvalidDataset, load_dataset
from charlab.eval.metrics import Scorer, summarise
from charlab.eval.report import write_eval
from charlab.make import make_dataset, run_record
from charlab.pipeline.caption import TRIGGER, FlorenceCaptioner, build_caption
from charlab.pipeline.export import Export
from charlab.pipeline.generate import QwenGenerator
from charlab.pipeline.jobs import Job, plan
from charlab.pipeline.prepare import InvalidReference, load_reference, prepare_reference
from charlab.pipeline.stages import StageUnavailable
from charlab.recipe import load_recipe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="charlab",
        description="Turn one character image into a training-ready LoRA dataset.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="command")

    generate = commands.add_parser("generate", help="generate a dataset from a reference image")
    generate.add_argument("reference", type=Path, help="character reference image")
    generate.add_argument(
        "--trigger", required=True, help="trigger word that starts every caption, e.g. mychar"
    )
    generate.add_argument("--out", type=Path, help="output folder (default: results/<trigger>)")
    generate.add_argument("--config", type=Path, help="YAML file with settings to override")
    generate.add_argument(
        "--description",
        help="optional sentence about the character's style and outfit, added to every prompt",
    )
    generate.add_argument(
        "--dry-run", action="store_true", help="print the generation plan without loading models"
    )
    generate.add_argument("--zip", action="store_true", help="also pack the dataset into <out>.zip")

    evaluate = commands.add_parser(
        "eval", help="score a finished dataset against its reference; writes eval.json and eval.md"
    )
    evaluate.add_argument("dataset", type=Path, help="folder of image/caption pairs")
    evaluate.add_argument(
        "--reference", type=Path, required=True, help="the character reference image"
    )
    evaluate.add_argument("--config", type=Path, help="YAML file with settings to override")

    ablate = commands.add_parser(
        "ablate", help="compare settings: make and score a dataset per character and variant"
    )
    ablate.add_argument("--plan", type=Path, required=True, help="YAML file: variants and runs")
    ablate.add_argument(
        "--references", type=Path, required=True, help="folder with one reference per character"
    )
    ablate.add_argument(
        "--out", type=Path, default=Path("results/ablation"), help="default: results/ablation"
    )
    ablate.add_argument(
        "--config", type=Path, help="YAML file with settings to override in every run"
    )
    ablate.add_argument(
        "--dry-run", action="store_true", help="list the runs without loading models"
    )
    return parser


def format_plan(jobs: Sequence[Job], header: Sequence[str], trigger: str) -> str:
    lines = [*header, ""]
    width = max(len(job.key) for job in jobs)
    for number, job in enumerate(jobs, start=1):
        width_px, height_px = job.size
        references = f"  + {', '.join(job.references)}" if job.references else ""
        lines.append(
            f"{number:>2}. {job.key:<{width}}  {width_px}x{height_px}  seed {job.seed}{references}"
        )
        lines += textwrap.wrap(
            job.prompt, width=96, initial_indent=" " * 4, subsequent_indent=" " * 4
        )
        lines += textwrap.wrap(
            f"{build_caption(trigger, job.caption, None)} + the image's description",
            width=96,
            initial_indent="    caption: ",
            subsequent_indent=" " * 13,
        )
    return "\n".join(lines)


def _error(message: str, command: str = "generate") -> int:
    print(f"charlab {command}: error: {message}", file=sys.stderr)
    return 2


def _generate(args: argparse.Namespace) -> int:
    if not TRIGGER.match(args.trigger):
        return _error(f"trigger must be one word of letters, digits, _ or -, got {args.trigger!r}")
    try:
        settings = load_settings(args.config)
        recipe = load_recipe(settings.recipe)
        reference = load_reference(args.reference, settings.prepare)
    except (OSError, ValidationError, InvalidReference) as error:
        return _error(str(error))

    jobs = plan(recipe, settings.seed, args.description)
    out = args.out or Path("results") / args.trigger
    if args.dry_run:
        header = [
            f"{len(jobs)} images from recipe {recipe.name!r}, base seed {settings.seed}",
            f"reference: {args.reference} ({reference.width}x{reference.height}, checked)",
            f"trigger:   {args.trigger}",
            f"output:    {out}",
            f"quality:   {settings.style} checks, up to {settings.quality.max_attempts} "
            "attempts per image",
            "repair:    "
            + (
                "off"
                if settings.face_repair.mode == "off"
                else f"{settings.face_repair.mode} faces"
            ),
        ]
        print(format_plan(jobs, header, args.trigger))
        return 0

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    try:
        results = make_dataset(
            args.reference,
            jobs,
            out,
            args.trigger,
            settings,
            QwenGenerator(settings.generation),
            FlorenceCaptioner(settings.caption),
            run_record(args.reference, args.description, settings, recipe),
        )
    except (InvalidReference, StageUnavailable) as error:
        return _error(str(error))

    print(f"saved {len(results)} images with captions to {out}")
    failed = [r.job.key for r in results if r.check and r.check.failures]
    print(f"quality: {len(results) - len(failed)}/{len(results)} images passed every check")
    if failed:
        print(f"failed all attempts (kept the best): {', '.join(failed)}")
    if args.zip:
        print(f"packed the dataset into {Export(out, jobs).zip()}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    try:
        settings = load_settings(args.config)
        dataset = load_dataset(args.dataset)
        # The prepared reference, as the quality gate saw it during generation.
        reference = prepare_reference(args.reference, settings.prepare)
        scores = Scorer(settings.quality)(reference, dataset)
    except (OSError, ValidationError, InvalidDataset, InvalidReference, StageUnavailable) as error:
        return _error(str(error), "eval")

    limit = settings.quality.same_character_below
    summary = summarise(scores, limit)
    run = {
        "charlab": __version__,
        "dataset": args.dataset.resolve().name,
        "reference": args.reference.name,
        "same_character_below": limit,
    }
    paths = write_eval(args.dataset, scores.images, summary, run)
    images, captions = summary["images"], summary["captions"]
    print(f"scored {images} images against {args.reference.name}")
    print(
        f"same character: {summary['identity']['same_character']}/{images}, "
        f"faces found: {summary['faces']['found']}/{images}, "
        f"captioned: {captions['captioned']}/{images}"
    )
    print(f"wrote {' and '.join(str(path) for path in paths)}")
    return 0


def _ablate(args: argparse.Namespace) -> int:
    try:
        ablation = load_plan(args.plan, args.references, args.config)
    except (OSError, yaml.YAMLError, ValidationError, InvalidPlan) as error:
        return _error(str(error), "ablate")

    print(f"{len(ablation.runs)} datasets, compared with {ablation.baseline!r}:")
    for run in ablation.runs:
        changes = json.dumps(run.variant.settings) if run.variant.settings else "defaults"
        chain = "" if run.variant.chain else ", no earlier images as Picture 2"
        done = (args.out / run.character / run.variant.name / "report.json").is_file()
        print(f"  {run.character}/{run.variant.name}: {changes}{chain}{' (made)' if done else ''}")
    if args.dry_run:
        return 0

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    try:
        paths = run_plan(ablation, args.out, args.config)
    except (InvalidReference, InvalidDataset, StageUnavailable) as error:
        return _error(str(error), "ablate")
    print(f"wrote {' and '.join(str(path) for path in paths)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate":
        return _generate(args)
    if args.command == "eval":
        return _evaluate(args)
    if args.command == "ablate":
        return _ablate(args)
    parser.print_help()
    return 0
