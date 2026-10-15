"""Command-line interface for the ``charlab`` command."""

import argparse
import logging
import os
import shlex
import subprocess
import sys
import textwrap
import time
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from charlab import __version__
from charlab.config import load_settings
from charlab.eval.dataset import InvalidDataset, load_dataset
from charlab.eval.metrics import Scorer, summarise
from charlab.eval.report import write_eval
from charlab.make import make_dataset, run_record
from charlab.pipeline.caption import TRIGGER, FlorenceCaptioner, build_caption
from charlab.pipeline.export import Export
from charlab.pipeline.generate import QwenGenerator
from charlab.pipeline.jobs import Job, Result, plan
from charlab.pipeline.prepare import InvalidReference, load_reference, prepare_reference
from charlab.pipeline.stages import StageUnavailable
from charlab.recipe import load_recipe

EXAMPLES = """\
examples:
  charlab generate reference.png --trigger diana --dry-run
      show the 16 shots, prompts and captions; needs no GPU
  charlab generate reference.png --trigger diana
      make the dataset in results/diana (needs an NVIDIA GPU)
  charlab generate reference.png --trigger diana --config 24gb.yaml
      the same on a GPU with 24 GB of memory
  charlab eval results/diana --reference reference.png
      score a finished dataset; needs no GPU
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="charlab",
        description="Turn one character image into a training-ready LoRA dataset.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="command")

    generate = commands.add_parser("generate", help="generate a dataset from a reference image")
    generate.add_argument("reference", type=Path, help="character reference image")
    generate.add_argument(
        "--trigger", required=True, help="trigger word that starts every caption, e.g. mychar"
    )
    generate.add_argument("--out", type=Path, help="output folder (default: results/<trigger>)")
    generate.add_argument(
        "--config", type=Path, help="YAML file with settings to override; 24gb.yaml for a 24 GB GPU"
    )
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


def _log_progress() -> None:
    """Show charlab's progress lines, without the model downloads' HTTP requests."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)


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
            f"mode:      {'24' if settings.generation.transformer == 'gguf' else '80'} GB GPU "
            f"({settings.generation.transformer})",
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

    _log_progress()
    start = time.perf_counter()
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

    export = Export(out, jobs)
    packed = export.zip() if args.zip else None
    print(finish(results, export, args, time.perf_counter() - start, packed))
    return 0


def duration(seconds: float) -> str:
    """6 min 12 s, 48 s or 1 h 3 min."""
    minutes, rest = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} h {minutes} min"
    return f"{minutes} min {rest} s" if minutes else f"{rest} s"


def shell_arg(text: str) -> str:
    """`text` quoted, if needed, so a printed command can be pasted into this system's shell."""
    return subprocess.list2cmdline([text]) if os.name == "nt" else shlex.quote(text)


def finish(
    results: Sequence[Result],
    export: Export,
    args: argparse.Namespace,
    seconds: float,
    packed: Path | None,
) -> str:
    """What the run made, how the checks went, and what to do with the dataset next."""
    out, count = export.out, len(results)
    lines = [
        "",
        f"Done in {duration(seconds)}. The dataset is in {out}: "
        f"{count} images, each with a caption.",
    ]
    checked = [r for r in results if r.check]
    if checked:
        failed = [export.image_name(r.job) for r in checked if r.check.failures]
        retried = [export.image_name(r.job) for r in checked if len(r.attempts) > 1]
        quality = f"Quality: {count - len(failed)} of {count} images passed every check"
        if retried:
            quality += f"; {len(retried)} needed another try ({', '.join(retried)})"
        lines.append(f"{quality}. Details: {out / 'report.json'}")
        if failed:
            lines.append(f"Failed every try, so the best one was kept: {', '.join(failed)}")
    if packed:
        lines.append(f"Packed into {packed}")
    reference = shell_arg(str(args.reference))
    lines += [
        "",
        "Next:",
        f"  check the dataset  charlab eval {shell_arg(str(out))} --reference {reference}",
        f"  train a LoRA       point ai-toolkit or kohya at {out}; every caption starts with "
        f'"{args.trigger}"',
    ]
    return "\n".join(lines)


def _evaluate(args: argparse.Namespace) -> int:
    _log_progress()
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
    data, text = paths
    print(f"\nScored {images} images against {args.reference.name}.")
    print(
        f"Same character: {summary['identity']['same_character']} of {images}. "
        f"Faces found: {summary['faces']['found']} of {images}. "
        f"Captioned: {captions['captioned']} of {images}."
    )
    print(f"Details: {text} (every number is in {data.name})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate":
        return _generate(args)
    if args.command == "eval":
        return _evaluate(args)
    parser.print_help()
    return 0
