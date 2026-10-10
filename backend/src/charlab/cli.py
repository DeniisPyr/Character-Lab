"""Command-line interface for the ``charlab`` command."""

import argparse
import re
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from charlab import __version__
from charlab.config import load_settings
from charlab.pipeline.jobs import Job, plan
from charlab.pipeline.prepare import InvalidReference, load_reference
from charlab.recipe import load_recipe

TRIGGER = re.compile(r"^[A-Za-z0-9_-]+$")


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
    return parser


def format_plan(jobs: Sequence[Job], header: Sequence[str]) -> str:
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
    return "\n".join(lines)


def _error(message: str) -> int:
    print(f"charlab generate: error: {message}", file=sys.stderr)
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
        ]
        print(format_plan(jobs, header))
        return 0

    print(
        "charlab generate: the model stages are not built yet (roadmap step 2); "
        "use --dry-run to see the plan",
        file=sys.stderr,
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate":
        return _generate(args)
    parser.print_help()
    return 0
