import json

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from charlab import ablation, make
from charlab.ablation import InvalidPlan, load_plan, render, run_plan
from charlab.eval.metrics import ImageScore, Scores
from charlab.pipeline.jobs import Check

PLAN = """\
variants:
  baseline: {}
  seed_43:
    settings: {seed: 43}
  gguf:
    settings: {generation: {transformer: gguf}}
  no_chain:
    chain: false
runs:
  - characters: all
    variants: [baseline]
  - characters: [anna]
    variants: [gguf, seed_43, no_chain, baseline]
"""


@pytest.fixture
def references(tmp_path):
    folder = tmp_path / "refs"
    folder.mkdir()
    for name, colour in (("anna", (200, 0, 0)), ("ben", (0, 0, 200))):
        image = Image.new("RGB", (512, 768), "white")
        image.paste(colour, (200, 300, 300, 500))
        image.save(folder / f"{name}.png")
    (folder / "notes.txt").write_text("not a reference", encoding="utf-8")
    return folder


def write_plan(tmp_path, text=PLAN):
    path = tmp_path / "plan.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_plan_runs_are_grouped_so_each_model_loads_once(tmp_path, references):
    plan = load_plan(write_plan(tmp_path), references)

    assert plan.baseline == "baseline"
    assert [(run.character, run.variant.name) for run in plan.runs] == [
        ("anna", "baseline"),
        ("ben", "baseline"),
        ("anna", "seed_43"),
        ("anna", "no_chain"),
        ("anna", "gguf"),
    ]
    no_chain = plan.runs[3].variant
    assert (no_chain.chain, no_chain.settings) == (False, {})
    assert plan.runs[0].reference == references / "anna.png"


@pytest.mark.parametrize(
    ("text", "error", "message"),
    [
        ("variants: {a: {}}\nruns: [{characters: all, variants: [b]}]\n", InvalidPlan, "'b'"),
        ("variants: {a: {}}\nruns: [{characters: [cleo], variants: [a]}]\n", InvalidPlan, "cleo"),
        ("variants: {}\nruns: []\n", InvalidPlan, "no variants"),
        (
            "variants: {a: {settings: {generation: {steps: 0}}}}\nruns: []\n",
            ValidationError,
            "greater than or equal to 1",
        ),
        ("variants: {a: {sed: 1}}\nruns: []\n", ValidationError, "Extra inputs"),
    ],
    ids=["unknown variant", "unknown character", "no variants", "bad setting", "typo"],
)
def test_bad_plans_are_rejected(tmp_path, references, text, error, message):
    with pytest.raises(error, match=message):
        load_plan(write_plan(tmp_path, text), references)


def test_character_names_must_work_as_trigger_words(tmp_path, references):
    (references / "anna.png").rename(references / "anna b.png")

    with pytest.raises(InvalidPlan, match="anna b: a character name must be one word"):
        load_plan(write_plan(tmp_path), references)


class FakeGenerator:
    """Paints each image in a grey set by its seed, so the fake scores follow the seed."""

    created = []

    def __init__(self, settings):
        self.settings = settings
        self.pictures = []
        self.unloaded = False
        FakeGenerator.created.append(self)

    def __call__(self, prompt, images, size, seed):
        self.pictures.append(len(images))
        return Image.new("RGB", size, (seed,) * 3)

    def unload(self):
        self.unloaded = True


class FakeCaptioner:
    def __init__(self, settings):
        pass

    def __call__(self, image):
        return "A person stands in a studio."


class FakeGate:
    def __init__(self, settings):
        pass

    def start(self, reference):
        pass

    def __call__(self, job, image):
        return Check(0.01)


class FakeScorer:
    """Identity is the first image's grey / 1000: its seed / 1000."""

    def __init__(self, settings):
        pass

    def __call__(self, reference, dataset):
        identity = float(np.asarray(dataset.entries[0].image).mean()) / 1000
        images = tuple(
            ImageScore(e.name, e.image.size, identity, None, None, None, e.caption)
            for e in dataset.entries
        )
        count = len(images)
        return Scores(images, np.zeros((count, count)), np.full((count, count), 20), ())


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setattr(ablation, "QwenGenerator", FakeGenerator)
    monkeypatch.setattr(ablation, "FlorenceCaptioner", FakeCaptioner)
    monkeypatch.setattr(ablation, "Scorer", FakeScorer)
    monkeypatch.setattr(make, "QualityGate", FakeGate)
    monkeypatch.setattr(FakeGenerator, "created", [])


def test_every_run_is_made_scored_and_compared(fakes, tmp_path, references):
    out = tmp_path / "out"

    data, text = run_plan(load_plan(write_plan(tmp_path), references), out)

    for run in ("anna/baseline", "ben/baseline", "anna/seed_43", "anna/no_chain", "anna/gguf"):
        names = {path.name for path in (out / run).iterdir()}
        assert len([name for name in names if name.endswith(".png")]) == 16
        assert {"manifest.json", "report.json", "eval.json", "eval.md", "run.json"} <= names
    bf16, gguf = FakeGenerator.created
    assert (bf16.settings.transformer, gguf.settings.transformer) == ("bf16", "gguf")
    assert bf16.unloaded and gguf.unloaded
    # bf16 made anna/baseline, ben/baseline, anna/seed_43, then anna/no_chain.
    assert max(bf16.pictures[:16]) == 2
    assert max(bf16.pictures[48:64]) == 1
    manifest = json.loads((out / "anna/no_chain/manifest.json").read_text("utf-8"))
    assert manifest["ablation"] == {"variant": "no_chain", "chain": False}
    assert manifest["trigger"] == "anna"

    report = json.loads(data.read_text("utf-8"))
    assert report["baseline"] == "baseline"
    assert [(r["character"], r["variant"]) for r in report["runs"]][:2] == [
        ("anna", "baseline"),
        ("ben", "baseline"),
    ]
    assert isinstance(report["runs"][0]["seconds"], int)
    markdown = text.read_text("utf-8")
    assert "| seed_43 | 1 | +0.001 | +0.001 | – | – | +0.000 | +0.0 |" in markdown
    assert "| gguf | 1 | +0.000 |" in markdown
    assert "| ben | baseline | 16/16 | 0.042 |" in markdown


def test_a_second_call_scores_again_without_making_the_datasets(fakes, tmp_path, references):
    plan = load_plan(write_plan(tmp_path), references)
    out = tmp_path / "out"
    run_plan(plan, out)
    FakeGenerator.created.clear()
    (out / "anna/gguf/eval.json").unlink()

    run_plan(plan, out)

    assert FakeGenerator.created == []
    assert (out / "anna/gguf/eval.json").is_file()


def summary(identity, face=None):
    return {
        "images": 16,
        "identity": {"mean": identity, "worst": identity, "same_character": 16},
        "face_identity": face and {"mean": face, "worst": face},
        "consistency": None,
        "diversity": None,
        "faces": {"found": 0},
    }


def test_variants_show_their_mean_change_from_the_baseline():
    rows = [
        {"character": "a", "variant": "base", "seconds": 90, "summary": summary(0.05, 0.02)},
        {"character": "b", "variant": "base", "seconds": None, "summary": summary(0.07)},
        {"character": "a", "variant": "x", "seconds": 60, "summary": summary(0.03, 0.03)},
        {"character": "b", "variant": "x", "seconds": 60, "summary": summary(0.05)},
        {"character": "c", "variant": "x", "seconds": 60, "summary": summary(0.5)},
    ]

    markdown = render({"charlab": "1.0", "baseline": "base", "runs": rows})

    # c has no baseline, so only a and b count; only a has a face score.
    assert "| x | 2 | -0.020 | -0.020 | +0.010 | +0.010 | – | – |" in markdown
    assert "| a | base | 16/16 | 0.050 | 0.050 | 0.020 | 0.020 | – | – | 0/16 | 1.5 |" in markdown
    assert "| b | base | 16/16 | 0.070 | 0.070 | – | – | – | – | 0/16 | – |" in markdown
