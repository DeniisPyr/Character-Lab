from pathlib import Path

import pytest
from PIL import Image

from charlab.config import load_settings
from charlab.pipeline.jobs import Check, Job, plan
from charlab.pipeline.runner import RETRY_SEED_STEP, run
from charlab.pipeline.stages import Stages
from charlab.recipe import load_recipe

REFERENCE = Path("reference.png")


class FakeStages:
    """Stand-ins for the model stages that record every call."""

    def __init__(self, wrong_size=False):
        self.calls = []
        self.picture_1 = Image.new("RGB", (64, 64), "white")
        self.wrong_size = wrong_size

    def prepare(self, reference):
        self.calls.append(("prepare", reference))
        return self.picture_1

    def generate(self, prompt, images, size, seed):
        width, height = size
        image = Image.new("RGB", (width + self.wrong_size, height), (seed % 256, 0, 0))
        self.calls.append(("generate", prompt, list(images), image))
        return image

    def caption(self, image):
        self.calls.append(("caption", image))
        return f"a {image.width}x{image.height} image"

    def stages(self):
        return Stages(self.prepare, self.generate, self.caption)


def shipped_jobs():
    settings = load_settings()
    return plan(load_recipe(settings.recipe), settings.seed)


def test_full_pipeline_with_fake_stages():
    jobs = shipped_jobs()
    fake = FakeStages()

    results = run(REFERENCE, jobs, fake.stages())

    assert [result.job for result in results] == jobs
    assert fake.calls[0] == ("prepare", REFERENCE)
    generated = {result.job.key: result.image for result in results}
    generate_calls = [call for call in fake.calls if call[0] == "generate"]
    for job, (_, prompt, images, image) in zip(jobs, generate_calls, strict=True):
        assert prompt == job.prompt
        assert images[0] is fake.picture_1
        expected = [generated[key] for key in job.references]
        assert all(a is b for a, b in zip(images[1:], expected, strict=True))
        assert image is generated[job.key]
        assert image.size == job.size
    assert [result.description for result in results] == [
        f"a {width}x{height} image" for width, height in (job.size for job in jobs)
    ]


def test_captioning_starts_after_all_images_are_generated():
    fake = FakeStages()

    run(REFERENCE, shipped_jobs(), fake.stages())

    kinds = [call[0] for call in fake.calls]
    assert kinds == ["prepare"] + ["generate"] * 16 + ["caption"] * 16


def test_each_image_is_handed_over_at_once_and_the_gpu_freed_before_captions():
    fake = FakeStages()

    run(
        REFERENCE,
        shipped_jobs(),
        fake.stages(),
        on_image=lambda job, image: fake.calls.append(("saved", job.key, image)),
        after_generate=lambda: fake.calls.append(("after_generate",)),
    )

    kinds = [call[0] for call in fake.calls]
    assert kinds == (
        ["prepare"] + ["generate", "saved"] * 16 + ["after_generate"] + ["caption"] * 16
    )
    generated = [call[3] for call in fake.calls if call[0] == "generate"]
    saved = [call[2] for call in fake.calls if call[0] == "saved"]
    assert all(a is b for a, b in zip(generated, saved, strict=True))


def test_without_a_caption_stage_results_have_no_descriptions():
    fake = FakeStages()

    results = run(REFERENCE, shipped_jobs(), Stages(fake.prepare, fake.generate))

    assert [result.description for result in results] == [None] * 16
    assert "caption" not in [call[0] for call in fake.calls]


def test_reference_must_be_generated_first():
    jobs = [Job("view/a", "A.", (64, 64), 1, ("view/b",)), Job("view/b", "B.", (64, 64), 2, ())]

    with pytest.raises(ValueError, match=r"view/a needs \['view/b'\]"):
        run(REFERENCE, jobs, FakeStages().stages())


def test_generator_must_return_the_requested_size():
    jobs = [Job("view/a", "A.", (64, 64), 1, ())]

    with pytest.raises(ValueError, match=r"returned \(65, 64\), expected \(64, 64\)"):
        run(REFERENCE, jobs, FakeStages(wrong_size=True).stages())


class FakeGate:
    """Fails the attempts whose seed is listed, with the identity given per seed."""

    def __init__(self, failing=(), identity=None):
        self.failing, self.identity = set(failing), identity or {}
        self.reference, self.checked = None, []

    def start(self, reference):
        self.reference = reference

    def __call__(self, job, image):
        seed = image.getpixel((0, 0))[0]  # the fake generator paints seed % 256 into red
        self.checked.append((job.key, seed))
        failures = ("too different",) if seed in self.failing else ()
        return Check(self.identity.get(seed, 0.01), failures=failures)


TWO_JOBS = [
    Job("view/a", "A.", (64, 64), 10, ()),
    Job("view/b", "B.", (64, 64), 11, ("view/a",)),
]
RETRY = 10 + RETRY_SEED_STEP  # job a's first retry seed; 1010 % 256 == 242


def gated(fake, gate):
    return Stages(fake.prepare, fake.generate, fake.caption, check=gate)


def test_the_gate_scores_the_prepared_reference_first():
    fake, gate = FakeStages(), FakeGate()

    run(REFERENCE, TWO_JOBS, gated(fake, gate), max_attempts=3)

    assert gate.reference is fake.picture_1
    assert gate.checked == [("view/a", 10), ("view/b", 11)]


def test_a_failed_image_is_made_again_and_later_jobs_get_the_kept_one():
    fake, gate, saved = FakeStages(), FakeGate(failing={10}), []

    results = run(
        REFERENCE,
        TWO_JOBS,
        gated(fake, gate),
        on_image=lambda job, image: saved.append((job.key, image.getpixel((0, 0))[0])),
        max_attempts=3,
    )

    first = results[0]
    assert [attempt.seed for attempt in first.attempts] == [10, RETRY]
    assert (first.kept, first.seed) == (1, RETRY)
    assert saved == [("view/a", RETRY % 256), ("view/b", 11)]
    last_inputs = [call[2] for call in fake.calls if call[0] == "generate"][-1]
    assert last_inputs[1] is first.image


def test_when_every_attempt_fails_the_best_one_is_kept():
    seeds = [10, 10 + RETRY_SEED_STEP, 10 + 2 * RETRY_SEED_STEP]
    identity = {10: 0.5, 1010 % 256: 0.2, 2010 % 256: 0.3}
    gate = FakeGate(failing={s % 256 for s in seeds}, identity=identity)

    results = run(REFERENCE, TWO_JOBS[:1], gated(FakeStages(), gate), max_attempts=3)

    assert [attempt.seed for attempt in results[0].attempts] == seeds
    assert results[0].seed == seeds[1]


def test_without_a_gate_each_image_is_made_once():
    fake = FakeStages()

    results = run(REFERENCE, TWO_JOBS, fake.stages(), max_attempts=3)

    assert [len(result.attempts) for result in results] == [1, 1]
    assert results[0].attempts[0].check is None
