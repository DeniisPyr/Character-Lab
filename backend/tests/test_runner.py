from pathlib import Path

import pytest
from PIL import Image

from charlab.config import load_settings
from charlab.pipeline.jobs import Job, plan
from charlab.pipeline.runner import run
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
    assert [result.caption for result in results] == [
        f"a {width}x{height} image" for width, height in (job.size for job in jobs)
    ]


def test_captioning_starts_after_all_images_are_generated():
    fake = FakeStages()

    run(REFERENCE, shipped_jobs(), fake.stages())

    kinds = [call[0] for call in fake.calls]
    assert kinds == ["prepare"] + ["generate"] * 16 + ["caption"] * 16


def test_reference_must_be_generated_first():
    jobs = [Job("view/a", "A.", (64, 64), 1, ("view/b",)), Job("view/b", "B.", (64, 64), 2, ())]

    with pytest.raises(ValueError, match=r"view/a needs \['view/b'\]"):
        run(REFERENCE, jobs, FakeStages().stages())


def test_generator_must_return_the_requested_size():
    jobs = [Job("view/a", "A.", (64, 64), 1, ())]

    with pytest.raises(ValueError, match=r"returned \(65, 64\), expected \(64, 64\)"):
        run(REFERENCE, jobs, FakeStages(wrong_size=True).stages())
