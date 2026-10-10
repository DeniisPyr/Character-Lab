import sys

import pytest
from PIL import Image

from charlab.config import load_settings
from charlab.pipeline.caption import Florence, FlorenceCaptioner, load_florence
from charlab.pipeline.stages import StageUnavailable

SETTINGS = load_settings().caption
TASK = "<MORE_DETAILED_CAPTION>"


class FakeInputs(dict):
    def to(self, device, dtype):
        self["moved_to"] = (device, dtype)
        return self


class FakeProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, text, images, return_tensors):
        self.calls.append(("encode", text, images.mode, images.size))
        return FakeInputs(input_ids="ids", pixel_values="pixels")

    def batch_decode(self, ids, skip_special_tokens):
        return [f"<s>{ids}</s>"]

    def post_process_generation(self, text, task, image_size):
        self.calls.append(("parse", text, task, image_size))
        return {task: "  A woman stands in a park.  "}


class FakeModel:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return "generated"


@pytest.fixture
def florence():
    return Florence(FakeModel(), FakeProcessor(), device="cpu", dtype="float32")


def make(florence, settings=SETTINGS):
    loads = []

    def load(settings):
        loads.append(settings)
        return florence

    return FlorenceCaptioner(settings, load), loads


def test_model_loads_once_on_the_first_call(florence):
    captioner, loads = make(florence)
    assert loads == []

    captioner(Image.new("RGB", (64, 32)))
    captioner(Image.new("RGB", (64, 32)))

    assert loads == [SETTINGS]


def test_caption_runs_the_configured_task_and_strips_the_text(florence):
    caption = make(florence)[0](Image.new("RGBA", (64, 32)))

    assert caption == "A woman stands in a park."
    encode, parse = florence.processor.calls
    assert encode == ("encode", TASK, "RGB", (64, 32))
    assert parse == ("parse", "<s>generated</s>", TASK, (64, 32))


def test_generation_uses_the_settings_and_no_sampling(florence):
    settings = SETTINGS.model_copy(update={"num_beams": 5, "max_new_tokens": 100})

    make(florence, settings)[0](Image.new("RGB", (64, 32)))

    call = florence.model.calls[0]
    assert (call["num_beams"], call["max_new_tokens"], call["do_sample"]) == (5, 100, False)
    assert call["moved_to"] == ("cpu", "float32")
    assert call["pixel_values"] == "pixels"


def test_missing_packages_are_reported(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)

    with pytest.raises(StageUnavailable, match=r"needs torch and transformers; .* \[gpu\] extra"):
        load_florence(SETTINGS)
