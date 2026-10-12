import json
import re
import sys
from pathlib import Path

import pytest
from PIL import Image

from charlab.config import load_settings
from charlab.pipeline.caption import (
    Florence,
    FlorenceCaptioner,
    build_caption,
    clean_description,
    load_florence,
    needs_cleaning,
)
from charlab.pipeline.stages import StageUnavailable

SETTINGS = load_settings().caption
TASK = "<MORE_DETAILED_CAPTION>"
# Florence-2-large's real descriptions of the 16 images from a GPU run (2026-10-10).
DESCRIPTIONS = json.loads(
    (Path(__file__).parent / "fixtures" / "florence_descriptions.json").read_text("utf-8")
)
FIXED_LOOK_OR_FILLER = re.compile(
    r"\bhair\b|\bappears to be in\b|\bher body\b|\bmakeup\b|\boverall mood\b|\bfocal point\b"
    r"|\brendering\b",
    re.IGNORECASE,
)


@pytest.mark.parametrize("name", sorted(DESCRIPTIONS))
def test_real_descriptions_lose_the_fixed_look_and_filler(name):
    cleaned = clean_description(DESCRIPTIONS[name])

    assert cleaned
    assert not FIXED_LOOK_OR_FILLER.search(cleaned)


def test_raw_descriptions_need_cleaning_and_finished_captions_do_not():
    for description in DESCRIPTIONS.values():
        assert needs_cleaning(description)
        assert not needs_cleaning(build_caption("mychar", "full body, front view", description))


def test_outfit_pose_and_background_are_kept():
    cleaned = clean_description(DESCRIPTIONS["turnaround_back"])

    assert cleaned == (
        "The image shows a young woman standing with her back towards the camera. She is "
        "wearing a beige off-the-shoulder dress with long sleeves and a round neckline. The "
        "dress is made of a soft, loose-fitting fabric and has a relaxed fit. She has bare "
        "feet and is standing on a white background."
    )


@pytest.mark.parametrize(
    ("sentence", "kept"),
    [
        (
            "The image is a close-up portrait of a young woman with long, wavy hair.",
            "The image is a close-up portrait of a young woman.",
        ),
        (
            "She is wearing a white dress and has long dark hair that is styled in loose waves.",
            "She is wearing a white dress.",
        ),
        ("Her hair is styled in loose waves.", ""),
        # A cut here once left "The woman and falls over her shoulders."
        ("The woman has long, wavy hair that is styled in loose waves and falls down.", ""),
        ("She has bright green eyes.", ""),
        ("She has fair skin and freckles.", ""),
        ("Her eyes are closed in a smile.", "Her eyes are closed in a smile."),
    ],
)
def test_hair_clauses_are_cut_and_fixed_traits_dropped(sentence, kept):
    assert clean_description(sentence) == kept


def test_caption_is_trigger_tags_then_cleaned_description():
    caption = build_caption(
        "mychar",
        "full body, back view, standing, plain white background",
        "She is standing with her back towards the camera. Her hair is long.",
    )

    assert caption == (
        "mychar, full body, back view, standing, plain white background. "
        "She is standing with her back towards the camera."
    )


@pytest.mark.parametrize(
    ("tags", "description", "caption"),
    [
        ("", "She is walking.", "mychar. She is walking."),
        ("full body.", None, "mychar, full body."),
        ("full body", "Her hair is long.", "mychar, full body."),
    ],
)
def test_caption_parts_can_be_missing(tags, description, caption):
    assert build_caption("mychar", tags, description) == caption


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
