"""Caption stage: describe each image with Florence-2, then build the training caption."""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PIL import Image

from charlab.config import CaptionSettings
from charlab.pipeline.stages import StageUnavailable

log = logging.getLogger(__name__)

# The rules below were written against Florence-2's descriptions of generated images. It
# describes the hair in almost every caption, guesses age and build, and ends with filler.
#
# Sentences dropped whole: the character's fixed look, which should bind to the trigger word
# instead; guesses about the person; and filler about the image as a whole. The outfit stays,
# so it can still be changed by prompt.
_DROP = re.compile(
    r"\bhair\b"
    r"|\b(?:blue|brown|green|grey|gray|hazel|amber|dark|black) eyes\b"
    r"|\b(?:fair|pale|light|olive|tan|dark) skin\b|\bfreckles\b|\bmakeup\b"
    r"|\b(?:her|his|their) body\b|\bappears to be in (?:her|his|their)\b|\byears old\b"
    r"|\boverall mood\b|\bfocal point\b|\brendering\b",
    re.IGNORECASE,
)
# "with long, wavy hair" or "and has long dark hair that is styled in loose waves": cut out,
# so the rest of the sentence (the shot, the pose, the outfit) stays. Only attached clauses:
# in "The woman has long hair that ... and falls over her shoulders", the hair is the whole
# sentence, which _DROP then removes.
_HAIR_CLAUSE = re.compile(
    r",?\s+(?:with|and\s+has)\s+(?:[\w-]+,?\s+){0,4}hair\b(?:\s+that\s+[^.]*)?(?=[.,]|\s+and\s)",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def clean_description(text: str) -> str:
    """Remove the character's fixed look and filler from a Florence-2 description."""
    kept = []
    for sentence in _SENTENCE_END.split(text.strip()):
        sentence = _HAIR_CLAUSE.sub("", sentence)
        if sentence and not _DROP.search(sentence):
            kept.append(sentence)
    return " ".join(kept)


def build_caption(trigger: str, tags: str, description: str | None) -> str:
    """The training caption: the trigger word, the recipe's tags, then the cleaned description.

    The tags state what Florence-2 often gets wrong: the view, the framing, the expression.
    """
    head = ", ".join(part for part in (trigger, tags.strip().rstrip(".")) if part) + "."
    body = clean_description(description or "")
    return f"{head} {body}" if body else head


@dataclass(frozen=True)
class Florence:
    """A loaded Florence-2 model with its processor, and where it runs."""

    model: Any
    processor: Any
    device: str
    dtype: Any


# Takes the settings and returns a loaded Florence. Tests pass a fake.
Loader = Callable[[CaptionSettings], Florence]


def load_florence(settings: CaptionSettings) -> Florence:
    """Load Florence-2 on the GPU if there is one, otherwise on a Mac GPU or the CPU.

    The weights (1.55 GB for the large model) are downloaded on first use to the Hugging Face
    cache (HF_HOME).
    """
    try:
        import torch
        from transformers import AutoProcessor, Florence2ForConditionalGeneration
    except ImportError as error:
        raise StageUnavailable(
            "captioning needs torch and transformers; install the package with the [gpu] extra"
        ) from error
    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        # bf16 support on Apple GPUs is incomplete; the model is small enough for float32.
        device, dtype = "mps", torch.float32
    else:
        device, dtype = "cpu", torch.float32

    log.info("loading %s on %s", settings.model, device)
    model = Florence2ForConditionalGeneration.from_pretrained(settings.model, dtype=dtype)
    # The processor the model was saved with; also keeps transformers from warning about it.
    processor = AutoProcessor.from_pretrained(settings.model, use_fast=False)
    return Florence(model.to(device), processor, device, dtype)


class FlorenceCaptioner:
    """The caption stage: returns Florence-2's description of an image.

    The model loads on the first call, after generation has finished with the GPU.
    """

    def __init__(self, settings: CaptionSettings, load: Loader = load_florence):
        self.settings = settings
        self._load = load
        self._florence: Florence | None = None

    def __call__(self, image: Image.Image) -> str:
        if self._florence is None:
            self._florence = self._load(self.settings)
        florence, task = self._florence, self.settings.task
        inputs = florence.processor(text=task, images=image.convert("RGB"), return_tensors="pt")
        ids = florence.model.generate(
            **inputs.to(florence.device, florence.dtype),
            max_new_tokens=self.settings.max_new_tokens,
            num_beams=self.settings.num_beams,
            # Beam search without sampling: the same image always gets the same caption.
            do_sample=False,
        )
        text = florence.processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = florence.processor.post_process_generation(text, task=task, image_size=image.size)
        return parsed[task].strip()
