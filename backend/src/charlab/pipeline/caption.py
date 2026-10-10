"""Caption stage: describe each image with Florence-2."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PIL import Image

from charlab.config import CaptionSettings
from charlab.pipeline.stages import StageUnavailable

log = logging.getLogger(__name__)


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
