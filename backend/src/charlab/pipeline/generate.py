"""Generate stage: Qwen-Image-Edit-2511 with the Lightning 4-step LoRA."""

import math
from collections.abc import Callable, Sequence
from typing import Any

from PIL import Image

from charlab.config import GenerationSettings

# Sampler settings for the Lightning LoRA, from ModelTC's generate_with_diffusers.py
# (github.com/ModelTC/Qwen-Image-Lightning).
LIGHTNING_SCHEDULER = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

# Takes the settings and returns a loaded pipeline. Tests pass a fake.
Loader = Callable[[GenerationSettings], Any]


def load_pipeline(settings: GenerationSettings) -> Any:
    """Load Qwen-Image-Edit in bf16 on the GPU, with the Lightning LoRA fused in.

    Needs about 62-68 GB of GPU memory. The weights (about 58 GB) are downloaded on first use
    to the Hugging Face cache (HF_HOME, default ~/.cache/huggingface).
    """
    try:
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
    except ImportError as error:
        raise RuntimeError(
            "generation needs torch and diffusers; install the package with the [gpu] extra"
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("generation needs an NVIDIA GPU with CUDA")

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        settings.model,
        scheduler=FlowMatchEulerDiscreteScheduler.from_config(LIGHTNING_SCHEDULER),
        torch_dtype=torch.bfloat16,
    )
    pipe.load_lora_weights(settings.lightning_lora.repo, weight_name=settings.lightning_lora.file)
    # Fused into the base weights, the LoRA costs nothing per step.
    pipe.fuse_lora()
    pipe.unload_lora_weights()
    # The runner logs progress per image; per-step bars would bury it.
    pipe.set_progress_bar_config(disable=True)
    return pipe.to("cuda")


def _generator(seed: int) -> Any:
    import torch

    # Noise drawn on the CPU does not depend on the GPU model.
    return torch.Generator("cpu").manual_seed(seed)


class QwenGenerator:
    """The generate stage.

    The pipeline loads on the first call, so a run that fails earlier (a bad reference, say)
    never spends minutes loading weights.
    """

    def __init__(self, settings: GenerationSettings, load: Loader = load_pipeline):
        self.settings = settings
        self._load = load
        self._pipe = None

    def __call__(
        self, prompt: str, images: Sequence[Image.Image], size: tuple[int, int], seed: int
    ) -> Image.Image:
        if self._pipe is None:
            self._pipe = self._load(self.settings)
        width, height = size
        return self._pipe(
            image=list(images),
            prompt=prompt,
            # Classifier-free guidance runs only with a negative prompt and a scale above 1;
            # the Lightning LoRA is distilled to run without it.
            negative_prompt=" " if self.settings.cfg > 1 else None,
            true_cfg_scale=self.settings.cfg,
            width=width,
            height=height,
            num_inference_steps=self.settings.steps,
            generator=_generator(seed),
        ).images[0]
