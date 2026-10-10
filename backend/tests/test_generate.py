import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from charlab.config import load_settings
from charlab.pipeline import generate
from charlab.pipeline.generate import QwenGenerator, check_gpu, load_pipeline
from charlab.pipeline.stages import StageUnavailable

SETTINGS = load_settings().generation


class FakePipe:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(images=[Image.new("RGB", (kwargs["width"], kwargs["height"]))])


@pytest.fixture
def pipe(monkeypatch):
    monkeypatch.setattr(generate, "_generator", lambda seed: f"generator({seed})")
    return FakePipe()


def make(pipe, settings=SETTINGS):
    loads = []

    def load(settings):
        loads.append(settings)
        return pipe

    return QwenGenerator(settings, load), loads


def test_pipeline_loads_once_on_the_first_call(pipe):
    generator, loads = make(pipe)
    assert loads == []

    generator("A.", [Image.new("RGB", (64, 64))], (832, 1216), 1)
    generator("B.", [Image.new("RGB", (64, 64))], (832, 1216), 2)

    assert loads == [SETTINGS]


def test_call_passes_images_size_steps_and_seed(pipe):
    images = [Image.new("RGB", (64, 64)), Image.new("RGB", (32, 32))]

    result = make(pipe)[0]("Turn around.", images, (832, 1216), 7)

    call = pipe.calls[0]
    assert call["image"] == images
    assert call["prompt"] == "Turn around."
    assert (call["width"], call["height"]) == (832, 1216)
    assert result.size == (832, 1216)
    assert (call["num_inference_steps"], call["true_cfg_scale"]) == (4, 1.0)
    assert call["negative_prompt"] is None
    assert call["generator"] == "generator(7)"


def test_guidance_above_1_sends_a_negative_prompt(pipe):
    settings = SETTINGS.model_copy(update={"cfg": 4.0})

    make(pipe, settings)[0]("A.", [Image.new("RGB", (64, 64))], (1024, 1024), 1)

    assert pipe.calls[0]["negative_prompt"] == " "


def fake_torch(available=True, memory_gb=80, cuda="13.0"):
    props = SimpleNamespace(total_memory=memory_gb * 1024**3)
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: available, get_device_properties=lambda i: props),
        version=SimpleNamespace(cuda=cuda),
    )


@pytest.mark.parametrize(("memory_gb", "transformer"), [(80, "bf16"), (24, "gguf")])
def test_big_enough_gpu_passes_the_check(memory_gb, transformer):
    check_gpu(fake_torch(memory_gb=memory_gb), transformer)


@pytest.mark.parametrize(
    ("torch", "transformer", "nvidia_smi", "message"),
    [
        (fake_torch(memory_gb=24), "bf16", True, "take 58 GB, and this GPU has 24 GB; .* gguf"),
        (fake_torch(memory_gb=48), "bf16", True, "needs an 80 GB GPU"),
        (fake_torch(memory_gb=12), "gguf", True, "text encoder alone takes 17 GB, .* has 12 GB"),
        (fake_torch(available=False), "bf16", False, "needs an NVIDIA GPU and none was found"),
        (fake_torch(available=False, cuda=None), "bf16", True, "no CUDA support; .*whl/cu130"),
        (fake_torch(available=False), "bf16", True, "built for CUDA 13.0, .*whl/cu126"),
    ],
    ids=["bf16 on 24 GB", "bf16 on 48 GB", "gguf on 12 GB", "no GPU", "CPU-only", "old driver"],
)
def test_unusable_gpu_is_reported_with_a_fix(monkeypatch, torch, transformer, nvidia_smi, message):
    monkeypatch.setattr(generate.shutil, "which", lambda name: "/usr/bin/x" if nvidia_smi else None)

    with pytest.raises(StageUnavailable, match=message):
        check_gpu(torch, transformer)


def test_missing_gpu_packages_are_reported(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "diffusers", None)

    with pytest.raises(StageUnavailable, match=r"needs torch and diffusers; .* \[gpu\] extra"):
        load_pipeline(SETTINGS)
