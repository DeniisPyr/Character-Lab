from pathlib import Path

import pytest
from PIL import Image

from charlab import __version__, cli
from charlab.cli import main
from charlab.pipeline.stages import StageUnavailable


@pytest.fixture
def reference(tmp_path):
    path = tmp_path / "ref.png"
    image = Image.new("RGB", (512, 768), "white")
    image.paste((200, 0, 0), (200, 300, 300, 500))  # a stand-in subject on a plain background
    image.save(path)
    return str(path)


class FakeGenerator:
    def __init__(self, settings):
        self.settings = settings

    def __call__(self, prompt, images, size, seed):
        return Image.new("RGB", size, "grey")


class NoGpu(FakeGenerator):
    def __call__(self, prompt, images, size, seed):
        raise StageUnavailable("generation needs an NVIDIA GPU with CUDA")


def test_version_flag_prints_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])

    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"charlab {__version__}"


def test_no_arguments_prints_help(capsys):
    assert main([]) == 0
    assert "usage: charlab" in capsys.readouterr().out


def test_dry_run_prints_the_plan(capsys, reference):
    assert main(["generate", reference, "--trigger", "mychar", "--dry-run"]) == 0

    out = capsys.readouterr().out
    assert "16 images from recipe 'lora-character', base seed 42" in out
    assert "(512x768, checked)" in out
    assert f"output:    {Path('results', 'mychar')}" in out
    assert " 1. turnaround/front" in out
    assert "16. scenes/meadow_rest" in out
    assert "+ portrait/front" in out
    assert "Show the character's full body standing upright" in out


def test_dry_run_uses_description_out_and_config(capsys, reference, tmp_path):
    config = tmp_path / "my.yaml"
    config.write_text("seed: 7\n", encoding="utf-8")

    code = main(
        [
            "generate", reference, "--trigger", "mychar", "--dry-run",
            "--description", "A soft painted illustration.",
            "--out", "elsewhere", "--config", str(config),
        ]
    )  # fmt: skip

    out = capsys.readouterr().out
    assert code == 0
    assert "base seed 7" in out
    assert "output:    elsewhere" in out
    assert "A soft painted illustration." in out


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--trigger", "my char"], "trigger must be one word"),
        (["--trigger", "mychar", "--config", "missing.yaml"], "No such file"),
    ],
)
def test_bad_input_is_reported(capsys, reference, extra, message):
    assert main(["generate", reference, "--dry-run", *extra]) == 2
    assert message in capsys.readouterr().err


def test_missing_reference_is_reported(capsys):
    assert main(["generate", "nope.png", "--trigger", "mychar", "--dry-run"]) == 2
    assert "nope.png: not found" in capsys.readouterr().err


def test_invalid_reference_is_reported(capsys, tmp_path):
    path = tmp_path / "tiny.png"
    Image.new("RGB", (64, 64)).save(path)

    assert main(["generate", str(path), "--trigger", "mychar", "--dry-run"]) == 2
    assert "too small" in capsys.readouterr().err


def test_invalid_config_is_reported(capsys, reference, tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text("generation:\n  stpes: 8\n", encoding="utf-8")

    assert main(["generate", reference, "--trigger", "x", "--config", str(config)]) == 2
    assert "Extra inputs" in capsys.readouterr().err


def test_real_run_saves_one_image_per_variant(capsys, monkeypatch, reference, tmp_path):
    monkeypatch.setattr(cli, "QwenGenerator", FakeGenerator)
    out = tmp_path / "out"

    assert main(["generate", reference, "--trigger", "mychar", "--out", str(out)]) == 0

    files = sorted(path.name for path in out.iterdir())
    assert len(files) == 16
    assert "turnaround_front.png" in files
    assert Image.open(out / "scenes_city_walk.png").size == (1280, 720)
    assert f"saved 16 images to {out}" in capsys.readouterr().out


def test_unavailable_stage_is_reported(capsys, monkeypatch, reference, tmp_path):
    monkeypatch.setattr(cli, "QwenGenerator", NoGpu)

    assert main(["generate", reference, "--trigger", "mychar", "--out", str(tmp_path)]) == 2
    assert "needs an NVIDIA GPU" in capsys.readouterr().err


def test_trigger_is_required(reference):
    with pytest.raises(SystemExit) as exc:
        main(["generate", reference, "--dry-run"])

    assert exc.value.code == 2
