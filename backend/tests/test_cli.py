import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from charlab import __version__, cli
from charlab.cli import main
from charlab.pipeline.jobs import Check
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
        self.calls = 0

    def __call__(self, prompt, images, size, seed):
        self.calls += 1
        return Image.new("RGB", size, "grey")

    def unload(self):
        pass


class NoGpu(FakeGenerator):
    def __call__(self, prompt, images, size, seed):
        raise StageUnavailable("generation needs an NVIDIA GPU with CUDA")


class CrashOnThird(FakeGenerator):
    def __call__(self, prompt, images, size, seed):
        if self.calls == 2:
            raise RuntimeError("CUDA out of memory")
        return super().__call__(prompt, images, size, seed)


class FakeCaptioner:
    def __init__(self, settings):
        self.settings = settings

    def __call__(self, image):
        return "A woman stands in a studio. Her hair is long and wavy."


class FakeGate:
    """Passes every image, except the first attempt of turnaround/front when told to."""

    fail_first_front = False

    def __init__(self, settings):
        self.seen = set()

    def start(self, reference):
        pass

    def __call__(self, job, image):
        first_try = job.key not in self.seen
        self.seen.add(job.key)
        if self.fail_first_front and job.key == "turnaround/front" and first_try:
            return Check(0.5, failures=("CCIP difference 0.500: not the same character",))
        return Check(0.01)


class FakeRepair:
    """Never repairs; records that the CLI created it."""

    created = []

    def __init__(self, settings, generate, gate):
        FakeRepair.created.append(settings.mode)

    def start(self, reference):
        pass

    def __call__(self, job, image, check, seed):
        return None


@pytest.fixture
def fake_models(monkeypatch):
    monkeypatch.setattr(cli, "QwenGenerator", FakeGenerator)
    monkeypatch.setattr(cli, "FlorenceCaptioner", FakeCaptioner)
    monkeypatch.setattr(cli, "QualityGate", FakeGate)
    monkeypatch.setattr(cli, "FaceRepairStage", FakeRepair)
    monkeypatch.setattr(FakeRepair, "created", [])
    monkeypatch.setattr(FakeGate, "fail_first_front", False)


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
    assert "caption: mychar, full body, front view, standing, plain white background." in out
    assert "quality:   anime checks, up to 3 attempts per image" in out
    assert "repair:    failed faces" in out


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


def test_real_run_writes_the_captioned_dataset(capsys, fake_models, reference, tmp_path):
    out = tmp_path / "out"

    assert main(["generate", reference, "--trigger", "mychar", "--out", str(out)]) == 0

    names = {path.name for path in out.iterdir()}
    assert len(names) == 16 * 2 + 2
    assert {"001_turnaround_front.png", "016_scenes_meadow_rest.txt", "manifest.json"} <= names
    report = json.loads((out / "report.json").read_text("utf-8"))
    assert report["summary"]["passed"] == 16
    assert report["summary"]["retried"] == report["summary"]["failed"] == []
    assert Image.open(out / "013_scenes_city_walk.png").size == (1280, 720)
    assert (out / "001_turnaround_front.txt").read_text("utf-8") == (
        "mychar, full body, front view, standing, plain white background. "
        "A woman stands in a studio.\n"
    )
    manifest = json.loads((out / "manifest.json").read_text("utf-8"))
    assert (manifest["trigger"], manifest["reference"]) == ("mychar", "ref.png")
    assert manifest["generation"]["transformer"] == "bf16"
    assert len(manifest["images"]) == 16
    printed = capsys.readouterr().out
    assert f"saved 16 images with captions to {out}" in printed
    assert "quality: 16/16 images passed every check" in printed


def test_a_failed_image_is_regenerated_with_a_new_seed(fake_models, reference, tmp_path):
    FakeGate.fail_first_front = True
    out = tmp_path / "out"

    assert main(["generate", reference, "--trigger", "mychar", "--out", str(out)]) == 0

    manifest = json.loads((out / "manifest.json").read_text("utf-8"))
    assert [image["seed"] for image in manifest["images"][:2]] == [42 + 1000, 43]
    report = json.loads((out / "report.json").read_text("utf-8"))
    assert report["summary"]["retried"] == ["001_turnaround_front.png"]
    front = report["images"][0]
    assert [attempt["passed"] for attempt in front["attempts"]] == [False, True]
    assert front["kept_attempt"] == 2


def test_zip_option_packs_the_dataset(capsys, fake_models, reference, tmp_path):
    out = tmp_path / "mychar"

    assert main(["generate", reference, "--trigger", "mychar", "--out", str(out), "--zip"]) == 0

    with zipfile.ZipFile(tmp_path / "mychar.zip") as packed:
        assert len(packed.namelist()) == 34
    assert "packed the dataset into" in capsys.readouterr().out


def test_images_made_before_a_crash_are_kept(fake_models, monkeypatch, reference, tmp_path):
    monkeypatch.setattr(cli, "QwenGenerator", CrashOnThird)

    with pytest.raises(RuntimeError, match="out of memory"):
        main(["generate", reference, "--trigger", "mychar", "--out", str(tmp_path / "out")])

    names = sorted(path.name for path in (tmp_path / "out").iterdir())
    assert names == ["001_turnaround_front.png", "002_turnaround_three_quarter.png"]


def test_unavailable_stage_is_reported(capsys, fake_models, monkeypatch, reference, tmp_path):
    monkeypatch.setattr(cli, "QwenGenerator", NoGpu)

    assert main(["generate", reference, "--trigger", "mychar", "--out", str(tmp_path)]) == 2
    assert "needs an NVIDIA GPU" in capsys.readouterr().err


def test_trigger_is_required(reference):
    with pytest.raises(SystemExit) as exc:
        main(["generate", reference, "--dry-run"])

    assert exc.value.code == 2


def test_face_repair_runs_unless_switched_off(fake_models, reference, tmp_path):
    off = tmp_path / "off.yaml"
    off.write_text("face_repair:\n  mode: off\n", encoding="utf-8")

    assert main(["generate", reference, "--trigger", "a", "--out", str(tmp_path / "a")]) == 0
    assert main(["generate", reference, "--trigger", "b", "--out", str(tmp_path / "b"),
                 "--config", str(off)]) == 0  # fmt: skip

    assert FakeRepair.created == ["failed"]
