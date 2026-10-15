import pytest
from pydantic import ValidationError

from charlab.config import CONFIGS, find_config, load_settings, merge
from charlab.recipe import load_recipe


def write(tmp_path, text, name="override.yaml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_load():
    settings = load_settings()

    assert settings.seed == 42
    assert settings.generation.model == "Qwen/Qwen-Image-Edit-2511"
    assert (settings.generation.steps, settings.generation.cfg) == (4, 1.0)
    assert settings.prepare.background_model == "birefnet-general"
    assert settings.generation.transformer == "bf16"
    assert (settings.face_repair.mode, settings.face_repair.use_reference) == ("off", True)


def test_default_recipe_exists_and_is_valid():
    settings = load_settings()

    assert settings.recipe.is_file()
    assert load_recipe(settings.recipe).name == "lora-character"


def test_override_changes_only_what_it_sets(tmp_path):
    settings = load_settings(write(tmp_path, "generation:\n  steps: 8\n"))

    assert settings.generation.steps == 8
    assert settings.generation.model == "Qwen/Qwen-Image-Edit-2511"
    assert settings.seed == 42


def test_empty_override_keeps_defaults(tmp_path):
    assert load_settings(write(tmp_path, "")) == load_settings()


def test_recipe_path_is_relative_to_the_file_that_sets_it(tmp_path):
    settings = load_settings(write(tmp_path, "recipe: my-recipe.yaml\n"))

    assert settings.recipe == tmp_path / "my-recipe.yaml"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("generation:\n  stpes: 8\n", "Extra inputs"),
        ("generation:\n  steps: 0\n", "greater than or equal to 1"),
        ("generation:\n  cfg: -1\n", "greater than or equal to 0"),
        ("seed: -5\n", "greater than or equal to 0"),
        ("prepare:\n  background: blur\n", "'auto', 'remove' or 'keep'"),
        ("generation:\n  transformer: fp8\n", "'bf16' or 'gguf'"),
        ("style: realistic\n", "Input should be 'anime'"),
        ("face_repair:\n  mode: sometimes\n", "'off', 'failed' or 'small'"),
        ("caption:\n  task: '<OD>'\n", "'<CAPTION>', '<DETAILED_CAPTION>' or"),
        ("prepare:\n  crop_margin: 2\n", "less than or equal to 1"),
    ],
)
def test_invalid_overrides_are_rejected(tmp_path, text, message):
    with pytest.raises(ValidationError, match=message):
        load_settings(write(tmp_path, text))


def test_merge_is_deep_and_leaves_inputs_unchanged():
    base = {"a": 1, "nested": {"x": 1, "y": 2}}
    override = {"nested": {"y": 3}, "b": 4}

    assert merge(base, override) == {"a": 1, "nested": {"x": 1, "y": 3}, "b": 4}
    assert base == {"a": 1, "nested": {"x": 1, "y": 2}}


def test_a_bare_off_switches_face_repair_off(tmp_path):
    # YAML reads `off` as false; the setting still understands it.
    assert load_settings(write(tmp_path, "face_repair:\n  mode: off\n")).face_repair.mode == "off"


def test_the_24gb_config_is_found_by_name_from_any_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert find_config("24gb.yaml") == CONFIGS / "24gb.yaml"
    assert load_settings("24gb.yaml").generation.transformer == "gguf"


def test_a_users_own_config_wins_over_a_ready_made_one(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write(tmp_path, "seed: 7\n", name="24gb.yaml")

    settings = load_settings("24gb.yaml")

    assert (settings.seed, settings.generation.transformer) == (7, "bf16")


def test_only_bare_names_fall_back_to_ready_made_configs(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "24gb.yaml")
    with pytest.raises(FileNotFoundError):
        load_settings("missing.yaml")
