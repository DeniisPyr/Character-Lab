import pytest
from pydantic import ValidationError

from charlab.config import load_settings, merge
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
