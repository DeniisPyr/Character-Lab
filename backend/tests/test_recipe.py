from pathlib import Path

import pytest
from pydantic import ValidationError

from charlab.recipe import Recipe, load_recipe

SHIPPED = Path(__file__).parents[1] / "configs" / "recipes" / "lora-character.yaml"


def make_recipe(variants, size=(1024, 1024), blocks=None):
    return {
        "name": "test",
        "identity": "Identity E.",
        "blocks": {"b": "Block B."} if blocks is None else blocks,
        "views": {"view": {"size": list(size), "variants": variants}},
    }


def test_shipped_recipe_has_four_views_of_four_variants():
    recipe = load_recipe(SHIPPED)

    assert list(recipe.views) == ["turnaround", "portrait", "expressions", "scenes"]
    assert [len(view.variants) for view in recipe.views.values()] == [4, 4, 4, 4]


def test_shipped_prompts_are_single_lines():
    recipe = load_recipe(SHIPPED)

    for key, _ in recipe.variants():
        prompt = recipe.prompt(key)
        assert "\n" not in prompt
        assert prompt.endswith(recipe.identity)


def test_prompt_order_is_instruction_blocks_setting_description_identity():
    recipe = Recipe.model_validate(
        make_recipe({"a": {"instruction": "Do A.", "blocks": ["b"], "setting": "Setting C."}})
    )

    assert recipe.prompt("view/a") == "Do A. Block B. Setting C. Identity E."
    assert recipe.prompt("view/a", description=" Desc D. ") == (
        "Do A. Block B. Setting C. Desc D. Identity E."
    )


def test_generation_order_puts_references_first():
    recipe = load_recipe(SHIPPED)
    order = recipe.generation_order()

    assert sorted(order) == sorted(key for key, _ in recipe.variants())
    for key, variant in recipe.variants():
        for reference in variant.references:
            assert order.index(reference) < order.index(key)


def test_generation_order_keeps_file_order_otherwise():
    recipe = Recipe.model_validate(
        make_recipe(
            {
                "a": {"instruction": "A.", "references": ["view/c"]},
                "b": {"instruction": "B."},
                "c": {"instruction": "C."},
            }
        )
    )

    assert recipe.generation_order() == ["view/b", "view/c", "view/a"]


@pytest.mark.parametrize(
    ("variants", "message"),
    [
        ({"a": {"instruction": "A.", "blocks": ["nope"]}}, "unknown blocks"),
        ({"a": {"instruction": "A.", "references": ["view/nope"]}}, "unknown reference"),
        ({"a": {"instruction": "A.", "references": ["view/a"]}}, "cannot reference itself"),
        (
            {
                "a": {"instruction": "A.", "references": ["view/b"]},
                "b": {"instruction": "B.", "references": ["view/a"]},
            },
            "cycle",
        ),
        (
            {
                "a": {"instruction": "A.", "references": ["view/b", "view/c", "view/d"]},
                "b": {"instruction": "B."},
                "c": {"instruction": "C."},
                "d": {"instruction": "D."},
            },
            "at most 2 references",
        ),
        ({"a": {"instruction": "A.", "refrences": ["view/b"]}}, "Extra inputs"),
        ({"Front View": {"instruction": "A."}}, "snake_case"),
        ({}, "at least one variant"),
    ],
)
def test_invalid_recipes_are_rejected(variants, message):
    with pytest.raises(ValidationError, match=message):
        Recipe.model_validate(make_recipe(variants))


def test_size_must_divide_by_16():
    with pytest.raises(ValidationError, match="multiples of 16"):
        Recipe.model_validate(make_recipe({"a": {"instruction": "A."}}, size=(1000, 1000)))


def test_load_recipe_reads_yaml(tmp_path):
    path = tmp_path / "recipe.yaml"
    path.write_text(
        "name: tiny\n"
        "identity: Keep it.\n"
        "views:\n"
        "  portrait:\n"
        "    size: [512, 512]\n"
        "    variants:\n"
        "      front:\n"
        "        instruction: >-\n"
        "          Face the\n"
        "          camera.\n",
        encoding="utf-8",
    )

    recipe = load_recipe(path)

    assert recipe.size("portrait/front") == (512, 512)
    assert recipe.prompt("portrait/front") == "Face the camera. Keep it."
