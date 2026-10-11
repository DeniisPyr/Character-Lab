"""Generation jobs: one per recipe variant, in the order they must run."""

from dataclasses import dataclass

from PIL import Image

from charlab.recipe import Recipe


@dataclass(frozen=True)
class Job:
    key: str  # "view/variant"
    prompt: str
    size: tuple[int, int]  # (width, height)
    seed: int
    references: tuple[str, ...]  # keys of earlier jobs passed as Picture 2, 3
    caption: str = ""  # the recipe's caption tags

    @property
    def view(self) -> str:
        return self.key.split("/")[0]

    @property
    def variant(self) -> str:
        return self.key.split("/")[1]


@dataclass(frozen=True)
class Result:
    job: Job
    image: Image.Image
    description: str | None  # from the caption stage, before the caption is built


def plan(recipe: Recipe, seed: int, description: str | None = None) -> list[Job]:
    """Jobs in generation order. Each variant's seed is the base seed plus its position in
    the recipe, so seeds stay the same when only the generation order changes."""
    position = {key: index for index, (key, _) in enumerate(recipe.variants())}
    return [
        Job(
            key=key,
            prompt=recipe.prompt(key, description),
            size=recipe.size(key),
            seed=seed + position[key],
            references=recipe.variant(key).references,
            caption=recipe.variant(key).caption,
        )
        for key in recipe.generation_order()
    ]
