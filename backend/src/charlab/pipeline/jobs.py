"""Generation jobs: one per recipe variant, in the order they must run."""

from dataclasses import dataclass

from PIL import Image

from charlab.pipeline.faces import Face
from charlab.recipe import Recipe


@dataclass(frozen=True)
class Job:
    key: str  # "view/variant"
    prompt: str
    size: tuple[int, int]  # (width, height)
    seed: int
    references: tuple[str, ...]  # keys of earlier jobs passed as Picture 2, 3
    caption: str = ""  # the recipe's caption tags
    face: bool = True  # whether the shot shows the face (not from behind)

    @property
    def view(self) -> str:
        return self.key.split("/")[0]

    @property
    def variant(self) -> str:
        return self.key.split("/")[1]


@dataclass(frozen=True)
class Check:
    """The quality gate's scores for one image."""

    identity: float  # CCIP difference from the reference; low means the same character
    face: Face | None = None  # the most confident face, if the shot should show one
    sharpness: float | None = None  # the face's sharpness relative to the reference face
    failures: tuple[str, ...] = ()  # why it failed; empty if it passed

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class Attempt:
    seed: int
    check: Check | None  # None without a quality gate


@dataclass(frozen=True)
class Repair:
    """A face repair: the repaired image's check, and whether it replaced the original."""

    seed: int
    check: Check
    kept: bool
    reason: str


@dataclass(frozen=True)
class Result:
    job: Job
    image: Image.Image
    description: str | None  # from the caption stage, before the caption is built
    attempts: tuple[Attempt, ...] = ()
    kept: int = 0  # the attempt whose image this is
    repair: Repair | None = None

    @property
    def seed(self) -> int:
        """The seed of the image that was kept."""
        return self.attempts[self.kept].seed if self.attempts else self.job.seed

    @property
    def check(self) -> Check | None:
        """The final image's check: the repair's if a repair was kept."""
        if self.repair and self.repair.kept:
            return self.repair.check
        return self.attempts[self.kept].check if self.attempts else None


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
            face=recipe.variant(key).face,
        )
        for key in recipe.generation_order()
    ]
