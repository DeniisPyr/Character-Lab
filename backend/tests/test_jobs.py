from charlab.config import load_settings
from charlab.pipeline.jobs import Job, plan
from charlab.recipe import Recipe, load_recipe


def shipped_plan(description=None):
    settings = load_settings()
    return plan(load_recipe(settings.recipe), settings.seed, description)


def test_shipped_plan_has_a_job_per_variant_with_references_first():
    jobs = shipped_plan()
    order = [job.key for job in jobs]

    assert len(jobs) == 16
    for job in jobs:
        for reference in job.references:
            assert order.index(reference) < order.index(job.key)


def test_shipped_plan_seeds_are_distinct_and_reproducible():
    jobs = shipped_plan()

    assert sorted(job.seed for job in jobs) == list(range(42, 58))
    assert shipped_plan() == jobs


def test_seed_follows_recipe_position_not_generation_order():
    recipe = Recipe.model_validate(
        {
            "name": "test",
            "identity": "Keep it.",
            "views": {
                "view": {
                    "size": [512, 768],
                    "variants": {
                        "a": {"instruction": "A.", "references": ["view/b"]},
                        "b": {"instruction": "B."},
                    },
                }
            },
        }
    )

    jobs = plan(recipe, seed=100)

    assert [(job.key, job.seed) for job in jobs] == [("view/b", 101), ("view/a", 100)]
    assert jobs[1].references == ("view/b",)
    assert jobs[1].size == (512, 768)


def test_description_goes_into_every_prompt():
    jobs = shipped_plan(description="A soft painted illustration.")

    assert all("A soft painted illustration. Keep the" in job.prompt for job in jobs)


def test_job_splits_its_key():
    job = Job("portrait/over_shoulder", "p", (1024, 1024), 1, ())

    assert (job.view, job.variant) == ("portrait", "over_shoulder")
