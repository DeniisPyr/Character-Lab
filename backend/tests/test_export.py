import json
import zipfile

from PIL import Image

from charlab.pipeline.export import Export
from charlab.pipeline.jobs import Job, Result

JOBS = [
    Job("turnaround/front", "Show the front.", (64, 96), 42, (), "full body, front view"),
    Job("scenes/city_walk", "Walk in a city.", (96, 64), 43, ("turnaround/front",), ""),
]
RUN = {"recipe": "test", "seed": 42}


def results(descriptions=("A woman stands. Her hair is long.", None)):
    return [
        Result(job, Image.new("RGB", job.size, "grey"), description)
        for job, description in zip(JOBS, descriptions, strict=True)
    ]


def test_images_are_numbered_in_generation_order(tmp_path):
    export = Export(tmp_path / "out", JOBS)

    for job in reversed(JOBS):
        export.save_image(job, Image.new("RGB", job.size))

    names = sorted(path.name for path in (tmp_path / "out").iterdir())
    assert names == ["001_turnaround_front.png", "002_scenes_city_walk.png"]
    assert Image.open(tmp_path / "out" / "002_scenes_city_walk.png").size == (96, 64)


def test_each_image_gets_a_caption_file_of_the_same_name(tmp_path):
    out = tmp_path / "out"

    Export(out, JOBS).write_captions(results(), "mychar", RUN)

    assert (out / "001_turnaround_front.txt").read_text("utf-8") == (
        "mychar, full body, front view. A woman stands.\n"
    )
    assert (out / "002_scenes_city_walk.txt").read_text("utf-8") == "mychar.\n"


def test_manifest_records_the_run_and_every_image(tmp_path):
    manifest = Export(tmp_path, JOBS).write_captions(results(), "mychar", RUN)

    data = json.loads(manifest.read_text("utf-8"))
    assert (data["recipe"], data["seed"], data["trigger"]) == ("test", 42, "mychar")
    first = data["images"][0]
    assert first == {
        "image": "001_turnaround_front.png",
        "caption_file": "001_turnaround_front.txt",
        "view": "turnaround",
        "variant": "front",
        "size": [64, 96],
        "seed": 42,
        "prompt": "Show the front.",
        "caption": "mychar, full body, front view. A woman stands.",
    }
    assert [image["seed"] for image in data["images"]] == [42, 43]


def test_zip_packs_the_dataset_next_to_its_folder(tmp_path):
    export = Export(tmp_path / "mychar", JOBS)
    for result in results():
        export.save_image(result.job, result.image)
    export.write_captions(results(), "mychar", RUN)

    archive = export.zip()

    assert archive == tmp_path / "mychar.zip"
    with zipfile.ZipFile(archive) as packed:
        assert sorted(packed.namelist()) == [
            "001_turnaround_front.png",
            "001_turnaround_front.txt",
            "002_scenes_city_walk.png",
            "002_scenes_city_walk.txt",
            "manifest.json",
        ]
