import json

import numpy as np
from PIL import Image

from charlab.config import load_settings
from charlab.pipeline.export import Export
from charlab.pipeline.faces import Face
from charlab.pipeline.jobs import Attempt, Check, Job, Result

LIMITS = load_settings().quality
FACE = Face((10, 20, 110, 140), 0.8512)
JOBS = [
    Job("turnaround/front", "A.", (64, 96), 42, ()),
    Job("turnaround/back", "B.", (64, 96), 43, (), face=False),
]
PASS = Check(0.0123456, FACE, 0.51234)
FAIL = Check(0.4, FACE, 0.5, ("CCIP difference 0.400: not the same character",))


def write(tmp_path, results):
    path = Export(tmp_path, JOBS).write_report(results, LIMITS)
    return json.loads(path.read_text("utf-8"))


def image(name):
    """Distinct test images: a wave pattern, mirrored for "blue". (Plain colours would all
    share one fingerprint.)"""
    y, x = np.indices((96, 64)) / 64
    pattern = Image.fromarray((127 + 127 * np.sin(6 * x) * np.cos(4 * y)).astype(np.uint8))
    pattern = pattern.convert("RGB")
    return pattern.transpose(Image.FLIP_LEFT_RIGHT) if name == "blue" else pattern


def test_report_records_every_attempt_and_the_kept_one(tmp_path):
    results = [
        Result(JOBS[0], image("red"), None, (Attempt(42, FAIL), Attempt(1042, PASS)), kept=1),
        Result(JOBS[1], image("blue"), None, (Attempt(43, Check(0.02)),)),
    ]

    report = write(tmp_path, results)

    assert report["summary"] == {
        "images": 2,
        "passed": 2,
        "retried": ["001_turnaround_front.png"],
        "failed": [],
        "near_duplicates": 0,
    }
    front = report["images"][0]
    assert (front["image"], front["passed"], front["kept_attempt"]) == (
        "001_turnaround_front.png",
        True,
        2,
    )
    assert front["attempts"][0]["failures"] == list(FAIL.failures)
    assert front["attempts"][1] == {
        "seed": 1042,
        "passed": True,
        "identity": 0.0123,
        "face": {"box": [10, 20, 110, 140], "size": 100, "confidence": 0.851},
        "sharpness": 0.512,
        "failures": [],
    }
    assert report["images"][1]["attempts"][0]["face"] is None
    assert report["limits"]["same_character_below"] == LIMITS.same_character_below


def test_an_image_that_failed_every_attempt_is_listed(tmp_path):
    results = [
        Result(JOBS[0], image("red"), None, (Attempt(42, FAIL),) * 3),
        Result(JOBS[1], image("blue"), None, (Attempt(43, PASS),)),
    ]

    report = write(tmp_path, results)

    assert report["summary"]["failed"] == ["001_turnaround_front.png"]
    assert report["summary"]["passed"] == 1


def test_near_exact_copies_are_reported(tmp_path):
    same = image("red")
    results = [
        Result(JOBS[0], same, None, (Attempt(42, PASS),)),
        Result(JOBS[1], same.copy(), None, (Attempt(43, PASS),)),
    ]

    report = write(tmp_path, results)

    assert report["near_duplicates"] == [
        {"a": "001_turnaround_front.png", "b": "002_turnaround_back.png", "different_bits": 0}
    ]


def test_without_a_gate_the_attempts_have_no_scores(tmp_path):
    results = [Result(job, image("red"), None, (Attempt(job.seed, None),)) for job in JOBS]

    report = write(tmp_path, results)

    assert report["images"][0]["passed"] is None
    assert report["images"][0]["attempts"] == [{"seed": 42, "passed": None}]
