import json

import numpy as np

from charlab.eval.metrics import ImageScore, Scores, summarise
from charlab.eval.report import render, write_eval
from charlab.pipeline.faces import Face

RUN = {
    "charlab": "1.2.3",
    "dataset": "mychar",
    "reference": "ref.png",
    "same_character_below": 0.1785,
}
FRONT = ImageScore(
    "001_front.png", (832, 1216), 0.0123456, Face((10, 20, 110, 140), 0.8567), 0.004321, 0.29,
    "mychar, front view. She smiles.",
)  # fmt: skip
BACK = ImageScore("002_back.png", (832, 1216), 0.063, None, None, None, None)
SCORES = Scores(
    (FRONT, BACK), np.array([[0, 0.1], [0.1, 0]]), np.array([[0, 5], [5, 0]]), ("003_x.txt",)
)


def report(tmp_path):
    summary = summarise(SCORES, RUN["same_character_below"])
    return write_eval(tmp_path, SCORES.images, summary, RUN)


def test_json_has_the_run_the_summary_and_every_image(tmp_path):
    data, _ = report(tmp_path)

    content = json.loads(data.read_text("utf-8"))
    assert {key: content[key] for key in RUN} == RUN
    assert content["summary"]["identity"]["worst_image"] == "002_back.png"
    assert content["images"] == [
        {
            "image": "001_front.png",
            "size": [832, 1216],
            "identity": 0.0123,
            "face": {"box": [10, 20, 110, 140], "size": 100, "confidence": 0.857},
            "face_identity": 0.0043,
            "sharpness": 0.29,
            "caption_words": 5,
        },
        {
            "image": "002_back.png",
            "size": [832, 1216],
            "identity": 0.063,
            "face": None,
            "face_identity": None,
            "sharpness": None,
            "caption_words": None,
        },
    ]


def test_markdown_has_the_summary_counts_and_image_tables(tmp_path):
    _, text = report(tmp_path)

    markdown = text.read_text("utf-8")
    assert markdown.startswith("# Evaluation of mychar\n\n2 images against the reference `ref.png`")
    assert (
        "| Identity: CCIP difference from the reference | lower | 0.038 | 0.063 | 002_back.png |"
        in markdown
    )
    assert "| 0.100 | 0.100 | 001_front.png and 002_back.png |" in markdown
    assert "| Diversity: fingerprint bits that differ, of 64 | higher | 5 | 5 |" in markdown
    assert "- **Faces found:** 1 of 2 (none in 002_back.png)" in markdown
    assert "- **Sizes:** 2 × 832×1216" in markdown
    assert "  - No caption: 002_back.png\n  - Captions without an image: 003_x.txt" in markdown
    assert "| 001_front.png | 832×1216 | 0.012 | 0.004 | 100 | 0.29 | 5 |" in markdown
    assert "| 002_back.png | 832×1216 | 0.063 | – | – | – | – |" in markdown


def test_scores_with_nothing_to_measure_show_a_dash():
    scores = Scores((BACK,), np.zeros((1, 1)), np.zeros((1, 1), dtype=int), ())
    summary = summarise(scores, RUN["same_character_below"])

    markdown = render({**RUN, "summary": summary, "images": []})

    assert "| Consistency: CCIP difference between two images | lower | – | – | – |" in markdown
    assert "- **Captions:** 0 of 1 images\n  - No caption: 002_back.png" in markdown
