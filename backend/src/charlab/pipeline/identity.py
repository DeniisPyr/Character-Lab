"""Score how much an image looks like the reference character.

Anime: CCIP from DeepGHS (deepghs/ccip_onnx, OpenRAIL). A feature model turns an image into a
768-number description of the character, and a small metric model turns two descriptions
into a difference: low for the same character. Preprocessing follows dghs-imgutils 0.19.0
(MIT, imgutils/metrics/ccip.py).
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from PIL import Image

from charlab.config import QualitySettings
from charlab.pipeline.onnx import load_onnx

INPUT_SIZE = 384
# CLIP's normalisation, which CCIP was trained with.
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)[:, None, None]
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)[:, None, None]


def encode(image: Image.Image) -> np.ndarray:
    """The feature model's input: the whole image squashed to 384x384, normalised, (3, H, W)."""
    resized = image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255
    return (pixels - MEAN) / STD


# Takes a model file to load and returns an onnxruntime session. Tests pass a fake.
Loader = Callable[[Any], Any]


class Ccip:
    """CCIP features and differences; both models load on first use."""

    def __init__(self, settings: QualitySettings, load: Loader = load_onnx):
        self.settings = settings
        self._load = load
        self._features = self._metric = None

    def features(self, image: Image.Image) -> np.ndarray:
        """The character description of one image, shape (768,)."""
        if self._features is None:
            self._features = self._load(self.settings.ccip_features)
        (output,) = self._features.run(["output"], {"input": encode(image)[None]})
        return output[0]

    def difference(self, a: np.ndarray, b: np.ndarray) -> float:
        """How different two characters look: below about 0.18, CCIP sees the same one."""
        return float(self.differences([a, b])[0, 1])

    def differences(self, features: Sequence[np.ndarray]) -> np.ndarray:
        """Every pair's difference in one run of the metric model: (N, N) for N features."""
        if self._metric is None:
            self._metric = self._load(self.settings.ccip_metric)
        (output,) = self._metric.run(["output"], {"input": np.stack(features).astype(np.float32)})
        return output
