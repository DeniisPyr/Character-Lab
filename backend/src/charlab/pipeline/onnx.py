"""Load the small ONNX models used by the quality checks."""

from typing import Any

from charlab.config import HubFile
from charlab.pipeline.stages import StageUnavailable

# These models are small and run once per image. A few threads are plenty, and a container
# that sees all of the host's CPUs would otherwise start one thread per CPU.
THREADS = 4


def load_onnx(model: HubFile) -> Any:
    """Download the model file once (to HF_HOME) and open it on the CPU."""
    try:
        import onnxruntime
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise StageUnavailable(
            "the quality checks need onnxruntime; install the package with the [gpu] extra"
        ) from error
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = THREADS
    return onnxruntime.InferenceSession(
        hf_hub_download(model.repo, model.file),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
