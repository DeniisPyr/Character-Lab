# Character Lab

[![CI](https://github.com/DeniisPyr/Character-Lab/actions/workflows/ci.yml/badge.svg)](https://github.com/DeniisPyr/Character-Lab/actions/workflows/ci.yml)

Turns one character image into a training-ready LoRA dataset.

From a single reference image, Character Lab generates 16 consistent images of the same
character with [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511):

| View | Images | Size |
|---|---|---|
| Turnaround | front, three-quarter, side, back | 832×1216 |
| Portrait | front, three-quarter, profile, over the shoulder | 1024×1024 |
| Expressions | joy, surprise, anger, sadness | 1024×1024 |
| Scenes | city walk, hilltop sunset, lantern market, meadow | 1280×720 |

> **Status: work in progress.** Reference preparation and image generation work and are
> tested on a GPU. Captions and dataset export come next (v0.1.0), followed by face
> refinement, a quality gate and evaluation.

## How it works

1. **Prepare.** The reference is checked (format, size, integrity) and cropped to the
   character.
   - No model is needed when the background is transparent or one plain colour.
   - Otherwise [rembg](https://github.com/danielgatis/rembg) with BiRefNet cuts the character
     out and puts it on white.
2. **Generate.** Each image is an instruction-style edit of the reference (Picture 1).
   Later views also get an earlier result as Picture 2, for example the front view for
   the back view, which keeps the outfit and face consistent. The dataset is described in
   a YAML recipe, [backend/configs/recipes/lora-character.yaml](backend/configs/recipes/lora-character.yaml).
   The Lightning LoRA cuts sampling to 4 steps.

Every image has its own seed, so a run is reproducible: the same reference, seed and
settings give byte-identical images. Repeat runs on two different A100 machines matched
exactly.

## Requirements

| What you want to do | What you need |
|---|---|
| Install, run the tests, print the plan (`--dry-run`) | Python 3.11+ on Linux, macOS or Windows |
| Generate images | An NVIDIA GPU, plus disk space for the models (see below) |

Generation has two modes, set with `generation.transformer`:

| | `bf16` (default) | `gguf` |
|---|---|---|
| GPU | 80 GB (A100, H100) | 24 GB (RTX 3090, RTX 4090) |
| What runs | the full model, all on the GPU | a [quantized transformer](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF) (Q6_K); each model moves to the GPU only while it runs |
| Peak GPU memory | 61.4 GiB | 21.0 GiB |
| First download | 58.6 GB | about 35 GB |
| 16 images | about 8.5 min: 5.5 min loading, 2.5 min generating | about 17 min: 3.7 min loading, 13 min generating |

These numbers were measured on one A100 80 GB, in October 2026. The `gguf` numbers come from
the same A100 with PyTorch limited to 23 GiB of GPU memory, to act like a 24 GB card; times
on a real RTX 4090 will differ. Side by side, the `gguf` images look the same as the `bf16`
ones. In `gguf` mode the models wait in system RAM between uses, so plan for about 48 GB of
RAM (an estimate; not yet measured on a home PC).

Generation is tested on Linux. Windows with an NVIDIA GPU should work, but is untested. Macs
have no NVIDIA GPU, so they can run only the dry run and the tests.

## Install

macOS / Linux:

```sh
git clone https://github.com/DeniisPyr/Character-Lab.git
cd Character-Lab
python3 -m venv .venv
source .venv/bin/activate
pip install -e './backend[gpu]'     # or './backend' for the dry run and tests only
```

Windows (PowerShell):

```powershell
git clone https://github.com/DeniisPyr/Character-Lab.git
cd Character-Lab
python -m venv .venv
.venv\Scripts\Activate.ps1
# PyPI's torch for Windows has no GPU support; install a CUDA build first.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -e "./backend[gpu]"
```

**Which torch build:**
- On Linux, the default torch from PyPI is built for CUDA 13 and needs NVIDIA driver 580 or
  newer.
- With an older driver (`nvidia-smi` shows a CUDA version below 13), install the CUDA 12.6
  build before the package:
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`.
- If something doesn't match, `charlab generate` stops before downloading anything and
  prints the command that fixes it.

## Usage

```sh
# Print the 16-image plan with every prompt and seed. Loads no models.
charlab generate path/to/reference.png --trigger mychar --dry-run

# Generate the images into results/mychar/
charlab generate path/to/reference.png --trigger mychar
```

Options:

| Option | What it does |
|---|---|
| `--trigger WORD` | required; the word that will call up the character in the trained LoRA. It will start every caption, and it names the output folder. |
| `--out DIR` | output folder (default `results/<trigger>`) |
| `--config FILE` | a YAML file with the settings to change |
| `--description "..."` | a sentence about the character's style or outfit, added to every prompt |

On a **24 GB GPU**, create a file `24gb.yaml`:

```yaml
generation:
  transformer: gguf
```

and add `--config 24gb.yaml`. Every default is in
[backend/configs/default.yaml](backend/configs/default.yaml), with comments. A config file
only needs the values it changes.

**The first real run downloads the models:**
- **Where:** Hugging Face models go to the cache in `HF_HOME` (default
  `~/.cache/huggingface`); the background-removal model goes to `REMBG_HOME` (default
  `~/.u2net`).
- **Downloading in advance:** to fetch the default models before running, use
  `hf download Qwen/Qwen-Image-Edit-2511` and
  `hf download lightx2v/Qwen-Image-Edit-2511-Lightning Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors`.
- **On RunPod:** the PyTorch template sets `HF_HUB_ENABLE_HF_TRANSFER=1` without installing
  `hf_transfer`, which makes every download fail. Run `unset HF_HUB_ENABLE_HF_TRANSFER`
  first.

### The reference image

- **One character**, ideally full body and facing the camera.
- **Background:** transparent or one plain colour works best and needs no extra model. Busy
  backgrounds are removed automatically.
- **Format and size:** PNG, JPEG or WebP; the shorter side at least 512 px; at most 20 MB and
  25 megapixels.

Use characters you created or have the rights to. Don't use photos of real people without
their consent.

## Development

```sh
pip install -e './backend[dev]'
pre-commit install                  # ruff and commit-message checks on every commit
cd backend
python -m pytest
python -m ruff check . && python -m ruff format --check .
```

The tests replace the models with fakes, so they run in seconds without a GPU. CI runs them
on Linux, macOS and Windows.

## Models and licenses

The code is MIT-licensed ([LICENSE](LICENSE)). It downloads these models at run time:

| Model | Used for | License |
|---|---|---|
| [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) | image generation | Apache-2.0 |
| [lightx2v/Qwen-Image-Edit-2511-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning) | 4-step sampling | Apache-2.0 |
| [unsloth/Qwen-Image-Edit-2511-GGUF](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF) | `gguf` mode | Apache-2.0 |
| [BiRefNet](https://github.com/ZhengPeng7/BiRefNet), via rembg (`birefnet-general`) | background removal | MIT |
