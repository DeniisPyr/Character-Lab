# Character Lab

[![CI](https://github.com/DeniisPyr/Character-Lab/actions/workflows/ci.yml/badge.svg)](https://github.com/DeniisPyr/Character-Lab/actions/workflows/ci.yml)

Turns one character image into a training-ready LoRA dataset.

From a single reference image, Character Lab generates 16 consistent images of the same
character with [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511),
captions each one, and writes them as image/caption pairs ready for a LoRA trainer:

| View | Images | Size |
|---|---|---|
| Turnaround | front, three-quarter, side, back | 832×1216 |
| Portrait | front, three-quarter, profile, over the shoulder | 1024×1024 |
| Expressions | joy, surprise, anger, sadness | 1024×1024 |
| Scenes | city walk, hilltop sunset, lantern market, meadow | 1280×720 |

> **Status: work in progress.** The full pipeline works and is tested on a GPU: preparing the
> reference, generating, checking quality, captioning and exporting the dataset. Face repair
> and evaluation come next.

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
   The Lightning LoRA cuts sampling to 4 steps. Each image is saved as soon as it is made.
3. **Check.** A quality gate scores each image as soon as it is made: is it the same
   character, and is the face there, big enough and sharp? A failed image is made again with
   a new seed (see [Quality gate](#quality-gate)).
4. **Caption.** The generator frees the GPU, then
   [Florence-2](https://huggingface.co/florence-community/Florence-2-large) describes each
   image. The caption is the trigger word, then tags from the recipe, then the cleaned
   description.
   - **The tags** state what Florence-2 often gets wrong: the view, the framing and the
     expression.
   - **The cleanup** removes the character's fixed look (hair, eye colour, skin), guesses
     about the person and filler. That way the look binds to the trigger word, while the
     outfit, pose and background stay described.
5. **Export.** Numbered image/caption pairs, a `manifest.json` and a `report.json` (see
   [Output](#output)).

Every image has its own seed, so a run is reproducible: the same reference, seed and
settings give byte-identical images on the same GPU model and torch CUDA build. Repeat runs on
three A100 machines with CUDA 13 matched exactly. An A100 with the CUDA 12.6 build gave close
variations instead: the same character, pose and composition, with small details such as
background shapes changed. The two builds' math libraries round slightly differently, and
diffusion amplifies that.

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
| First download | about 60 GB | about 36 GB |
| Full run, 16 images with captions | 6.5–15 min: 3.5–11.5 min loading, 2–2.5 min generating and checking, 0.5–1 min captioning | 16–18 min: 3–4 min loading, 10.5–13 min generating, 1 min captioning |

**About these numbers:**
- **Where they come from:** runs on A100 80 GB GPUs, in October 2026. The `gguf` numbers come
  from the same A100s with PyTorch limited to 23 GiB of GPU memory, to act like a 24 GB card.
  Times on a real RTX 4090 will differ.
- **Loading time** depends mostly on the machine: the same A100 type took 3.5 minutes on one
  host and 11.5 on another.
- **Captioning** needs only 2.4 GiB of GPU memory, because the generator frees the GPU first.
- **Quality:** side by side, the `gguf` images look the same as the `bf16` ones.
- **System RAM:** in `gguf` mode the models wait in system RAM between uses, so plan for about
  48 GB of RAM. This is an estimate, not yet measured on a home PC.

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

# Generate and caption the dataset into results/mychar/
charlab generate path/to/reference.png --trigger mychar
```

Options:

| Option | What it does |
|---|---|
| `--trigger WORD` | required; the word that will call up the character in the trained LoRA. It will start every caption, and it names the output folder. |
| `--out DIR` | output folder (default `results/<trigger>`) |
| `--config FILE` | a YAML file with the settings to change |
| `--description "..."` | a sentence about the character's style or outfit, added to every prompt |
| `--zip` | also pack the dataset into `<out>.zip` |

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
  `hf download Qwen/Qwen-Image-Edit-2511`,
  `hf download lightx2v/Qwen-Image-Edit-2511-Lightning Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors`
  and `hf download florence-community/Florence-2-large`. The quality gate's two small models
  (about 200 MB) download on first use.
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

## Output

```
results/mychar/
  001_turnaround_front.png         the image
  001_turnaround_front.txt         its caption
  ...
  016_scenes_meadow_rest.png
  016_scenes_meadow_rest.txt
  manifest.json                    the run's settings and models; seed, prompt and caption per image
  report.json                      the quality gate's scores for every attempt at every image
results/mychar.zip                 with --zip
```

LoRA trainers such as ai-toolkit and kohya pair each image with the `.txt` file of the same
name. A caption, for example `002_turnaround_three_quarter.txt`:

> mychar, full body, three-quarter view, standing, plain white background. The image is a
> full-length portrait of a young woman standing in a relaxed pose. She is wearing a beige
> off-the-shoulder dress with long sleeves and a round neckline. The dress is made of a soft,
> lightweight fabric and has a relaxed fit. She has a neutral expression on her face and is
> looking directly at the camera.

The hair is left out on purpose, so it binds to `mychar`. Each caption's tags are set in the
recipe; `--dry-run` shows how each caption will start.

## Quality gate

Each image is scored right after it is generated. Both models are small and run on the CPU
with ONNX Runtime, adding well under a second per image.

| Check | Passes when | Measured on one character's 16 images |
|---|---|---|
| Same character ([CCIP](https://huggingface.co/deepghs/ccip_onnx), whole image vs the reference) | difference below 0.1785 | 0.008–0.063; 12 other anime characters scored 0.32–0.48 |
| Face present ([anime face detector](https://huggingface.co/deepghs/anime_face_detection)) | a face is found; the recipe marks shots from behind as having none | found in all 15 front-facing shots |
| Face size | shorter side of at least 64 px | 79–423 px |
| Face sharpness | at least 0.07× the reference face's (variance of the Laplacian) | 0.14× and up; a blurred copy scored 0.02× |

**When a check fails:**
- The image is made again with the seed plus 1000, up to 3 tries.
- This happens before the next image, so later shots that use it as Picture 2 get the kept
  version.
- If no try passes, the best one is kept and listed in the summary.

**`report.json`** records every try: its seed, scores, face box and failure reasons. It also
lists near-exact copies, found with a 64-bit image fingerprint. These are reported, not
rejected: distinct shots on a white background can be as close as 5 bits, so a fingerprint
can only catch true copies.

**Changing the limits:** they are under `quality:` in
[backend/configs/default.yaml](backend/configs/default.yaml). The models suit anime and
illustrated characters (`style: anime`, the only style so far). A realistic style with its
own detector and identity model is planned.

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

The code is MIT-licensed ([LICENSE](LICENSE)). The quality gate's image preprocessing follows
DeepGHS's [imgutils](https://github.com/deepghs/imgutils) (MIT). It downloads these models at
run time:

| Model | Used for | License |
|---|---|---|
| [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) | image generation | Apache-2.0 |
| [lightx2v/Qwen-Image-Edit-2511-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning) | 4-step sampling | Apache-2.0 |
| [unsloth/Qwen-Image-Edit-2511-GGUF](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF) | `gguf` mode | Apache-2.0 |
| [florence-community/Florence-2-large](https://huggingface.co/florence-community/Florence-2-large) | captions | MIT |
| [deepghs/anime_face_detection](https://huggingface.co/deepghs/anime_face_detection) (`face_detect_v1.4_s`) | quality gate: faces | MIT |
| [deepghs/ccip_onnx](https://huggingface.co/deepghs/ccip_onnx) (`ccip-caformer-24-randaug-pruned`) | quality gate: identity | OpenRAIL |
| [BiRefNet](https://github.com/ZhengPeng7/BiRefNet), via rembg (`birefnet-general`) | background removal | MIT |
