<p align="center">
  <img src="frontend/icon.png" width="160" height="160" alt="Character Lab icon">
</p>

<h1 align="center">Character Lab</h1>

<p align="center"><strong>One character image in, a training-ready LoRA dataset out.</strong></p>

<p align="center">
  <a href="https://github.com/DeniisPyr/Character-Lab/actions/workflows/ci.yml"><img src="https://github.com/DeniisPyr/Character-Lab/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/DeniisPyr/Character-Lab/releases/latest"><img src="https://img.shields.io/github/v/release/DeniisPyr/Character-Lab?label=release" alt="Latest release"></a>
  <a href="https://deniispyr.github.io/Character-Lab/"><img src="https://img.shields.io/badge/example-see_a_real_run-2ea44f" alt="Example: see a real run"></a>
</p>

Give Character Lab one image of your character. It makes 16 new images of the same character
in four kinds of shots with [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511),
checks each one, captions it, and writes image/caption pairs ready for a LoRA trainer such as
ai-toolkit or kohya.

<p align="center">
  <a href="https://deniispyr.github.io/Character-Lab/"><img src="examples/preview.jpg" alt="One reference image of Diana, a woman with long dark curly hair in an oversized cream sweater, and the 16 images made from it: four full-body views, four portraits, four expressions and four scenes"></a>
</p>

<p align="center"><sub>A real run with the default settings, not retouched. Every image, with its caption and
scores, is on the <a href="https://deniispyr.github.io/Character-Lab/">showcase page</a> and in
<a href="examples/diana">examples/diana</a>.</sub></p>

## Quick start

You need Python 3.11+ and an NVIDIA GPU with 80 GB of memory, or 24 GB in the quantized mode
(see [Requirements](#requirements)). On macOS or Linux:

```sh
git clone https://github.com/DeniisPyr/Character-Lab.git
cd Character-Lab
python3 -m venv .venv && source .venv/bin/activate
pip install -e './backend[gpu]'

charlab generate reference.png --trigger diana   # the dataset, in results/diana
```

## What you get

16 images in four kinds of shots:

| Shots | Images | Size |
|---|---|---|
| Turnaround | front, three-quarter, side, back | 832×1216 |
| Portrait | front, three-quarter, profile, over the shoulder | 1024×1024 |
| Expressions | joy, surprise, anger, sadness | 1024×1024 |
| Scenes | city walk, hilltop sunset, lantern market, meadow | 1280×720 |

They are written as numbered image/caption pairs:

```
results/character_name/
  001_turnaround_front.png         the image
  001_turnaround_front.txt         its caption
  ...
  016_scenes_meadow_rest.png
  016_scenes_meadow_rest.txt
  manifest.json                    settings, seeds, prompts and captions
  report.json                      quality check scores
results/character_name.zip         with --zip
```

LoRA trainers pair each image with the `.txt` file of the same name. A caption is the trigger
word, the shot, then a description of the image.

> What the captions never describe, the LoRA learns as part of `character_name`; what they do
> describe, such as the outfit, the pose and the background, stays yours to change in a prompt.

## Requirements

- **Python 3.11 or newer**, on Linux, macOS or Windows.
- **An NVIDIA GPU** to generate images. Pick the mode that fits your card:

| | Standard (default) | 24 GB mode (`gguf`) |
|---|---|---|
| GPU memory | 80 GB, such as an A100 or H100 | 24 GB, such as an RTX 3090 or 4090 |
| One dataset takes | 6.5–15 min | 16–18 min |
| First download | about 60 GB | about 36 GB |

**On a 24 GB GPU**, add `--config 24gb.yaml`:

```sh
charlab generate reference.png --trigger diana --config 24gb.yaml
```

**Good to know:**
- **Most of a run is loading the model.** That depends on the machine: from 3.5 to 11.5
  minutes on the same type of GPU.
- **GPU memory, standard mode:** about 40 GB on average over a run, and 61 GB at the peak,
  while generating.
- **GPU memory, 24 GB mode:** about 11 GB on average over a run, and 21 GB at the peak, so it
  fits a 24 GB card.

## Quality checks

Each image is scored right after it is made. Both models are small and run on the CPU, adding
well under a second per image.

| Check | Passes when | Measured on the example's 16 images |
|---|---|---|
| Same character ([CCIP](https://huggingface.co/deepghs/ccip_onnx), whole image vs the reference) | difference below 0.1785 | 0.008–0.062 |
| Face present ([anime face detector](https://huggingface.co/deepghs/anime_face_detection)) | a face is found; the recipe marks shots from behind as having none | found in all 15 front-facing shots |
| Face size | shorter side of at least 64 px | 84–432 px |
| Face sharpness | at least 0.07× the reference face's | 0.15× and up |

**When a check fails:**
- The image is made again with the seed plus 1000, up to 3 tries.
- This happens before the next image, so later shots built on it use the kept version.
- If no try passes, the best one is kept and named in the summary at the end.

**`report.json`** records every try: its seed, scores, face box and failure reasons. It also
lists near-exact copies of another image, found with an image fingerprint.

**Changing the limits:** they are under `quality:` in
[backend/configs/default.yaml](backend/configs/default.yaml).

## Known limits

- **The same-character check is loose for semi-realistic characters.** CCIP was made for
  anime characters, where 12 other characters scored 0.32–0.48, far above the limit. Across
  six semi-realistic characters, 65% of pairs of *different* characters still scored below
  it, and no lower limit worked for all six. The check catches broken images and very
  different characters, but not drift: a grim reaper's skull became a human face in some
  close-ups, another character lost her tentacles in most shots, and every one of those
  images passed. Look through the images before you train. A better check is in progress.
- **The face checks suit anime and illustrated faces.** A realistic style with its own
  detector and identity model is planned.
- **Face repair is experimental and off by default** (see
  [Face repair](#face-repair-experimental)).
- **Generation was measured on A100 GPUs only.** The 24 GB numbers come from an A100 limited
  to 23 GiB, not from a real 24 GB card.

## Scoring a dataset

`charlab eval` scores a finished dataset against its reference, on the CPU, in about 15
seconds for 16 images on a Mac. It reads any folder of image/caption pairs, so it also works
for comparing two runs, or a dataset made with another tool. It writes `eval.md` (tables) and
`eval.json` (every number) into the folder, and prints a summary:

```text
Scored 16 images against reference.png.
Same character: 16 of 16. Faces found: 15 of 16. Captioned: 16 of 16.
Details: results/diana/eval.md (every number is in eval.json)
```

| Score | What it measures | Better |
|---|---|---|
| Identity | CCIP difference between each image and the reference, as in the quality checks | lower |
| Face identity | CCIP difference between each face, cut out, and the reference's face | lower |
| Consistency | CCIP difference between every two images | lower |
| Face size and sharpness | as in the quality checks | higher |
| Diversity | how many of 64 fingerprint bits differ between two images | higher |
| Captions | missing captions, captions without an image, the trigger word, length, duplicates, and captions that still describe the character's fixed look | – |

Each score gives its mean and its worst image or pair, so you know which image to look at
first. The example's scores are in [examples/diana/eval.md](examples/diana/eval.md).

**Read the scores as a comparison**, not as a verdict: only whole-image identity has a limit,
and it is loose for semi-realistic characters (see [Known limits](#known-limits)). The face
score catches damaged faces but not small changes to the features, and the fingerprint sees
the layout and brightness, not the pose or expression.

## Face repair (experimental)

Off by default. Switched on, it redraws a face that fails the checks, with the same Qwen
model:
1. The face is cut out at 2.2× its size and scaled up, so the model sees a close-up.
2. The model redraws the close-up with sharper detail and the same framing, pose,
   expression, hair and lighting. It also sees a close-up of the reference's face.
3. The result is blended back through a soft oval over the face; nothing outside it changes.
4. The repaired image replaces the original only if the layout, the face position, the
   sharpness and the identity score all still hold. `report.json` records every repair, kept
   or rejected.

To switch it on, add to a config passed with `--config`:

```yaml
face_repair:
  mode: failed   # faces that fail every try; `small` also redraws faces under 128 px
```

**Why it is off:**
- **It changes fine facial features, and no check sees it.** In tests it restored blurred
  and JPEG-damaged faces, but without the reference's face it also opened the eyes wider,
  turned brown irises grey-green and thickened the brows, while every score said the face
  was fine. A LoRA trained on such faces learns a slightly different face.
- **It pulls side views toward the front**, so most side-view repairs are rejected.
- **It rarely has anything to do.** With the default settings, 1 of the 64 images of four
  characters failed every try: a backlit face at sunset, scored as too blurry.


## Models and licenses

The code is MIT-licensed ([LICENSE](LICENSE)). The quality checks' image preprocessing
follows DeepGHS's [imgutils](https://github.com/deepghs/imgutils) (MIT). Character Lab
downloads these models at run time:

| Model | Used for | License |
|---|---|---|
| [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) | image generation | Apache-2.0 |
| [lightx2v/Qwen-Image-Edit-2511-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning) | 4-step sampling | Apache-2.0 |
| [unsloth/Qwen-Image-Edit-2511-GGUF](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF) | `gguf` mode | Apache-2.0 |
| [florence-community/Florence-2-large](https://huggingface.co/florence-community/Florence-2-large) | captions | MIT |
| [deepghs/anime_face_detection](https://huggingface.co/deepghs/anime_face_detection) (`face_detect_v1.4_s`) | quality checks: faces | MIT |
| [deepghs/ccip_onnx](https://huggingface.co/deepghs/ccip_onnx) (`ccip-caformer-24-randaug-pruned`) | quality checks: identity | OpenRAIL |
| [BiRefNet](https://github.com/ZhengPeng7/BiRefNet), via rembg (`birefnet-general`) | background removal | MIT |
