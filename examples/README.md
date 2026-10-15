# Example dataset

`diana/` is a dataset as `charlab generate` wrote it, from `reference.png` with the default
settings:

```sh
charlab generate reference.png --trigger diana
```

- **16 image/caption pairs**, ready for a LoRA trainer.
- **`manifest.json`**: the settings, and each image's seed, prompt and caption.
- **`report.json`**: the quality gate's scores. All 16 images passed on the first try.
- **`eval.md` and `eval.json`**: the scores from `charlab eval diana --reference reference.png`,
  with charlab 0.3.0.

`preview.jpg`, the overview at the top of the main README, shows the reference and the 16
images side by side.

**How it was made:** on an A100 80 GB with the CUDA 12.6 build of PyTorch, in October 2026, by
charlab 0.2.0.dev0, whose settings are the same as 1.0.0's defaults. The same seed on the
CUDA 13 build gives close variations, not identical images. The manifest names the reference
`Stuff_00020_.png`, its original file name.
