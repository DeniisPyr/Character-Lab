# Evaluation of diana

16 images against the reference `reference.png`, scored by charlab 0.3.0.

| Score | Better | Mean | Worst | Worst image or pair |
|---|---|---|---|---|
| Identity: CCIP difference from the reference | lower | 0.026 | 0.062 | 004_turnaround_back.png |
| Face identity: CCIP difference from the reference face | lower | 0.025 | 0.077 | 007_portrait_profile.png |
| Consistency: CCIP difference between two images | lower | 0.043 | 0.114 | 004_turnaround_back.png and 011_expressions_anger.png |
| Face size: shorter side in px | higher | 243 | 84 | 003_turnaround_side.png |
| Face sharpness: × the reference face's | higher | 1.13 | 0.15 | 002_turnaround_three_quarter.png |
| Diversity: fingerprint bits that differ, of 64 | higher | 27.2 | 5 | 002_turnaround_three_quarter.png and 003_turnaround_side.png |

CCIP sees the same character below a difference of 0.1785.

- **Same character:** 16 of 16 images
- **Faces found:** 15 of 16 (none in 004_turnaround_back.png)
- **Sizes:** 4 × 832×1216, 8 × 1024×1024, 4 × 1280×720
- **Captions:** 16 of 16 images
  - 16 start with `diana`
  - 36–91 words, 64.2 on average
  - 16 distinct
  - 0 describe the character's fixed look or have filler

## Images

| Image | Size | Identity | Face identity | Face px | Sharpness | Caption words |
|---|---|---|---|---|---|---|
| 001_turnaround_front.png | 832×1216 | 0.008 | 0.007 | 107 | 0.20 | 82 |
| 002_turnaround_three_quarter.png | 832×1216 | 0.010 | 0.022 | 106 | 0.15 | 67 |
| 003_turnaround_side.png | 832×1216 | 0.027 | 0.035 | 84 | 0.18 | 60 |
| 004_turnaround_back.png | 832×1216 | 0.062 | – | – | – | 55 |
| 005_portrait_front.png | 1024×1024 | 0.013 | 0.003 | 257 | 2.04 | 51 |
| 006_portrait_three_quarter.png | 1024×1024 | 0.015 | 0.012 | 283 | 0.63 | 65 |
| 007_portrait_profile.png | 1024×1024 | 0.041 | 0.077 | 273 | 0.45 | 73 |
| 008_portrait_over_shoulder.png | 1024×1024 | 0.015 | 0.024 | 269 | 0.18 | 59 |
| 009_expressions_joy.png | 1024×1024 | 0.020 | 0.019 | 432 | 3.28 | 56 |
| 010_expressions_surprise.png | 1024×1024 | 0.027 | 0.006 | 373 | 3.33 | 53 |
| 011_expressions_anger.png | 1024×1024 | 0.061 | 0.037 | 418 | 2.71 | 42 |
| 012_expressions_sadness.png | 1024×1024 | 0.025 | 0.023 | 394 | 2.54 | 36 |
| 013_scenes_city_walk.png | 1280×720 | 0.026 | 0.009 | 128 | 0.51 | 81 |
| 014_scenes_hilltop_sunset.png | 1280×720 | 0.022 | 0.020 | 184 | 0.16 | 67 |
| 015_scenes_lantern_market.png | 1280×720 | 0.025 | 0.054 | 164 | 0.19 | 91 |
| 016_scenes_meadow_rest.png | 1280×720 | 0.025 | 0.022 | 178 | 0.42 | 89 |
