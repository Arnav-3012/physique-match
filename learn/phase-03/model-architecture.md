# P3 model architecture

Two models were built and one shipped. Both share the encoder in `ml/src/model.py`.

## Input
`(n, 3, 256, 192)` float in [0, 1]: channel 0 = front mask, channel 1 = side mask, channel 2 = zeros. Masks are the 384x512 canonical silhouettes from `ml/src/preprocess.py` area-downscaled to 192x256 (`downscale`, shared with the API). "Front-only" = channel 1 set to 0 (`dataset.py:blank_side`).

## Encoder: ResNet-18 (ImageNet), classifier removed
Shapes for one input (channels × H × W):

| Stage | Output | Trains at rung |
|---|---|---|
| normalise (ImageNet mean/std buffers) | 3×256×192 | - |
| conv1 7×7/2 + bn1 + ReLU | 64×128×96 | full |
| maxpool 3×3/2 | 64×64×48 | frozen |
| layer1 (2 BasicBlocks) | 64×64×48 | full |
| layer2 | 128×32×24 | full |
| layer3 | 256×16×12 | last2, full |
| layer4 | 512×8×6 | last2, full |
| global average pool | 512 | - |

~11.18 M parameters. Frozen stages are kept in `eval()` (`Encoder.train`, model.py:43) so BatchNorm statistics stay ImageNet's.

## Shipped: `SiameseSubtract` (baseline B, ADR-035)
```
scan -> Encoder -> e (512) -> Linear(512,256) -> ReLU -> Linear(256,8) -> × y_std + y_mean
                                                          = per-person ratios (measurement / height)
gap = ratios(goal) - ratios(current)
```
- Head: 133,384 parameters. Whole model ≈ 11.31 M parameters x 4 bytes = 45.2 MB, which is exactly the ONNX file size (45.2 MB, limit 50).
- Loss: `(1 − p) · huber(full) + p · huber(front-only)` with `p = side_dropout = 0.5`, on standardised ratios (model.py:132).
- Exactly antisymmetric by construction.
- Run twice at inference (goal, current), 15.6 ms p50 each on one thread locally, 31.3 ms for the pair.

## Ablation: `Siamese` (pair head, ADR-004)
```
e_g, e_c -> concat[e_g, e_c, e_g−e_c, e_g⊙e_c] (2048) -> Linear(2048,256) -> ReLU -> Linear(256,8) -> × delta_std
```
Head: 526,600 parameters. Trained on all ordered pairs of a 32-subject batch (992 pairs, only 32 encodes). Val `full`: .2709 vs B .2791 on the selection metric, but ADR-034 rule (CI > 0 on ≥ 6/8 vs B) gave 1/8 -> ADR-006 negative result, left the candidate list.

## Why ResNet-18
Small (45 MB, 16 ms on CPU), pretrained features transfer to edge/shape structure in silhouettes, and the Cloud Run image has no torch: onnxruntime runs it. Larger encoders were not tried; train/val gap (Huber .0416 vs .0707 for the shipped seed) says data, not capacity, is the limit (P8 backlog).

## What breaks it
- **Clothing and pose**: a loose top moved chest/waist by +5-9 cm between two captures of the same person (one pair, anecdotal; noise floor, not bias; front-only val shift ≤ 0.7 cm).
- **Clipped masks** (side view touching the frame): excluded from training (26% of photos).
- **Height or weight are not inputs**: B on Test-B is about level with a ridge on height, weight, sex (ADR-035); the value is not needing them for a goal photo.
- **Back regions**: no BodyM labels; `upper_back` unmeasured, `lats` from an ANSUR II ridge (R² .408 female), `unranked`.
- **Model hash**: any retrain is a new model hash and a fresh Test-B look would need an ADR.
