# P4 model architecture: every model this phase touched

P4 touched four things that "see" or "think": the **renderer** (not a learned model), the **MediaPipe person mask** (reused from P1), the **MobileNetV3-Small definition network** (trained here), and the **ONNX Runtime Web** engine that runs it. Real files and numbers; where something was not measured or not re-read for this doc, it says so.

---

## 1. The picture first

```
 front photo (any size)
      │  MediaPipe Pose Landmarker (P1): gives a person mask
      ▼
 crop around the person (+5% margin), outside the mask = grey 128, resize
      ▼
 tensor 3 x 256 x 192  (colour x height x width), values 0..1
      ▼
 ┌────────────────────────────────────────────────────────────────┐
 │ ONNX graph (one per output, fp32, 3.73 MB each)                │
 │  normalise (ImageNet mean/spread, inside the graph)            │
 │  MobileNetV3-Small features ─► average pool ─► 576 numbers     │
 │  dropout 0.2 (off at inference)                                │
 │  one linear layer 576 ─► 4 logits                              │
 └────────────────────────────────────────────────────────────────┘
      ▼
 class = 1 + (how many of the 4 logits are above 0)  ─► band low / mid / high
```
Two such graphs run side by side on the same tensor: `definition.fp32.onnx` and `adiposity.fp32.onnx`.

---

## 2. The trunk: MobileNetV3-Small

**Plain picture.** A small, fast image reader built for phones. It first looks for edges, then textures, then shapes, in about a dozen stages, and ends with a list of 576 numbers describing the whole picture.

**What is inside (torchvision `mobilenet_v3_small`, `ml/src/definition/model.py`).**
- **Depthwise-separable convolutions:** instead of one expensive filter that mixes all colours and positions at once, do a cheap per-channel spatial filter and then a cheap 1x1 mix. Far fewer multiplications for similar quality.
- **Inverted residual blocks:** expand to more channels, filter, then squeeze back, with a shortcut that adds the input to the output (so gradients flow).
- **Squeeze-and-Excite:** a tiny gate that learns which channels matter for this image.
- **Hard-swish / ReLU activations:** cheap curved functions (a ReLU is `max(0, x)`).
- **Ends** at **576 channels**; the code names it `FEATURES = 576`. With a 256 x 192 input and the network's overall stride of 32, the last feature map is 8 x 6 x 576 (derived from the stride, not printed in a log). **Global average pooling** averages each of the 576 maps into one number.

**Initialisation:** ImageNet weights (`MobileNet_V3_Small_Weights.DEFAULT`), then everything is fine-tuned in the second stage (probe first with the trunk frozen).

**Size.** Each exported graph is **3.73 MB** fp32. That is about 0.93 million weights at 4 bytes each, a figure **derived from the file size**, not counted in code (the ImageNet classifier is removed, which is why it is much smaller than the full 2.5 M model).

**Why chosen, alternatives rejected.**
| Choice | Reason | Rejected |
|---|---|---|
| MobileNetV3-Small | small and fast in WASM (14.3 ms combined) | ResNet-18 (45 MB, the P3 backbone; too big for two browser graphs) |
| ImageNet pretrained | few features to learn from scratch; the ladder (overfit, probe, fine-tune) | training from zero on 2,808 identities |
| Front view only for the demo outputs | the browser has one frame; the abs cue is on the front | back view (the back head failed the incremental gate anyway) |

---

## 3. The heads: ordinal, four logits each

**Plain picture.** Four tiny judges, each asked "is this body above level k?" for k = 1 to 4.

**Layer-by-layer.**
| Stage | Shape | Note |
|---|---|---|
| input | (B, 3, 256, 192) | B = batch (32 in training, 1 in the browser) |
| trunk features | (B, 576, 8, 6) | derived from stride 32 |
| average pool, flatten | (B, 576) | one number per channel |
| dropout 0.2 | (B, 576) | training only |
| head (per output) | (B, 4) | `nn.Linear(576, 4)` = 2,308 weights incl. bias |
| class | 1 + number of logits above 0 | 1 to 5 |

**2-head training model (`DefinitionNet`).** Two heads, `definition` and `adiposity`, on the shared 576 numbers; output (B, 2, 4); loss = binary cross-entropy on the cumulative targets (`coral_loss`). **Why one shared trunk:** the two outputs read the same picture, and one trunk is cheaper than two.

**8-head training model (`ml/src/definition/model8.py`; ADR-053 Amendment 5; I did not re-read `model8.py` for this document).** Heads: overall, abs, chest, shoulders, arms, back, legs, adiposity, trained on both views with the loss masked by view (see `concepts.md` Step 11). Only the **definition** (overall) head of this model is served.

**Served graphs.** `ml/src/definition/export_fp32.py` wraps each trained model in a `SingleHead` graph: ImageNet normalisation, trunk, one head, one output named `logits`. The browser feeds the input named `x`.

---

## 4. What the training data looks like to the network

- **Input at training:** the rendered figure (transparent PNG) composited on a random solid, gradient or noise background, scaled, colour-jittered, blurred, noised and JPEG-compressed, then resized to 256 x 192 (`data.py`). At evaluation, a fixed grey 128 background.
- **Labels:** class 1 to 5 per head, computed from the sliders (Step 3 of `concepts.md`); quantile bins; frozen edges.
- **Sets:** the *bump* set (with procedural muscle lines, the demo model) and the *flat* set (smooth, for the post-demo A/B). Same identities, same bodies.
- **Splits:** 2,808 train, 396 val, 792 test identities; lighting slice 50.

---

## 5. Training recipe (real values)

| Item | Value |
|---|---|
| optimiser | AdamW, weight decay 0.0001 |
| learning rates | probe: head 0.001; fine-tune: head 0.0003, trunk 0.0001 |
| batch | 32 identities |
| epochs / patience | 30 max, early stop after 6 without improvement |
| checkpoint | best val mean MAE, never test |
| device | Apple M4 Pro, MPS (`PYTORCH_ENABLE_MPS_FALLBACK=1`) |
| speed | about 22 s per epoch (2-head fine-tune), 38 to 45 s (8-head) |
| seed | 0 (fixed); learning-curve points: one seed each |
| logs | every 20 steps (`log_every`) |

**Numbers.** 2-head: overfit PASS (1.000 / 1.000 from step 60); probe best val mean MAE .7399; fine-tune best .3384 at epoch 14. 8-head: probe .5245; fine-tune .3139. Test (scored once): see `phase-overview.md` section 6.

---

## 6. The runtime: ONNX Runtime Web

**Plain picture.** A small interpreter inside the browser that reads the `.onnx` file and does the arithmetic.

- **Version:** `onnxruntime-web` 1.22.0 (`apps/web/package.json`); execution provider **WASM**, effectively one thread because the page is not cross-origin isolated (`crossOriginIsolated` false in the bench).
- **Loading:** on demand (the runtime is not in the main bundle); both graphs load in parallel in 149 ms; absolute `wasmPaths` avoid Vite's `?import` problem.
- **Verified:** WASM against Python ONNX Runtime on 16 fixture images: max difference 1.91e-05 (definition), 3.53e-05 (adiposity), 0 class mismatches.
- **WebGPU:** planned as the first choice in the plan text; not used. The measured WASM time (14.3 ms) already beats the 300 ms limit, so there was no reason to add it.
- **Memory:** frames live in `rgbStore` only until scoring ends.

---

## 7. The person mask (reused, not trained here)

MediaPipe Pose Landmarker (P1, ADR-023) produces the person mask in the browser. P4 only consumes it: `frontTensor(img, prob)` returns `null` if no person is found. **It was not tested on real photos in P4** (the render-path check used the PNG alpha channel as the mask). Segmentation errors on real photos (hair, loose shorts, background clutter) are an untested source of error.

---

## 8. What breaks it (known and suspected)

| Failure | Why | Status |
|---|---|---|
| Real photos | trained on rendered people only | **observed** on two scans of one person; unproven cause |
| Covered torso | no skin cues; scores meaningless | torso check warns (rough guess, thresholds unvalidated) |
| Shorts, clothes, belts | training bodies wear nothing | suspected |
| Different framing or distance | crop/margin differ from training | partly covered by the same crop rule; not stress-tested |
| Muscular and soft | "bulky" absent from the data | stated in ADR-044 Amendment 1 and the model card |
| Dim or very hard light | only two training rigs; the held-out `top` rig: definition .906, adiposity .818 / MAE .600 on n = 50 | measured, thin |
| Women's adiposity from the 8-head model | worse than the 2-head (front-only MAE .328 against .270) | why the hybrid uses the 2-head for adiposity |
| int8 | every variant failed parity | cause unknown, fp32 shipped |
| Phones | latency untested | logged as untested |
| Test-retest | same person flipped adiposity high then low | not investigated |

---

## 9. Interview short answers

1. **Why MobileNetV3-Small and not ResNet-18?** Two browser graphs must stay small and fast: 3.73 MB each and 14.3 ms combined, against a 45.2 MB ResNet-18 for the silhouette model.
2. **Why share a trunk but use separate heads?** Both outputs read the same picture; separate heads keep each answer's thresholds independent.
3. **Where is normalisation applied?** Inside the ONNX graph, so training and inference cannot disagree.
4. **How do you know the exported graph is right?** Parity against torch (4.29e-05 and 7.61e-05 max logit difference) and against Python ONNX Runtime in WASM (0 class mismatches).
5. **What is the weakest part of the pipeline?** The synthetic-to-real gap; the mask on real photos and the torso check are also unvalidated.
