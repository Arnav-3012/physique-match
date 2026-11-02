# P3 concepts — Siamese CNN, from 0 to 100

Every concept: intuition -> how it works -> the maths -> where it lives in *our* code -> common mistakes -> 3 quiz questions.
All numbers come from `docs/logs/dev-log.md` (P3 entries). Line numbers are as of the P3 close-out.

Reading order matters: 1-3 are the model, 4-6 the training signal, 7-11 the training recipe, 12-13 the export, 14-17 the evaluation.

---

## 1. Siamese network and shared weights
**Intuition.** To compare two people you must measure both with the *same ruler*. If goal and current went through two different networks, a difference in their outputs could just be a difference between the networks.
**How.** One encoder, applied twice. "Shared weights" means literally one set of parameters used for both inputs, so gradients from both sides update the same numbers.
**Maths.** `e_g = f_θ(x_g)`, `e_c = f_θ(x_c)`, same θ.
**Our code.** `ml/src/model.py:27` `Encoder` (ResNet-18 minus classifier). `GapNet.forward` (model.py:75) calls `self.encoder` twice. In training both inputs go through *one* forward pass (`train.py:76`, `torch.cat([x, blank_side(x)])`) so BatchNorm statistics cover everything the model sees at inference.
**What shipped is a degenerate Siamese.** ADR-035: the shipped `SiameseSubtract` (model.py:108) encodes each person alone, predicts their ratios, and subtracts. The pair-head `Siamese` (model.py:79) is kept as the ablation.
**Mistakes.** Thinking "Siamese" requires a pair head. It does not; the shared encoder is the Siamese part.
**Quiz.**
1. Why one forward pass for both inputs in `train_step`? *BatchNorm batch statistics then include blanked-side inputs, matching inference.*
2. What would break if goal and current used separate encoders? *Output differences would mix model differences with person differences; the gap would not be antisymmetric.*
3. Which class ships? *`SiameseSubtract`; the pair head failed 1/8 against it on val.*

## 2. Embeddings
**Intuition.** A 192x256 mask is ~49k numbers, mostly redundant. An embedding is a short vector (here 512) that keeps what matters for the task.
**How.** ResNet-18 convolutions + global average pooling give a 512-vector per image (`fc = nn.Identity()`, model.py:33).
**Maths.** `f_θ: [0,1]^(3×256×192) -> R^512`.
**Our code.** `EMBED = 512` (model.py:17). The input has 3 channels = (front mask, side mask, zeros) so ImageNet weights load unchanged and "side blanked" is simply channel 1 = 0 (`dataset.py:72-80`, `as_input`, `blank_side`).
**Mistakes.** Treating the embedding as interpretable. Nothing forces dimension k to mean "chest"; the head reads it out.
**Quiz.**
1. Why a zero third channel? *Reuses pretrained 3-channel conv1; no architecture change.*
2. Where is "front-only" implemented? *`blank_side` zeroes channel 1.*
3. Size of the embedding? *512.*

## 3. The head `[e_g, e_c, e_g − e_c, e_g ⊙ e_c]`
**Intuition.** Give the head everything a comparison could use: each person, how they differ, and where they agree/multiply.
**How.** Concatenate 4×512 = 2048 features -> Linear(2048, 256) -> ReLU -> Linear(256, 8).
**Maths.** `ŷ = W₂ ReLU(W₁ [e_g; e_c; e_g−e_c; e_g⊙e_c])`, scaled by `delta_std`.
**Our code.** `Siamese.gap` (model.py:83-85).
**Result.** On val it tied baseline B on 7/8 regions and won 1/8 (calf) at `full`: ADR-034 FAIL, ADR-006 negative result. Why: the target is `y_goal − y_current` (ADR-032), which a per-person regressor plus subtraction represents *exactly*; the pair head has no extra structure to learn, only an antisymmetry it has to discover (measured error .0005-.0035, B exactly 0).
**Mistakes.** Assuming a more expressive head must win. It can only win if the target is not already linear in per-person quantities.
**Quiz.**
1. What does `e_g − e_c` give that `e_g, e_c` do not? *An explicit difference feature, easier for the MLP.*
2. Why did it fail to beat subtraction? *Target is a difference of per-person values.*
3. What did we do with the failed model? *Kept as ablation; leaves candidate list.*

## 4. Proportion-gap target (ADR-032)
**Intuition.** "How much bigger would my chest be, relative to my height, with your proportions?" Height cancels the trivial effect of being taller.
**Maths.** `y_p = m_p / h_p` (measurement / height_cm), `Δ(A,B) = y_A − y_B`.
**Our code.** `pairs.py:35` (`y_<region> = column / height_cm`), `true_deltas` (pairs.py:42-44). UI turns it back into cm by multiplying by the user's own height ("at your height").
**Consequence.** The net never sees height at inference, so there is no height shortcut. Baselines A and B use the same definition.
**Mistakes.** Comparing raw cm gaps across people of different heights.
**Quiz.**
1. Units of Δ? *Dimensionless (cm/cm).*
2. Why does `true_deltas` have shape (n, n, regions)? *Every ordered pair (i, j).*
3. How does a user see centimetres? *Δ × their own height.*

## 5. Standardised targets
**Intuition.** Chest ratios (~.59) and forearm ratios (~.15) live on different scales; an unscaled loss would be dominated by whichever region has the biggest numbers.
**Maths.** `ỹ = (y − μ)/σ` using train-split statistics; for pairs, `std(y_i − y_j) = σ√2` for independent subjects.
**Our code.** `GapNet.fit_labels` (model.py:66), `delta_std` (model.py:71-73). Both live as buffers in the module so the ONNX export carries them.
**Mistakes.** Fitting μ, σ on val/test (leakage).
**Quiz.**
1. Why √2? *Variance of a difference of two independent variables = 2σ².*
2. Where are μ and σ stored? *Module buffers `y_mean`, `y_std`.*
3. Fitted on which split? *Train only.*

## 6. Huber loss
**Intuition.** Squared error punishes outliers hard; absolute error ignores how wrong you are near zero. Huber is quadratic near 0, linear far away.
**Maths.** `L(e) = ½e²` if `|e| < 1` else `|e| − ½` (δ = 1, on standardised errors).
**Our code.** `F.huber_loss` (model.py:105, 130); the same formula re-implemented in numpy for val reporting (`train.py:80-83`) so train and val are in one unit.
**Mistake we made.** Early probe log compared train Huber (.18) with val standardised MAE (.50); not comparable. For normal errors Huber .18 ≈ MAE .48. Fixed by printing both in Huber (dev-log, Step 2).
**Quiz.**
1. Why not plain MSE? *Mislabelled/odd photos (clothes, pose) would dominate.*
2. Units of the epoch line? *Standardised Huber.*
3. What is `floor_huber`? *Huber of predicting delta = 0.*

## 7. Antisymmetry
**Intuition.** If A has a bigger chest than B by +3, then B has a smaller chest than A by −3: `f(A,B) = −f(B,A)`.
**How.** Pair head: learned, not enforced (all-pairs batches contain both orders, so "swap and negate" augmentation needs no code). Subtract model: exact by construction, `value(g) − value(c)`.
**Measured.** Pair head |f(i,j)+f(j,i)| = .0005-.0035 (9-17% of MAE); B exactly 0.
**Our code.** `SiameseSubtract.gap` (model.py:118).
**Quiz.**
1. Why is B exactly antisymmetric? *Subtraction is.*
2. Why no swap augmentation code? *Both (i,j) and (j,i) are in the batch.*
3. Is antisymmetry a quality? *It is a consistency check, not accuracy.*

## 8. In-batch all-pairs
**Intuition.** Encoding is the expensive part; comparing is cheap. Encode K subjects once, then compare all of them.
**Maths.** K = 32 subjects -> 32·31 = 992 ordered pairs from 32 encodes.
**Our code.** `Siamese.loss` (model.py:87-105): `keep = 1 − I`, `nonzero` gives (goal, current) index pairs; `train.py:64` `subject_batches` samples subjects first, then a random photo each (avoids subjects with 27 photos dominating).
**The correction (user).** The effective data is ~1,391 train subjects, *not* pairs. 992 pairs per batch add no independent information. Never quote pair counts as the sample size.
**Quiz.**
1. Why exclude (i, i)? *Target is always 0, evaluation never scores them.*
2. How many pairs from K subjects? *K(K−1).*
3. Why sample subject first? *Some subjects have up to 27 photos.*

## 9. Transfer-learning ladder
**Intuition.** Start from a network that already sees edges and shapes (ImageNet) and unfreeze progressively; stop at the first rung that works.
**Rungs.** `probe` (head only) -> `last2` (layer3+4) -> `full` (all).
**Our code.** `RUNGS` (model.py:20-24), `Encoder.freeze` (model.py:38), separate learning rates (head .001, encoder .0001; `train.py:45-46`, AdamW + cosine).
**Our numbers (val_mae, standardised).** Siamese probe .4974 -> last2 .2916 -> full .2709. Baseline B last2 .2927 -> full .2791. Floor (predict 0) .7961. Training time 4-5 s/epoch (probe), 6 s (last2), 12 s (full) on the M4 Pro (MPS).
**Mistakes.** Treating the probe as a gating rung; it is a sanity check only.
**Quiz.**
1. What does the probe tell you? *Whether features are usable at all and the pipeline is sane.*
2. Why a lower LR for the encoder? *Preserve pretrained features.*
3. Which rung shipped? *`full`.*

## 10. BatchNorm when frozen
**Intuition.** BatchNorm keeps running mean/variance. A "frozen" layer whose BatchNorm still updates its statistics is not frozen.
**Our code.** `Encoder.train` override (model.py:43-47): frozen modules stay in `eval()` so ImageNet statistics are kept. Also the BatchNorm batch-size check: K ≥ 16 (≥ 32 encodes per batch) in every full-rung config.
**Mistake to avoid.** `requires_grad_(False)` alone does not stop BN statistics updating.
**Quiz.**
1. What does `requires_grad_(False)` leave running? *BN running stats update in train mode.*
2. Why K ≥ 16? *BN needs enough samples for stable batch statistics.*
3. Where is it enforced? *`Encoder.train`.*

## 11. Overfit one batch
**Intuition.** If a model cannot memorise 16 examples, the pipeline (labels, loss, shapes, gradients) is broken. Do this before any full run.
**Our numbers.** Siamese (16 subjects, full rung, no augmentation, no side dropout): loss .0093 @ step 20, .0001 @ 100, .000033 @ 300. Baseline B: .0023 @ 20, ~0 final. PASS.
**Our code.** `Config.overfit` (train.py:53), one `fixed` batch (train.py:146).
**Quiz.**
1. Why drop augmentation for this test? *You want memorisation to be possible.*
2. What does a loss stuck at the floor mean? *Bug, not capacity.*
3. Is it a quality measure? *No, a plumbing check.*

## 12. Augmentation (mask-appropriate)
**Intuition.** Real browser masks differ from BodyM masks (slight rotation, framing). Augment *without changing the quantity being predicted*.
**Our code.** `augment.py:11-13`: rotate ±3°, zoom ±3% **isotropic only** (stretching one axis would change the proportions we regress), shift ±2%. `jitter` applies an independent affine per view. Side-view dropout (p = .5) in the loss (model.py:132): ADR-025, so a front-only goal photo is a valid input.
**Deferred, only if val needs it:** edge noise, morphological jitter, clothing bulge, neck cut-row jitter.
**Quiz.**
1. Why no anisotropic zoom? *Changes width/height ratio = the label.*
2. Why dropout only on the goal side? *Current always has both views.*
3. Cost of front-only? *At most .0017 MAE (chest) on Test-B.*

## 13. ONNX and int8
**Intuition.** ONNX is a framework-neutral file of the computation graph, so the API runs it with `onnxruntime` and no torch (image stays small, ADR-028).
**Our numbers.** Single-person ONNX (opset 17, legacy exporter): 45.2 MB (limit 50). Parity vs torch max |Δ| 5.96e-07 over 32 val subjects (tolerance 2e-4). Latency, 1 thread, M4 Pro: one encode p50 15.6 / p95 16.0 ms; two encodes p50 31.3 / p95 31.6 ms (limits 75 / 150). **int8 was not needed**: it is the fallback on a miss (ADR-035). Re-confirm on 1 vCPU Cloud Run at P7.
**Our code.** `export_onnx.py:40-62` (normalisation and label scale are *inside* the graph), `session`, `latency_ms`.
**Mistakes.** Measuring latency with multiple threads or without warm-up (`WARMUP = 10`, `RUNS = 100`).
**Quiz.**
1. Why export one person, not a pair? *Run twice and subtract; model is the same for goal/current.*
2. What would int8 cost? *Accuracy risk, so only if size/latency fail.*
3. Why parity check? *Proves export did not change the function.*

## 14. MAE
**Maths.** `MAE = mean |ŷ − y|` over all ordered pairs i ≠ j (diagonal NaN) and per region.
**Our code.** `abs_errors`, `region_mae` (evaluate.py:45-54). In training the selection metric is the mean of full and front-only MAE divided by `delta_std`.
**Reading Test-B MAE.** Shoulders .0083 vs A .0107; waist .0308 vs .0397 (full input). These are proportion units; ×172 cm gives centimetres at that height.
**Quiz.**
1. Why report MAE and not RMSE? *Robust, directly interpretable.*
2. Why mask the diagonal? *A subject vs itself is always 0.*
3. Why two slices? *Full input and front-only goal.*

## 15. Subject-level splits and leakage
**Intuition.** BodyM has several photos per subject. A random split puts the same body in train and test; the model memorises people.
**Our numbers.** train 1,715 / val 303 / testA 87 / testB 399 subjects; one subject shared by both tests dropped. Usable (unclipped side mask): 1,391 train, 251 val, 351 Test-B subjects. Pair universe on Test-B: 351·350 = 122,850.
**Our code.** `data/split.py`, test "no subject in two splits", `canonical_subjects` (pairs.py:27-35) picks one seeded photo per subject so full and front-only slices use the *same* photo.
**Mistake we made.** `extract.py` keyed testA/testB by photo, assigning 7 photos wrongly; fixed by keying (split, subject) (ADR-033 housekeeping, mistakes.md).
**Quiz.**
1. Why a canonical photo per subject at eval? *No pair sampling, fully reproducible list.*
2. What is leakage here? *Same subject in two splits.*
3. Why drop the shared subject? *It belonged to both test sets.*

## 16. Bootstrap CIs
**Intuition.** One MAE difference could be luck. Resample *subjects* (the independent unit) many times to see how much the difference moves.
**Maths.** Multinomial counts `m_i` (Σ = n); pair (i, j) weight `m_i·m_j`; 1,000 resamples; 95% CI = 2.5-97.5 percentiles of `MAE_base − MAE_model`.
**Our code.** `compare` and `weighted_mae` (evaluate.py:82-110): an einsum over the pair-error tensor, so 1,000 resamples cost almost nothing. `underpowered` flag when the CI half-width > 10% of the baseline MAE.
**Mistakes.** Resampling pairs (they share subjects, so the CI would be far too narrow).
**Quiz.**
1. Why resample subjects, not pairs? *Pairs are not independent.*
2. What is the weight of pair (i, j)? *m_i × m_j.*
3. What does CI lower bound > 0 mean? *Model beats baseline in that region at 95%.*

## 17. The gate rule (ADR-033) and Test-B discipline
**Rule.** Per baseline, `lo > 0` on ≥ 6/8 regions and `hi < 0` on none. Fixed pair list with its hash committed before scoring; Test-B scored once per final candidate; a second look needs an ADR.
**Our code.** `verdict` (evaluate.py:113-117), `claim_test_b` ledger (evaluate.py:120-129, `evals/testb_ledger.json`) refuses a model hash it has seen. `evaluate.py` aborts if the Test-B list differs from `evals/testb_pairs.json` (hash 9abfd0c74c1f).
**Our result.** B full seed 0 (hash 213383d7c86f): 8/8 vs A on both slices, none worse; thinnest lower bounds shoulders .0016 and waist .0044 (front-only). Honest limits: B's MAE rose 31-55% val -> Test-B, and a null ridge that knows weight ties B on several regions (B's value: it needs no height/weight for the goal).
**Mistakes.** Tuning against Test-B, re-running until it passes.
**Quiz.**
1. Why a ledger? *Makes a second look impossible without an ADR.*
2. Why 6/8 and not 8/8? *Forearm barely beats zero; the rule is fixed up front.*
3. What does the ADR-006 fallback say? *If it does not beat B, document a negative result and ship the better model.*

---

## Extras from P3 close-out (not in the original list, but you will be asked)
- **Calibration / reliability** (`ml/src/calibrate.py`): per region and slice, `reliability = clip(1 − sqrt(s_goal² + s_full²)/σ_prior, 0, 1)`. Full / front-only: shoulders .50/.49, chest .73/.71, waist .75/.74, hips .76/.75, upper_arm .68/.67, forearm .61/.61, thigh .69/.68, calf .61/.61. Optimistic: calibrated on val, Test-B ran 31-55% worse.
- **Unique-share test** (V-taper, ADR-038): `1 − R²(index gap ~ waist gap)` = .148 female, .171 male, below the fixed .2 -> the V-taper gap stays display-only.
- **Reachable ceiling** (ADR-040/041/042): waist at a body-fat floor -> three zones; female ceiling lies outside ANSUR's range, so zone wording carries a "cannot verify" caveat.
- **Lats** (ADR-036/037): ANSUR II ridge, noisy-input R² .408 (female) < .5 -> `unranked`, shown as "back width (estimate)".

---

# Glossary
- **Ablation** — a deliberately weaker variant kept to measure what a component adds (our pair head).
- **Antisymmetry** — `f(A,B) = −f(B,A)`.
- **AdamW** — Adam with decoupled weight decay.
- **Baseline A** — the geometric (no-ML) analyzer.
- **Baseline B** — one per-person regressor, run on both people and subtracted; the shipped model.
- **BatchNorm** — normalises activations with batch statistics in training, running statistics at inference.
- **BodyM** — public silhouette + measurements dataset (CC BY-NC).
- **Bootstrap** — resample to estimate uncertainty of a statistic.
- **Canonical photo** — the one seeded photo per subject used for evaluation.
- **CI** — confidence interval; here 95%, percentile bootstrap.
- **Clipped mask** — silhouette touching the frame edge; excluded (`side_clipped`).
- **Delta (Δ)** — `y_goal − y_current`, the gap.
- **Embedding** — fixed-length learned vector for an input.
- **Encoder** — ResNet-18 without classifier.
- **Floor** — reference score of a trivial predictor (predict 0).
- **Front-only slice** — goal side view blanked.
- **Gate** — pre-registered pass/fail rule.
- **Huber loss** — quadratic near zero, linear far.
- **Leakage** — test information in training.
- **Ledger** — record of Test-B looks per model hash.
- **MAE** — mean absolute error.
- **MPS** — Apple GPU backend of PyTorch.
- **Null ridge** — ridge on height, weight, sex; a reference, not a baseline.
- **ONNX / onnxruntime** — portable model format / its CPU runtime.
- **Opset** — ONNX operator-set version (17).
- **Parity** — ONNX output equals torch output within a tolerance.
- **Probe (linear probe)** — frozen encoder, trained head; a sanity check.
- **Reliability** — per-region confidence from calibrated val residual std.
- **Rung** — one level of the transfer-learning ladder.
- **Shared weights** — the same parameters applied to both inputs.
- **Side dropout** — blanking the goal's side view in training (ADR-025).
- **Standardised target** — `(y − μ)/σ` with train statistics.
- **Subject-level split** — all photos of a subject in one split.
- **Test-B** — in-the-wild held-out subjects; looked at once per final candidate.
- **Underpowered** — CI too wide to resolve a 10% relative gain.
