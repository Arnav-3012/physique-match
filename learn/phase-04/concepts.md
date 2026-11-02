# P4 concepts, step by step

Each step follows the same order: **plain picture, how it works, the maths (every symbol explained), our real files and numbers, data source and licence, what went wrong, three quiz questions.** Numbers come from `docs/logs/dev-log.md`. Where a number was not measured, it says so. Files are under `/Users/arnav/physique-gap/`.

---

## Step 1. Synthetic data and the domain gap

**Plain picture.** You want to teach someone to spot ripe fruit, but you have no real fruit, so you hand them thousands of very good plastic fruit and tell them which are "ripe" by the colour you painted. They get perfect at plastic fruit. Real fruit has bruises, odd light and shapes they never saw. How well they do on real fruit is the **domain gap**: the difference between the world we trained on and the world we use it in.

**How it works here.** We cannot train on real shirtless photos: nothing is labelled, and the project forbids scraped photos (ADR-018). Demo models train on synthetic data plus BodyM only (ADR-020), and BodyM is silhouettes, which cannot show definition. So we render people.

**The risk, written before training (ADR-049 item 6).** The label is a *function of the sliders we used to build the body*. So a high score on held-out fake bodies shows the network can invert our formula. It does not show that it reads real bodies.

**Numbers.** Held-out fake bodies: definition rho .909 to .923. Real photos: two scans looked wrong (Step 17). That difference is the lesson.

**Data source and licence.** Nothing is scraped. Rendered with MPFB (Step 2). BodyM is not used by this model.

**What went wrong.** Exactly this: the model looked great on fake data and wrong on the owner's real scans.

**Quiz.**
1. *Why does a high score on fake held-out bodies not prove the model works on real people?* Because the labels come from our own formula, so the network can learn the formula without learning what real definition looks like.
2. *What is a domain gap?* The difference between the training world and the real world.
3. *Which part of the project rules pushed us to synthetic data?* ADR-018 (no scraped photos) and ADR-020 (synthetic + BodyM only).

---

## Step 2. Rendering fake people: Blender, MPFB, Cycles, and the spike

**Plain picture.** A game character creator: sliders for sex, height, muscle, weight; press a button, get a 3D body. A camera photographs it from the front and the back. A "spike" is a small, throwaway test to find out whether the idea is possible before you commit.

**How it works.**
- **MPFB 2.0.18** (a Blender add-on, built on MakeHuman) builds the body from sliders ("macro" sliders: gender, age, muscle, weight, height) plus "targets" that move parts of the mesh.
- **Blender 5.2.3** renders with **Cycles**, a ray-tracing renderer, on the Apple GPU (METAL). The figure is on a transparent background.
- **Light rigs:** `key_left` and `key_right` (small lights) are training rigs; `top` is reserved for a held-out-lighting test slice and nothing trains on it (ADR-045, pre-registered).

**Spike numbers (320 to 384 px, 24 to 32 samples):** Cycles on METAL confirmed; `view_s` (seconds per view) about 0.41 s for Cycles and 0.10 s for EEVEE; `setup_s` (building the body) about 0.09 s for the default body only, which is not a sizing number. Full render later: `setup_s` 0.157 to 0.173 s, `view_s` 0.42 to 0.43 s. MPFB has per-region muscle targets (pectoral, dorsi, shoulder, upper and lower arm, upper and lower leg) and only a `stomach-tone` target for the abdomen. **The mesh is smooth, so definition lines do not exist in the images** (this triggered Step 5).

**Cost model (ADR-047 Amendment 1).** Render time per identity = `setup_s + 4 x view_s` (two views x two sets, bump and flat). At about 1.9 s per identity, 3,996 identities is about 2.1 h.

**Data source and licence.** MPFB's CC0 base mesh, targets and skins; the code is GPL; the project claims no rights over renders per its FAQ; **we did not read the licence files ourselves** (model card). The clothing pack (MPFB system assets) was not found, so bodies are bare with a plain skin colour (stated domain gap).

**What went wrong.** My first MPFB call failed (`default_set=False`), and my first slider probe guessed call names that did not exist; both were fixed from the spike's own report.

**Quiz.**
1. *Why do a spike before writing `render.py`?* To measure the real cost and what the sliders really do, so the plan is sized from numbers and not guesses.
2. *What does `view_s` measure and why does it matter?* Seconds to render one view; it sets the budget for how many identities fit in the time cap.
3. *Why is one light rig never used for training?* So a held-out-lighting test can show how the model copes with light it never saw.

---

## Step 3. Latents and labels: how "definition" and "adiposity" are defined

**Plain picture.** We cannot ask a render "how defined are you?". So we invent a hidden dial for each fake person: how visible their muscle should be. Then we write a rule: *a defined muscle is hidden by fat, and cannot be more visible than the muscle underneath allows.* The dial after that rule becomes the label.

**How it works (ADR-049, `synth/latents.py`).**
- `d` is a random definition dial.
- `hide(fat)` shrinks it as fat grows (fat hides definition).
- `cap(muscle)` is a ceiling from the muscle slider position.
- `d_eff = min(d x hide(fat), cap(muscle))`. This is the definition latent. About 2.0% of identities end at exactly 0 (the floor).
- **Adiposity** (`adip`, ADR-053) is the waist girth that the weight slider adds *at fixed muscle*, read from a lookup fitted on a 5 x 5 muscle x weight probe grid (`synth/probe.py`, `synth/fit.py`). It is a **waist-girth proxy**, not a body-fat percentage. Fitted `a_ref` 28.84 cm (female) and 37.05 cm (male).

**The measured findings that changed the plan.**
- Weight adds waist mainly at low muscle (female +28.8 cm at muscle 0 against +5.2 at 0.75; male +37.1 against +7.8). So adiposity sorts bodies by muscle softness, a high-muscle body keeps full definition, and **"bulky" (muscular and soft) cannot be produced** (ADR-053 Amendment 2, ADR-044 Amendment 1). A real muscular person with extra fat will likely get high definition and low adiposity.
- The macro muscle slider adds soft mass at its low end (male waist depth 24.95 cm at muscle 0 against 18.71 at 0.5), so a low-muscle body can look softer than its adiposity label says.
- The cap cannot be calibrated to the measured muscle effect because that effect is not monotone, so it keeps a position form with placeholder constants (stated as unmeasured).
- **Clamps frozen by the owner:** muscle 0 to 0.75, weight 0.25 to 0.75, both sexes (slider ends give distorted bodies). The owner would not accept invented clamps, so the fit was rebuilt from measured effects first.

**Hashes (so the labels cannot drift).** Fit `d03e7a14...8cb2`; plan `f839cba9...5416`; edges `55be96b2...7c83`; labels `556adaac...c9f`.

**Quiz.**
1. *Why does fat appear inside the definition label?* Because fat hides muscle lines in real life; a soft but muscular body should score lower definition.
2. *Why is adiposity called a "proxy"?* It is the waist girth the weight slider adds, not a measured body-fat percentage.
3. *What does "bulky cannot be produced" mean for a real user?* The data has no muscular-and-soft bodies, so such a person gets an unreliable pair of scores.

---

## Step 4. Classes by quantiles and the coverage check

**Plain picture.** Line up all 2,808 training people by the dial and cut the line into five equal groups. Group 1 is the lowest fifth, group 5 the highest fifth. "Class 5" therefore means "in the top fifth of *our fake crowd*", not "this much muscle".

**How it works.** The cutpoints are the 20%, 40%, 60% and 80% points of the training values, computed **once** and frozen (`synth/plan/edges.json`, ADR-045). Frozen edges: definition about 0.147 / 0.300 / 0.441 / 0.614; adiposity 0.061 / 0.134 / 0.218 / 0.400. Because 2.0% of `d_eff` is exactly 0 and sits on the first edge, a **tie rule** sends a value equal to an edge to the lower class (`latents.label`; test added).

**Before rendering we checked coverage (`synth/coverage.py`)**: each class 20.0% of train per head; the smallest sex x muscle x fat cell is 11.1% of a sex (rule: at least 7%); the N_MIN prefix (612) holds out 123 identities, 6 to 8 per cell. After rendering (`synth/post_render.py`) the same checks passed on the real files.

**The UI bands** (ADR-053 Amendment 4): low = classes 1 and 2, mid = class 3, high = classes 4 and 5.

**Quiz.**
1. *If every person in the world were very lean, what would "class 3" mean?* Still the middle fifth of the training crowd; classes are relative to the sampled mix.
2. *Why compute the edges once and freeze them?* So labels cannot be adjusted after seeing results.
3. *What does the tie rule fix?* A pile of identical zeros on the first edge would otherwise be split arbitrarily.

---

## Step 5. Making muscle lines visible: bump maps

**Plain picture.** A smooth plastic torso has nothing to see. A **bump map** is a trick: a greyscale picture that tells the renderer "pretend the surface is slightly raised here and dented there". Light then makes grooves appear, like pressing lines into clay without changing its shape.

**How it works (ADR-049, `synth/cues.py`, `synth/bump.py`).** Numpy draws height maps (abs grid, chest, shoulder, back, leg and arm lines) from approximate body landmarks. They are sampled in 3D object space and fed to Blender's Bump node. The strength is `d_eff x` a log-normal random factor, **so it is not the label**. Two key numbers: **Bump Distance 0.008** (frozen; 0.004 was clearly weaker in a strip test) and **`SIDE_BLEND` 0.01** (was 0.06, which smeared front and back maps on thin limbs; the owner waived the one-round rule once for this bug).

**Result (read by me, by eye, not a metric).** Abs: clear for both sexes at high `d` and low fat; absent at high fat; flat at low `d`. Back: spine and shoulder-blade lines readable but short diagonal slashes. Legs: vertical lines, seam-like. Arms: triceps lines visible in the back view only. **Chest and shoulders: not readable.** Cost: bump `view_s` 0.428 against flat 0.418 (+2%). The ADR-049 fallback ("most regions fail") did not fire.

**Resolution check** (`synth/downsample_check.py`, criterion fixed before looking): at 256 and 224 px, 3 to 4 abs grooves and the central line are countable on both sexes at high `d`, lean (female weaker at 224). It was judged by eye, with no metric, one body per sex.

**The honest limit.** The cues are *procedural*, not anatomy. **Abs is the only clearly learnable cue**, which is why the "overall definition" head effectively reads the abdomen and why we make no regional claim.

**Quiz.**
1. *Why must the bump strength not equal the label?* If it did, the network could read the strength directly; adding randomness keeps the label from being a direct image property.
2. *Why only one fix round?* A pre-registered cap against endless tuning.
3. *Why does the model mostly "see the abs"?* It is the only cue that rendered clearly.

---

## Step 6. Splits by identity, the lighting slice, pre-registration

**Plain picture.** If your exam questions are the same as your practice questions, a high mark means nothing. An **identity** is one fake person with front and back images; if the same person appears in train and test, the model has seen the answer. So we split by person, using a fixed fingerprint (hash) of the id, and decide the split **before** any image exists.

**How it works (ADR-045, `synth/plan.py`).** Block-hash split 7 / 1 / 2 per cell: train 2,808, val 396, test 792. The 50 held-out **lighting-slice** identities are rendered again under the unused rig `top`. Everything is **pre-registered**: thresholds, class edges, hashes and the rig roles are written down before any look at results.

**The test ledger.** `evaluate.py` refuses to score test a second time unless told (`--again`). The 2-head model was scored once; the 8-head model twice with the same checkpoint to add the conditions the owner set afterwards (the first numbers reproduced exactly).

**Mild selection (stated).** Choosing the hybrid used test numbers, so the reported hybrid test figures are slightly optimistic (model card).

**Quiz.**
1. *Why split by identity, not by image?* The front and back image of one person are near-copies; splitting by image leaks the answer.
2. *What does pre-registering protect against?* Moving the goalposts after seeing results.
3. *What is the lighting slice for?* A test under a light the model never trained on.

---

## Step 7. Making training images less perfect (domain-gap mitigation)

**Plain picture.** If every training photo is a clean cut-out on nothing, the model is lost when a real background appears. So we paste the figure on different backgrounds and make the picture slightly worse in the ways real photos are.

**How it works (`ml/src/definition/data.py`).** Random **solid, gradient or noise** backgrounds, then scale, colour jitter, blur, noise and JPEG compression. At evaluation, the background is a fixed grey (128) so scores are repeatable. The browser does the same: person mask, then grey 128 outside. (The plan said real CC0 background photos; the built version uses synthetic backgrounds.)

**Quiz.**
1. *Why a fixed grey at evaluation?* So scores do not move with a random background.
2. *Why must the browser use the same grey?* Same preprocessing in training and inference (project rule).
3. *Does augmentation close the domain gap?* No; it can only reduce sensitivity to the effects we imitate.

---

## Step 8. The transfer-learning ladder: overfit, probe, fine-tune

**Plain picture.** You hire a photographer who already knows what edges, textures and shapes look like (the ImageNet-trained network). First you check your whole setup can learn at all (overfit one batch). Then you only train the last small part (probe). Only then do you let the whole thing adapt (fine-tune).

**How it works (`ml/src/definition/train.py`; configs `ml/configs/definition_{overfit,probe,finetune}.yaml`).**
- **Overfit one batch:** 32 identities, no augmentation, 200 steps. It must reach accuracy 0.99 on both heads (`OVERFIT_ACCURACY`).
- **Probe:** trunk frozen, only the head learns, `lr_head` 0.001.
- **Fine-tune:** everything learns, starting from the probe's best file; `lr_head` 0.0003, `lr_backbone` 0.0001 (smaller, to avoid wrecking good pretrained knowledge), AdamW, weight decay 0.0001, batch 32, patience 6, 30 epochs max.
- **Early stopping and checkpoint choice:** the best epoch is picked on **val** mean MAE only; the test split is never read in `train.py`.

**The maths (AdamW update).** `w <- w - lr x (m / (sqrt(v) + eps) + lambda x w)`, where `w` is a weight, `m` and `v` are running averages of the gradient and of its square, `lr` is the step size, `eps` stops division by zero, and `lambda` (weight decay 0.0001) shrinks weights a little each step. If `lr` is huge, training jumps around; if tiny, it barely moves.

**Real numbers (val).** Overfit: exact 1.000 / 1.000 from step 60. Probe: stopped at epoch 12, best mean MAE .7399, **adiposity rho .484** (frozen ImageNet features did not carry the waist-width cue). Fine-tune: 21 epochs, best at 14, mean MAE .3384, adiposity rho **.895**, definition rho .909. About 22 s per epoch on MPS.

**What went wrong: BatchNorm.** The first overfit run **failed** (loss .0024 but exact accuracy 0.438 / 0.344). Cause: *BatchNorm* keeps running averages of mean and spread for use at inference (momentum 0.01), and after only 200 steps those averages still lag. The accuracy was measured in eval mode, so it used the lagging averages. Fix: measure in batch-statistics mode with dropout off. The loss curve and the model code did not change (`mistakes.md`).

**Quiz.**
1. *What does the overfit check prove, and what does it not?* That the pipeline (labels, loss, optimiser) can learn; not that the model generalises.
2. *Why use a smaller learning rate for the pretrained trunk?* To adapt it without erasing what ImageNet already taught it.
3. *Why was adiposity weak in the probe but strong after fine-tuning?* Frozen ImageNet features did not carry the body-width cue; unfreezing let the trunk learn it.

---

## Step 9. Ordinal classification: CORAL heads

**Plain picture.** Shirt sizes S, M, L, XL are ordered. Saying "XL" for someone who is "S" is a worse mistake than saying "M". A plain 5-way classifier treats all mistakes alike. Instead, ask four ordered yes/no questions: "above class 1? above 2? above 3? above 4?" The class is 1 plus the number of yeses.

**How it works (`ml/src/definition/model.py`).** Each head is a linear layer from 576 features to **4 logits**; logit `k` means P(class > k+1). Training uses binary cross-entropy on the cumulative targets.

**The maths.** For a true class `y` (0 to 4), target `t_k = 1` if `y > k`, else 0. Loss = mean over `k` of `-[t_k log(sigma(z_k)) + (1 - t_k) log(1 - sigma(z_k))]`, where `z_k` is logit `k` and `sigma(z) = 1 / (1 + e^-z)` squashes to 0 to 1. Prediction: class = `sum_k [z_k > 0]` (0 to 4, shown as 1 to 5). If all logits are very negative the class is 1; if all very positive, 5. Real-scan example: definition logits 4.15, 3.03, 1.07, -0.72, three positive, so class 4 (high); adiposity 1.11, -0.26, -2.39, -5.02, one positive, so class 2 (low).

**Alternatives rejected.** Plain softmax (ignores order); a single regression number (needs arbitrary cutpoints, and the gate wants classes).

**Quiz.**
1. *Why does CORAL respect order?* The cumulative yes/no targets make a 5 for a 1 violate more questions than a 5 for a 4.
2. *Logits 2.0, 0.5, -1.0, -3.0: which class?* Two are above 0, so class 3.
3. *Why four logits for five classes?* Four thresholds separate five ordered classes.

---

## Step 10. Metrics and the gate

**Plain picture.** Three questions: did it get the order right (rho), how far off is it on average in classes (MAE), and how often exactly right (exact)?

**The maths.**
- **Spearman rho:** correlation of the *ranks* of predictions and truth. 1 means perfect order, 0 none. (Without ties: `1 - 6 x sum(d^2) / (n (n^2 - 1))`, with `d` the rank difference; with ties, Pearson correlation on ranks, as here.)
- **MAE:** mean of `|predicted class - true class|`. MAE 0.3 means about a third of a class off on average.
- **Exact:** share of identities where the class matches.

**Gate (ADR-047 item 10, unchanged, never loosened).** Per head on the synthetic test set: rho >= 0.8 and MAE <= 0.6; front-only and held-out-lighting slices are reported alongside; per-group results only if the group has at least 30 samples. 2-head test: definition .909 / .359 / exact .660; adiposity .901 / .337 / exact .681; by sex, male definition .915 / .336, female .904 / .381. **Pass.**

**Thin margin worth knowing.** On the lighting slice (n 50) the 2-head adiposity MAE is **.600**, exactly the limit. That is a thin margin, not a clear pass.

**The incremental gate.** A regional head earns a regional display only if its rho is at least 0.05 above the overall score on its own label. The best was legs, +0.019, then arms +0.015; chest and shoulders +0.000. **None passed.** Likely cause (inferred, not tested): all six regional labels come from one shared definition latent, so they carry almost the same information.

**Quiz.**
1. *Rho .9 and MAE .35: what do they say together?* Order is mostly right and typical error is a third of a class.
2. *Why is a thin lighting-slice margin worth reporting?* n is 50 and the value equals the limit, so it could easily fail on another draw.
3. *Why did regional heads fail the incremental gate?* Their labels are nearly redundant with the overall one.

---

## Step 11. Multi-view, masked loss, and the 8-head model

**Plain picture.** One student, eight questions. Some questions can only be answered from the front photo (abs, chest, shoulders), one only from the back (back), others from both (arms, legs). If a question cannot be answered from the photo, we leave it out of that photo's marking.

**How it works (`ml/src/definition/{views,items8,data8,model8,train8,evaluate8}.py`, ADR-053 Amendment 5).** Heads: overall, abs, chest, shoulders, arms, back, legs, adiposity. The loss is **masked by view**: front-only heads are scored on front images, the back head on back images, and arms, legs and adiposity on both. The `n` in the logs shows it: 792 test identities give 792 front-only samples or 1,584 two-view samples. Because it trains on both views it sees **twice the images** (5,616 against 2,804 at 100%).

**Results.** Test, overall definition rho .923 MAE .302 (against the 2-head .909 / .359); primary gate passes on all 8 heads. A/B against the 2-head (tolerance 0.01): pass, so the 8-head model is not worse. **Caveat:** the definition gain cannot be attributed to the regional heads, because the extra data from two views is a confound.

**Quiz.**
1. *What does "masked by view" mean?* A head is only scored on the views that can answer its question.
2. *Why is the 8-head gain confounded?* It also trained on twice the images.
3. *What does the A/B tolerance of 0.01 mean?* The 8-head model may be at most 0.01 worse to count as "not worse".

---

## Step 12. The hybrid decision (and its honesty cost)

**Plain picture.** Two students: one is better at the first question, the other at the second. You let each answer the question they are better at. The risk: you chose after seeing the exam results, so the combined score is a little flattering.

**What happened (ADR-053 Amendments 5 and 6).** The 8-head model passed every condition for definition. For adiposity the second scoring on the same front-only samples showed 8-head MAE .354 against the 2-head .337 (over the .01 tolerance), and the lighting slice .779 / .660 against .818 / .600. So: **definition from the 8-head model, adiposity from the 2-head model.** By sex, the 8-head adiposity is worse for women (front-only MAE .328 against .270) and slightly better for men (.379 against .404).

**Stated limits.** Chosen after seeing test results (mild selection); the 8-head model misses the "not worse" rule by .007 on MAE for adiposity (margins thin); no region passed the incremental gate; the size cap was raised after sizes were measured. All are in the model card.

**Quiz.**
1. *Why is the hybrid's test score slightly optimistic?* The model per head was picked after seeing test results.
2. *Which head came from which model?* Definition from the 8-head; adiposity from the 2-head.
3. *Why not ship one model?* Neither model passed every condition for both outputs.

---

## Step 13. The learning curve (should we render more?)

**Plain picture.** Train on a quarter of the data, then half, then all. If the score keeps climbing, more data helps; if it flattens, it does not.

**Result (val, one seed per point, ADR-047 item 9).** 2-head mean MAE .5922 (705 identities), .3144 (1,402), .3384 (2,808): flat from half. 8-head .4632, .3658, .3139: still improving, but with twice the images from two views. Both 50% fine-tunes hit the 30-epoch cap with the score still moving, so they may be under-trained. The rule sets no threshold, so the call is the owner's: render no more for the demo (a second full render is about 2.25 h and the real problem is the domain gap, not data volume).

**Quiz.**
1. *What does a flat curve say?* More of the same data will not help.
2. *Why is one seed per point a weakness?* Epoch-to-epoch noise is about 0.02, comparable to the differences.
3. *Why would more fake data not fix the real-photo failure?* The gap is between fake and real, not a shortage of fake.

---

## Step 14. Export: ONNX, parity, and why int8 failed

**Plain picture.** A trained PyTorch model lives in Python. The browser cannot run it. **ONNX** is a common travelling format; **ONNX Runtime Web** runs it in the browser (WASM = a fast low-level code format). **Quantisation** shrinks numbers from 32-bit decimals to 8-bit whole numbers to make the file smaller; **parity** checks that the travelled model gives the same answers.

**How it works (`ml/src/definition/export_fp32.py`).** Two single-output graphs. Normalisation (ImageNet mean and spread) is **inside** the graph, so the browser only gives numbers in 0 to 1. Parity: fp32 ONNX against torch on 396 val identities: max logit difference 4.29e-05 (definition) and 7.61e-05 (adiposity), both under the 2e-4 limit. ONNX Runtime Web 1.22.0, WASM, one thread, Node, 16 fixture images: max difference 1.91e-05 and 3.53e-05; class mismatches 0.

**Why int8 failed (pre-registered in ADR-053 Amendment 6).** Static QDQ int8, calibration on 128 training identities, per-channel; requirement: agreement >= 0.95, max class step <= 1. Result: agreement .374 (definition) and .227 (adiposity), max step 4, val rho down to .562 and .253. Three time-boxed variants (percentile, entropy excluding 4 warned convs, percentile excluding depthwise, first and last) all failed (agreement .15 to .30). The quantiser warned that four Conv biases exceeded the int32 range. **Cause not established**; a mistake in my quantisation setup is not ruled out. int8 sizes were 1.27 to 1.57 MB.

**Decision (owner).** Combined cap raised from 5 MB to 8 MB; ship fp32: 3.73 MB x 2 = **7.46 MB**. The int8 diagnosis is TBD after demo.

**Data source and licence of the weights.** ImageNet-pretrained MobileNetV3-Small from torchvision (licence terms not re-checked in this phase); the rest is trained by us on synthetic data.

**Quiz.**
1. *What does parity prove?* The exported graph computes the same function as the trained model.
2. *Why is normalisation inside the graph?* So training and inference cannot drift apart through a forgotten step.
3. *Why was keeping some layers fp32 not enough?* The pattern (lower agreement with more layers kept) does not fit "a few fragile layers", so the cause lies elsewhere or in my setup.

---

## Step 15. Running in the browser: preprocessing and latency

**Plain picture.** Before the model sees you, the app cuts you out, centres you, shrinks you to a fixed size, and paints everything else grey, exactly like the training pictures.

**How it works (`apps/web/src/definition/preprocess.ts`, `score.ts`, `models.ts`).**
1. MediaPipe gives a person mask.
2. `boundingBox` finds the person; `cropFit` crops with a **5% margin** (same as the outlines, so framing matches training).
3. Pixels outside the mask become grey 128.
4. Resize to **192 x 256** with supersampling (average n x n sub-samples, up to 4 x 4, like the area-aware resize the training images went through), values scaled to [0, 1], channel-first (CHW).
5. Two ONNX sessions run; class = 1 + (number of logits above 0); bands low 1 to 2, mid 3, high 4 to 5.

**Latency (laptop, Chrome 156, WASM, effectively one thread; `apps/web/bench.html`).** Sessions load in parallel in 149 ms; first run 32 ms; over 100 runs: preprocessing p50 3.6 ms, definition p50 5.4 ms, adiposity p50 5.3 ms; **combined p50 14.3 ms, p95 14.9 ms** against the 300 ms limit. Caveats: the test frame is synthetic at 540 x 720; the model files came from the local dev server; a phone is **untested**. p50 is the middle run; p95 is the run that 95% of runs beat.

**Browser-path check on renders (val only, fixed before running).** Definition rho .926 / MAE .278 (Python .928 / .265); adiposity .875 / .409 (Python .895 / .346); margins rho >= reference - 0.03 and MAE <= reference + 0.10. Pass. It rules out crop, mask, resize, scaling and output-mapping bugs **on renders**, not the segmenter or a real photo.

**What went wrong.** Vite appended `?import` to a root-relative dynamic import and refused a file in `public/`; fixed with an absolute `wasmPaths`. My own test spread 147,456 floats into `Math.min` and overflowed the stack.

**Quiz.**
1. *Why must the browser crop use the same margin as the training crop?* Otherwise the person fills a different part of the frame than the model expects.
2. *What does p95 14.9 ms tell you that p50 does not?* Almost the slowest runs are still fast, so there are no bad outliers.
3. *What does the render-path check not cover?* The segmenter and real-photo framing.

---

## Step 16. Privacy by design and the scope gate

**Plain picture.** The photo is used like a mirror: the app looks at it, writes down two words, and throws the photo away.

**How it works (ADR-013, ADR-051, ADR-052).**
- **Opt-in, off by default;** dev-only for now (owner). Full-body scope only; ages 18 and over only (hidden otherwise, rejected server-side). A clothed scan gets no score, with a stated reason.
- **`capture/rgbStore.ts`** keeps the front frame in memory only and clears every frame after scoring. A test guards that no image data goes into any browser store. (Before this change the claim "frames are released" was false; scratch and preview canvases kept the last frame. It was fixed and tested.)
- Only the two bands may leave the device (optional field `definition` on `/analyze`, schema `DefinitionScores {definition, adiposity, experimental}`, no regional fields). The server rejects non-masks (ADR-003).
- **Torso check (`capture/torso.ts`).** A user scanned with clothes under the bare-torso option and nothing warned. Now the torso's colour (r/g chromaticity) is compared with the person's own face and index fingers: within 0.04 and share >= 0.5 reads as bare. If not, a "Your torso looks covered" alert offers retake, change option, or "my torso is bare, score it". The thresholds are **unvalidated starting guesses**.
- **Manual DevTools check:** the owner reported all browser storage showing 0 after a scan (the network half was not reported, so it is not recorded as done).

**Quiz.**
1. *What leaves the device?* Two bands (plus masks and keypoints for the size part), never a photo.
2. *Why a torso check from skin colour?* Scoring a covered torso would be meaningless; colour is a cheap rough guess and can be wrong, so the user can override.
3. *Why is the opt-in dev-only?* The first real scans looked wrong, so users do not see the score yet.

---

## Step 17. The real-photo failure and how we tested for a bug

**Plain picture.** The model said "high muscle, high softness" for someone who is neither. Before blaming the training, rule out a plumbing mistake in the app.

**How we tested (`apps/web/scripts/browser_path.diag.ts`, thresholds fixed before running).** Feed the 396 val renders through the app's own preprocessing and the served ONNX files, and compare with the Python numbers. Pass: rho >= Python - 0.03 and MAE <= Python + 0.10. It passed. Then a dev preview (`DevPreview.tsx`) showed the model's own input for a real scan: clean, centred, full height, torso skin share 1.00.

**Reading.** Not a browser bug. Leading explanation (not proven): the synthetic-to-real gap. Other suspects: **test-retest instability** (the same person got adiposity high, then low; whether clothing or pose differed is not recorded), knee-length patterned shorts, a training set with no clothing, and a pair (high definition with high adiposity) that is rare in the training data because fat hides definition. n = 1 person; this is an anecdote, not a test.

**What would settle it.** Rank 6 or more consented adults yourself *before* the model sees them and compare with the model's order (Spearman); repeat one person 10 minutes apart; train the flat model (about 15 to 20 min) and compare.

**Quiz.**
1. *Why check the render path before blaming the training?* A plumbing error would also give wrong answers on real photos, and it is cheap to rule out.
2. *Why is n = 1 weak evidence?* One person cannot tell a systematic failure from noise.
3. *What does a test-retest flip tell you?* The output is unstable for the same body, which is a problem regardless of accuracy.

---

## Step 18. The dataset search and the ethics line

**Plain picture.** If the fake data is the problem, find real photos with labels. We looked.

**Findings (Level 2 research, licences read from the primary pages for SHAPY, Unsplash, Pexels, Pixabay and the Hugging Face card; the rest from papers and search summaries).** See the table in `phase-overview.md` section 9. In short: no open set has definition labels; DXA-paired sets exist on request (Shape Up! Adults, ShapedNet) but 2D photo sharing is not confirmed; SHAPY has 10 validation people (HBW) and agency photo URLs under a non-commercial research licence; stock sites do not address ML training or model releases; the Hugging Face sample is 21 people under CC BY-NC-ND.

**The ethics line.** We did not scrape bare-torso photos from the web or Reddit, even with an "above 18" search term: age cannot be verified from a search, the people shown did not consent to this use, and platform terms bar it. The owner can lift ADR-018 for the project but cannot give consent for third parties. Keeping the photos private does not repair missing consent or age.

**Next, in order:** ordering check on consented adults, flat model, data-request emails (Shape Up!, ShapedNet), SHAPY HBW validation as a waist-proxy check, P8 volunteers with a consent form.

**Quiz.**
1. *Why does "private" not make scraping safe?* Consent, age and terms are unaffected by who sees the files.
2. *Which public data could validate adiposity but not definition?* DXA or waist-measured sets (labels are body fat or girth, not definition).
3. *What kind of label does definition need?* Human ratings of the photos, with rater agreement reported.

---

# Glossary

- **Adiposity (here):** a waist-girth proxy score, not body-fat percent.
- **AdamW:** optimiser that adapts step sizes per weight and applies weight decay.
- **A/B:** comparing two models on the same data with a tolerance.
- **Augmentation:** deliberately damaging training images (blur, noise, JPEG, colour) to teach robustness.
- **Band:** low (classes 1 to 2), mid (3), high (4 to 5) shown to users.
- **BatchNorm:** layer that normalises activations using batch statistics in training and running averages at inference.
- **Bump map / Bump Distance:** greyscale map faking surface relief; the distance sets how strong (0.008).
- **CORAL:** ordinal classification as four cumulative yes/no questions.
- **Cycles / METAL:** Blender's ray-tracing renderer / Apple GPU backend.
- **d, d_eff, hide, cap:** the definition dial, the dial after fat hiding and the muscle ceiling.
- **Domain gap:** difference between training data and real use.
- **DXA:** dual-energy X-ray scan measuring body composition.
- **Early stopping:** stop when the validation score stops improving.
- **Fine-tune / probe:** train everything / train only the head.
- **fp32 / int8:** 32-bit decimals / 8-bit integers.
- **Identity:** one rendered fake person (front and back images).
- **Incremental gate:** a regional head must beat the overall rho by 0.05.
- **Latent:** a hidden number the labels are computed from.
- **Lighting slice:** 50 identities rendered under the unused light `top`.
- **MAE:** mean absolute error in classes.
- **MPFB:** MakeHuman add-on for Blender that builds bodies from sliders.
- **Ordinal:** categories with a natural order.
- **ONNX / ONNX Runtime Web:** portable model format / its browser runtime.
- **Parity:** exported model equals the trained one within a tolerance.
- **p50 / p95:** median / 95th-percentile run time.
- **Pre-registration:** writing thresholds and rules down before looking at results.
- **Proxy:** a measurable stand-in for what you really want.
- **Quantile bin:** equal-share cut of a sorted list.
- **QDQ quantisation:** int8 with quantise and dequantise nodes; needs calibration data.
- **Rho (Spearman):** rank correlation, 1 = perfect order.
- **rgbStore:** in-memory holder of camera frames, cleared after scoring.
- **Selection bias:** a result that looks better because you chose it after seeing it.
- **Test-retest:** same subject measured twice should give the same answer.
- **Transfer learning:** starting from a model trained on other data (ImageNet).
- **WASM:** WebAssembly, fast code the browser runs.

---

# Ten viva questions, with short answers

1. **Why synthetic data, and what is the main risk?** No labelled real data and no-scraping rules (ADR-018, 020). Risk: the label is a function of our sliders, so high synthetic scores prove the model inverts our formula, not that it reads real bodies (ADR-049 item 6). It showed on a real scan.
2. **Define the definition label.** `d_eff = min(d x hide(fat), cap(muscle))`, cut into five classes at frozen quantiles of the training set (edges about 0.147 / 0.300 / 0.441 / 0.614). Classes are relative to the fake crowd.
3. **Why ordinal CORAL heads?** Order matters; four cumulative yes/no questions, trained with binary cross-entropy, class = 1 + number of positive logits. Plain softmax ignores order.
4. **Why the overfit-one-batch check, and how did it fail once?** It proves the pipeline can learn. First run failed (0.438) because eval-mode BatchNorm used lagging running averages; measuring in batch-statistics mode gave 1.000 from step 60.
5. **Why split by identity?** Front and back images of one person are near-copies; image-level splits leak. Block-hash split 7/1/2 gave 2,808 / 396 / 792.
6. **Report the gate numbers and the thin one.** 2-head test: definition .909 / .359, adiposity .901 / .337 (limits rho >= 0.8, MAE <= 0.6). Thin: lighting-slice adiposity MAE .600 on n = 50, exactly the limit.
7. **Why a hybrid and what is the caveat?** 8-head model for definition (rho .923 / MAE .302), 2-head for adiposity (the 8-head missed "not worse": .354 against .337). Chosen after seeing test results, so mildly optimistic; no region passed the incremental gate (best +0.019 against 0.05).
8. **Why fp32, and what do you not know about int8?** Every int8 variant failed parity (agreement .15 to .37 against .95). The cause is unknown; my setup is not ruled out. Owner raised the cap to 8 MB; pair is 7.46 MB.
9. **Is the browser path correct, and how do you know?** Render path through the app's preprocessing: definition .926 / .278 against Python .928 / .265; WASM parity 0 class mismatches; latency p50 14.3 ms, p95 14.9 ms. This covers renders, not the segmenter or real photos.
10. **The model looked wrong on a real photo. What do you do?** Ruled out a browser bug; leading cause is the synthetic-to-real gap (unproven). Ship only as an experimental, hidden opt-in; do the ordering check on consented adults, flat model, and get real labelled data via requests or consented volunteers (P8). Never scrape bare-torso photos.
