# Dev log
One entry per session. ≤ 10 lines. Numbers over adjectives.

```
## YYYY-MM-DD · Phase NN
Done:
Metrics:      (e.g. MAE per region, RAG recall@5, cache hit rate, p50 latency)
Blockers:
Next:
```

## 2026-11-01 · Phase 00 (session 1)
Done: scaffold written by Claude (files only, no commands run): pyproject (uv workspace), schemas + tests, FastAPI `/health`, Dockerfile, Vite/React/TS/Tailwind hello-world, CI + HF-sync workflows, .env.example, CLAUDE.md working-agreement rules.
Metrics: none yet, nothing executed.
Blockers: user must run install/test commands (see chat); accounts/keys requested phase by phase.
Decision: HF Spaces/Hub dropped (not free) -> ADR-022; API + models local, web on Vercel. Removed HF token, deploy-space workflow, HF Space README frontmatter.
Also: ruff isort `known-first-party` fix; prettier pre-commit hook removed (bad tag). 3 entries added to mistakes.md.
Gate (P0): pytest 14 passed, 1 warning in 0.13s (PASS) · web live at https://physique-match.vercel.app/ (PASS) · `/health` both providers ok: user-reported, output not seen. First `/health` run showed `not_configured`; fixed with `--env-file ../../.env`. Repo: Arnav-3012/physique-match.
Closed 2026-11-01 on the user's confirmation.
Process: `/sensei` learn docs skipped for P0 (setup phase); only P3-P6 get them. Datasets downloaded only in their own phase.
Next: P1 (private 4-view capture). Awaiting user go-ahead.

## 2026-11-01 · Phase 01
Done: browser capture in `apps/web/src/capture/` (Pose Landmarker mask, user-paced 4-view turn, upload path, consent + 18+, review "what we send"), `lib/scan.ts`, vitest suites (yaw, sharpness, quality, silhouette, scan). ADR-023; 4 entries in mistakes.md.
Metrics: `npm run build` green (63 modules, 407 kB JS / 129 kB gzip). Laptop 4/4 valid, turn 7.6 s on the passing run (earlier runs 19.6 s, 5.9 s with failures, 52.9 s with the reference bug). Masks 2-4 KB each (384x512).
Gate (P1): captures ~8 of 10 (laptop + phone, user-reported, 4 phone captures valid), **short of 10; accepted by the user 2026-11-01**. Network-tab check **PASS** (user-reported 2026-11-01): every request was localhost, nothing to another domain. Per-capture times not recorded for all runs.
Blockers: none. Untested: busy backgrounds, iOS Safari GPU path, loose-clothing heuristic thresholds.
Process: `/sensei` skipped for P1 per protocol; user asked for explain docs, written to `learn/phase-01/` (model-architecture.md, phase-overview.md).
Next: P2 (BodyM pipeline + geometric baseline + body comp). BodyM download happens in P2 only.

## 2026-11-01 · Phase 02 · Deploy spike (ADR-028)
Done (docs/files only, nothing executed): ADR-028 (Cloud Run supersedes ADR-022; ADR-007/008/009/022 status lines updated); CLAUDE.md, architecture §6, phase-plan (spike block in P2, P3/P5/P7), README, privacy log note, mistakes traps. Written: `spikes/cloudrun/` (app, Dockerfile, asset generator, bench), `infra/kill_switch/` (simulation on by default); `services/api/Dockerfile` now honours `$PORT`.
Checked on provider pages: Mumbai (asia-south1) is Tier 1, so 180k vCPU-s / 360k GiB-s hold; Free Trial closes instead of charging; UPI not mentioned anywhere found.
Spike results (fill as measured; **nothing below is measured yet**):
- Trial: signed up TBD · ends TBD (signup + 90 d) · **day-85 upgrade decision due TBD (signup + 85 d)**
- Project: `physique-gap-3012` (the only project), billing linked: `gcloud billing projects describe` shows `billingEnabled: true` (user-pasted 2026-11-01). gcloud logged in, default project set. Billing account ID kept out of the repo.
- Billing account currency / payment method that worked: TBD (user to confirm INR + card/UPI)
- Region (console): TBD
- Budget alert created: TBD (amount, scope, credits counted?)
- Kill switch, simulation: TBD (log line "Billing disabled. (Simulated)", billing still linked)
- Image size (as stored) baked / fetched / with scipy: TBD
- Cold start (baked vs fetched): TBD · warm p50/p95: TBD (n >= 30)
- `X-Forwarded-For` shape: TBD
- Usage + billing report: TBD ($0 expected)
- Python/onnxruntime/numpy/Pillow versions from `/healthz`: TBD
Blockers: user does account + billing setup; Claude gives one command at a time.
Next: user checklist -> budget -> kill switch (simulation) -> deploy private with max-instances -> measure. DEFERRED by the user (ADR-031): resume at deploy time (P7); P3 is not blocked. Done so far: APIs billingbudgets/pubsub/cloudbilling enabled, topic `budget-alerts` created; no budget, no function, nothing deployed.

## 2026-11-01 · Phase 02 · BodyM pipeline, geometric baseline, body comp (spike NOT part of this entry)
Done: BodyM (87 MiB, 2,505 subjects) -> subject-level split (train 1,715 / val 303 / testA 87 / testB 399; one subject shared by both tests dropped from testB) -> canonical 384x512 frame (`ml/src/preprocess.py`) -> silhouette-only row finder + scale-free features -> `GeometricAnalyzer` (gap) -> waist model -> `bodycomp.py` -> `/analyze` with mask validation, 413 cap, in-memory rate limit. ADR-029, ADR-030; 8 entries in mistakes.md.
Gate 2 (changed after the first result, ADR-029): silhouette-only rho >= 0.6 on 7/8 regions, testA and testB. testA rho: shoulders .83, chest .85, waist .91, hips .83, upper arm .87, thigh .78, calf .79, forearm .30. testB: .80 .83 .91 .80 .88 .70 .73 .28. Original gate (silhouette-only vs weight-aware null): 1/8 on testA. Null rho .80-.93 on every region. Silhouette adds beyond null: 7/8 (testA), 5/8 (testB; chest, hips, thigh no).
Gate 3 waist MAE, photo level (silhouette / null): val 3.05 / 4.32 · testA 2.70 / 3.00 · testB 3.39 / 3.50 cm; p90 5.65 (testA), 7.28 (testB); residual sd 4.05 cm. PASS (testA <= 5 and below null).
Gate 4 RFM/Mifflin/floors: 11 tests, all combinations of sex x activity x goal x weight x height x waist. Gate 5 `/analyze`: RGB, grey-texture, wrong-size, JPEG, extra field -> 422; 413 before parsing; 429 + Retry-After; gap identical when only weight changes; logs carry deltas, no masks. Suite 69 passed before the route tests; route tests green, user-reported ("success"; count not seen).
Gate 6 parity, 3 captures (12 masks): head-cut |z| <= 1.9 after setting NECK_CUT_OFFSET 0.02 -> 0.0 (phone captures cut highest); only 2/12 masks have a stat with |z| >= 2 (area, loose tee); front/back width-profile r .96-.99; `arms_separated` 6/6; `crotch_found` 2/6 (shorts hide the crotch, so confidence drops). **Deviation:** 3 of the planned 5 captures (no laptop near, no phone far).
Not done: left/right asymmetry, chest/back split (no ground truth); `upper_back`/`lats` unvalidated (confidence 0.3). Known: `extract.py` assigns 7 testB photos of the shared subject to testA by subject id (subject-level numbers unaffected).
Blockers: P2 not closed. Deploy-spike gate (entry above) is open and blocks P3. For the spike: the API imports `ml/src/preprocess.py` and `scipy.ndimage`; the image must carry the first and the spike should measure the second (ADR-028).
Next: finish the spike; then P2 close-out (tag `phase-02`), then P3. Suggested commit message for this work is in the chat.
