# Mistakes log
Format: date · phase · what went wrong · root cause · fix · rule to prevent it.
Claude Code: append here whenever a fix fails twice, a wrong assumption is found, or a bug costs > 30 min.

## Pre-seeded traps (avoid these)
- **Subject leakage**: BodyM has repeat photos of the same subject. Split by `subject_id` only.
- **Train/inference skew**: browser MediaPipe masks ≠ BodyM lab silhouettes. Run the same `preprocess.py` (crop, head removal, resize, binarise) on both, and test on BodyM Test-B (in-the-wild) not just Test-A.
- **Scale ambiguity**: pixels aren't cm. Always normalise by silhouette height (and user-entered height for absolute values).
- **Caching personalised output**: never cache full reports (ADR-008).
- **MPS gotchas**: some ops fall back to CPU silently; set `PYTORCH_ENABLE_MPS_FALLBACK=1` and check speed; no float64 on MPS.
- **HF Space cold start**: free Spaces sleep. UI must handle a 30–60 s first request.

## Log
<!-- 2026-11-xx · P1 · ... -->
- 2026-11-01 · P0 · Planned on HF Docker Space + HF Hub as free · assumed free-tier terms from memory; account shows Docker/Gradio are paid · dropped HF hosting, API + models local (ADR-022) · **Rule:** verify a "free" claim on the provider's current page before an ADR depends on it.
- 2026-11-01 · P0 · `pre-commit` failed: `mirrors-prettier` tag `v3.3.3` not found · pinned a hook `rev` from memory; that repo is archived · removed the prettier hook (ruff only) · **Rule:** never write tool versions/tags from memory; run `uv run pre-commit autoupdate` or check the repo's tags.
- 2026-11-01 · P0 · `/health` showed `not_configured` right after keys were added · uvicorn doesn't load `.env`; I'd given a `source` workaround that was easy to skip, and marked P0 DONE before seeing the output · switched to `--env-file ../../.env` · **Rule:** never mark a gate passed without the actual output; give one-command run lines that can't be half-followed.
- 2026-11-01 · P0 · `ruff` flagged imports in tests (I001) · isort didn't know `app` and `physique_schemas` live in subfolders · set `known-first-party` in `pyproject.toml` · **Rule:** when adding a new top-level package outside the root, add it to `known-first-party`.
- 2026-11-01 · P1 · Time-slotted capture (2.5 s per view) gave no "turn now" cue and silently skipped views · I designed for a user who paces to a clock · user-paced capture: big prompt, hold bar, beep + flash, "Capture now" fallback · **Rule:** a guided flow must show what it wants and when it accepted; never skip a step silently.
- 2026-11-01 · P1 · "Body not in frame" shown with the whole body visible, and Capture now blocked by it · the check required visibility >= 0.5 for both sides and the nose; in profile/back the far limbs and face are hidden · judge each left/right pair by its better-visible landmark, name what is missing, let Capture now bypass it · **Rule:** landmark visibility is not "in frame"; do not demand landmarks the pose legitimately occludes.
- 2026-11-01 · P1 · Frontal poses captured as Left/Right while Front/Back timed out (52.9 s) · the frontal reference was an EMA that only accepted larger values, so one bad early frame left it too high and everything looked "turned"; retakes started with no reference · median over a rolling window, kept across retakes, plus an independent depth cross-check · **Rule:** never learn a reference with a one-way update; cross-check a noisy signal with an independent one.
- 2026-11-01 · P1 · "Body touches frame edge" on every view from 3 stray mask pixels, and a side-view check that accepted ~50 deg of turn · thresholds picked as guesses, not from the debug readout · edge run must be >= 3% of the edge length; side views need >= 72 deg and a 1 s hold · **Rule:** show the number behind a failing check in the message so tuning uses data.
