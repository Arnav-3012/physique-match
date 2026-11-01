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
