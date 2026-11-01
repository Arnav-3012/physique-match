# PhysiqueGap API

FastAPI backend. Accepts only 1-bit silhouettes, keypoints and numbers (never photos).
Runs locally (ADR-022): `cd services/api && uv run uvicorn app.main:app --reload --env-file ../../.env`
(uvicorn does not read `.env` on its own; without `--env-file`, `/health` reports `not_configured`).
The `Dockerfile` is kept for whichever free host we pick at P7.
