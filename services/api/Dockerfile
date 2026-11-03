# Build from the repo root, the API imports the schemas and three ml/src files:
#   docker build -f services/api/Dockerfile -t physique-api .
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user services/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user packages/schemas ./packages/schemas
RUN pip install --no-cache-dir ./packages/schemas

COPY --chown=user services/api/app ./app
COPY --chown=user ml/src/preprocess.py ml/src/calibrate.py ./ml/src/
COPY --chown=user ml/src/data/regions.py ./ml/src/data/
# Knowledge base for /plan (ADR-057): index, manifest, baselines, templates, region map, nutrition.
# The default KB_DIR is computed from the repo layout and is wrong in the image, so set both paths.
ENV KB_DIR=/home/user/app/kb EMBEDDER_DIR=/home/user/app/ml/data/models/bge-small
COPY --chown=user kb/*.py kb/*.json kb/chunks.jsonl kb/embeddings.npy ./kb/
# bge-small int8 embedder, baked in (sha256 pinned in kb/manifest.json, checked by tests, ADR-043)
COPY --chown=user ml/data/models/bge-small/model_quantized.onnx ml/data/models/bge-small/tokenizer.json \
     ./ml/data/models/bge-small/
ENV REQUIRE_EMBEDDER=true

# The deployed API must never serve baseline A by accident: refuse to start without the model
# (ADR-035). Local runs do not set this and fall back to A with a warning.
ENV REQUIRE_MODEL=true

# Cloud Run injects $PORT (ADR-028)
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
