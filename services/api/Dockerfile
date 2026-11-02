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

# The deployed API must never serve baseline A by accident: refuse to start without the model
# (ADR-035). Local runs do not set this and fall back to A with a warning.
ENV REQUIRE_MODEL=true

# Cloud Run injects $PORT (ADR-028)
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
