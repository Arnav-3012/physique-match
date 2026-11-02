FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv

COPY requirements.txt .
RUN pip install -r requirements.txt

# Measures what ml/src/preprocess.py's scipy.ndimage import costs in image size (ADR-028).
ARG WITH_SCIPY=0
RUN if [ "$WITH_SCIPY" = "1" ]; then pip install scipy; fi

# Baked variant: model/dummy.onnx present at build time. Fetch variant: leave model/ empty and set
# MODEL_URL on the service; app.py then downloads at container start.
COPY model/ /models/
COPY app.py .

RUN useradd -m -u 1000 user
USER user

# Cloud Run injects $PORT
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}
