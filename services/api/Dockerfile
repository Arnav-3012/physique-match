FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user app ./app

# Cloud Run injects $PORT (ADR-028)
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
