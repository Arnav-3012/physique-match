"""Cloud Run deploy spike (ADR-028): can the production image shape serve ONNX + PNG on 1 vCPU?

Not product code. Measures image size, cold start, warm latency and how the client IP arrives.
"""

import io
import os
import time
import urllib.request
from importlib.metadata import version
from pathlib import Path

_BOOT = time.perf_counter()

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from PIL import Image  # noqa: E402

MAX_BYTES = 512 * 1024  # same cap as the real /analyze (architecture.md section 5)
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/models/dummy.onnx"))
MODEL_URL = os.environ.get("MODEL_URL")  # set to fetch at start instead of baking
FETCHED_PATH = Path("/tmp/dummy.onnx")  # Cloud Run /tmp is in-memory: counts toward the 2 GiB


def load_session() -> tuple[ort.InferenceSession, str, float]:
    t0 = time.perf_counter()
    source, path = "baked", MODEL_PATH
    if not path.exists():
        if not MODEL_URL:
            raise RuntimeError(f"{path} missing and MODEL_URL not set")
        urllib.request.urlretrieve(MODEL_URL, FETCHED_PATH)
        source, path = "fetched", FETCHED_PATH
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session, source, time.perf_counter() - t0


session, MODEL_SOURCE, MODEL_LOAD_S = load_session()
INPUT = session.get_inputs()[0]
IN_DIM = INPUT.shape[1]
STARTUP_S = time.perf_counter() - _BOOT

app = FastAPI(title="physique-gap deploy spike")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in os.environ.get("WEB_ORIGINS", "").split(",") if o],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "model_source": MODEL_SOURCE,
        "model_load_s": round(MODEL_LOAD_S, 3),
        "startup_s": round(STARTUP_S, 3),
        "versions": {p: version(p) for p in ("onnxruntime", "numpy", "pillow", "fastapi")},
    }


@app.get("/whoami")
def whoami(request: Request):
    """How the client IP reaches the app (decides whether per-IP rate limits are viable)."""
    return {
        "client_host": request.client.host if request.client else None,
        "x_forwarded_for": request.headers.get("x-forwarded-for"),
        "forwarded": request.headers.get("forwarded"),
    }


@app.post("/infer")
async def infer(request: Request):
    body = await request.body()
    if len(body) > MAX_BYTES:
        raise HTTPException(413, f"payload {len(body)} B > {MAX_BYTES} B")
    t0 = time.perf_counter()
    try:
        img = Image.open(io.BytesIO(body)).convert("L")
    except Exception as exc:
        raise HTTPException(400, "not a decodable image") from exc
    pixels = np.asarray(img, dtype=np.float32) / 255.0
    t_decode = time.perf_counter()

    # Resample to the dummy model's input width; the real preprocess.py replaces this.
    x = np.resize(pixels.ravel(), (1, IN_DIM)).astype(np.float32)
    out = session.run(None, {INPUT.name: x})[0]
    t_infer = time.perf_counter()
    return {
        "image": list(img.size),
        "payload_bytes": len(body),
        "out_shape": list(out.shape),
        "decode_ms": round((t_decode - t0) * 1000, 2),
        "infer_ms": round((t_infer - t_decode) * 1000, 2),
        "server_ms": round((t_infer - t0) * 1000, 2),
    }
