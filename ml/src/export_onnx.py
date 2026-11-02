"""Export the shipped model (ADR-035): a single-person ONNX that maps one scan to its proportions
(measurement / height per labelled region). The API runs it on the goal and on the current scan
and subtracts. Also checks ONNX against torch, size and single-thread latency, and writes the
calibration. Run:
  uv run --group train python -m ml.src.export_onnx --ckpt ml/data/runs/baseline_b_full/best.pt
Writes services/api/app/analyzer/gap_regressor.onnx (gitignored, goes to HF Hub at deploy) and
gap_regressor.json (committed). Reads val only.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from ml.src.calibrate import calibrate, reliability
from ml.src.data.dataset import Views, as_input
from ml.src.data.regions import LABELLED_REGIONS
from ml.src.predict import load_checkpoint, person_values
from ml.src.preprocess import MODEL_H, MODEL_W

OUT_DIR = Path(__file__).resolve().parents[2] / "services/api/app/analyzer"
ONNX_PATH, META_PATH = OUT_DIR / "gap_regressor.onnx", OUT_DIR / "gap_regressor.json"
MAX_MB = 50
# ADR-035, written before measuring: single-encode p50 <= 75 ms AND two-encode p95 <= 150 ms, one
# CPU thread, local proxy. On a miss: int8, then re-measure.
MAX_ENCODE_P50_MS = 75
MAX_REQUEST_P95_MS = 150
PARITY_ATOL = 2e-4
PARITY_SUBJECTS = 32
WARMUP, RUNS = 10, 100


class Proportions(nn.Module):
    """(n, 3, H, W) in [0, 1] -> (n, regions) measurement / height. Normalisation and label scale
    are inside the graph."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.value(self.model.encoder(x))


def export(model: nn.Module, path: Path) -> None:
    torch.onnx.export(
        Proportions(model).eval(),
        torch.zeros(1, 3, MODEL_H, MODEL_W),
        path,
        input_names=["x"],
        output_names=["proportions"],
        dynamic_axes={"x": {0: "n"}, "proportions": {0: "n"}},
        opset_version=17,
        dynamo=False,  # the legacy exporter needs no onnxscript
    )


def session(path: Path, threads: int | None = None) -> ort.InferenceSession:
    options = ort.SessionOptions()
    if threads:
        options.intra_op_num_threads = options.inter_op_num_threads = threads
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def latency_ms(sess: ort.InferenceSession, x: np.ndarray, encodes: int) -> tuple[float, float]:
    """p50 and p95 of `encodes` sequential single-scan runs."""
    times = []
    for i in range(WARMUP + RUNS):
        start = time.perf_counter()
        for _ in range(encodes):
            sess.run(None, {"x": x})
        if i >= WARMUP:
            times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return statistics.median(times), times[int(0.95 * len(times)) - 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    args = ap.parse_args()

    model = load_checkpoint(args.ckpt, "cpu").eval()
    export(model, ONNX_PATH)
    size_mb = ONNX_PATH.stat().st_size / 1e6

    val = Views("val")
    views = val.pick(np.arange(len(val)))
    x = as_input(views.float() / 255)
    with torch.no_grad():
        expected = Proportions(model)(x[:PARITY_SUBJECTS]).numpy()
    got = session(ONNX_PATH).run(None, {"x": x[:PARITY_SUBJECTS].numpy()})[0]
    parity = float(np.abs(got - expected).max())

    one = x[:1].numpy()
    sess = session(ONNX_PATH, threads=1)
    run_p50, run_p95 = latency_ms(sess, one, encodes=1)
    req_p50, req_p95 = latency_ms(sess, one, encodes=2)

    full, front = person_values(model, views, "cpu")
    meta = {
        "model_hash": hashlib.sha256(args.ckpt.read_bytes()).hexdigest(),
        "input": {"name": "x", "shape": ["n", 3, MODEL_H, MODEL_W],
                  "channels": "front, side, zeros; float in [0, 1]; side is zeros if absent"},
        "output": {"name": "proportions", "regions": list(LABELLED_REGIONS),
                   "unit": "measurement / height_cm"},
        "calibration": calibrate(full, front, val.y.numpy()),
        "export": {"onnx_mb": round(size_mb, 1), "parity_max_abs": parity,
                   "run_ms_p50_p95_1thread": [round(run_p50, 1), round(run_p95, 1)],
                   "request_ms_p50_p95_1thread": [round(req_p50, 1), round(req_p95, 1)]},
    }  # fmt: skip
    META_PATH.write_text(json.dumps(meta, indent=1))

    print(f"model {meta['model_hash'][:12]} -> {ONNX_PATH.name}, {META_PATH.name}")
    print(f"size   {size_mb:.1f} MB (limit {MAX_MB}): {'PASS' if size_mb <= MAX_MB else 'FAIL'}")
    print(
        f"parity max |onnx - torch| {parity:.2e} over {PARITY_SUBJECTS} val subjects "
        f"(tolerance {PARITY_ATOL:.0e}): {'PASS' if parity <= PARITY_ATOL else 'FAIL'}"
    )
    print(
        f"1 thread, one encode:      p50 {run_p50:.1f} ms (limit {MAX_ENCODE_P50_MS}), "
        f"p95 {run_p95:.1f} ms"
    )
    print(
        f"1 thread, two encodes:     p50 {req_p50:.1f} ms, p95 {req_p95:.1f} ms "
        f"(limit {MAX_REQUEST_P95_MS})"
    )
    fast = run_p50 <= MAX_ENCODE_P50_MS and req_p95 <= MAX_REQUEST_P95_MS
    print(f"latency rule (ADR-035): {'PASS' if fast else 'FAIL, next step is int8'}")
    print("region reliability (full-input goal / front-only goal; current has both views):")
    print(f"  {'region':10s} {'full':>6s} {'front':>6s} | sigma_full sigma_front sigma_prior")
    for region, s in meta["calibration"].items():
        rel_full = reliability(s["sigma_full"], s["sigma_full"], s["sigma_prior"])
        rel_front = reliability(s["sigma_front"], s["sigma_full"], s["sigma_prior"])
        print(
            f"  {region:10s} {rel_full:6.2f} {rel_front:6.2f} | {s['sigma_full']:.4f}     "
            f"{s['sigma_front']:.4f}      {s['sigma_prior']:.4f}"
        )


if __name__ == "__main__":
    main()
