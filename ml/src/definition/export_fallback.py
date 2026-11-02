"""Export the demo model after the int8 variants failed (ADR-053 Amendment 6): the 8-head model alone, fp32.

    uv run --group train python -m ml.src.definition.export_fallback

One graph: [0, 1] RGB (n, 3, 256, 192) in, outputs `definition` and `adiposity` (4 CORAL logits each), ImageNet
normalisation inside the graph. No regional output is in the graph. Reads val only, never test: the graph has to
match the already scored checkpoint, so the test numbers stand.

Fixed before the first run: max |onnx - torch| logits <= 2e-4 per output; size <= 5 MB; latency p50 and p95 <= 300 ms
(Python ORT, 1 CPU thread, PNG decode + composite + resize + forward: a LOCAL PROXY; the in-browser number is measured
after the web wiring). The file reaches the served folder only after all of these pass.
"""

import hashlib
import json
import shutil
import statistics
import time

import numpy as np
import torch
from torch import nn

from ml.src.definition.data import MEAN, STD, DefinitionDataset
from ml.src.definition.evaluate import head_metrics
from ml.src.definition.export_onnx import (
    EXPECTED_SHA,
    FIXTURE_IMAGES,
    FP32_ATOL,
    MAX_COMBINED_MB,
    MAX_COMBINED_MS,
    RUNS_TIMED,
    WARMUP,  # fmt: skip
    WEB_DIR,
    WORK_DIR,
    classes,
    session,
    unit_tensor,
)
from ml.src.definition.items import load_items
from ml.src.definition.model8 import DefinitionNet8
from ml.src.definition.train import RUNS

RUN = "definition8_finetune"
OUTPUTS = ("definition", "adiposity")
FILE_NAME = "definition.onnx"
STALE = (
    "definition.int8.onnx",
    "adiposity.int8.onnx",
)  # failed parity in the first export run; never to be served


class FrontHeads(nn.Module):
    """[0, 1] RGB (n, 3, H, W) -> (definition, adiposity) logits, each (n, 4). Eval mode: no dropout."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.trunk, self.definition, self.adiposity = (
            model.trunk,
            model.heads["definition"],
            model.heads["adiposity"],
        )
        self.register_buffer("mean", torch.from_numpy(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.from_numpy(STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.trunk((x - self.mean) / self.std)
        return self.definition(z), self.adiposity(z)


def main() -> None:
    stale = [n for n in STALE if (WEB_DIR / n).exists()]
    if stale:
        raise SystemExit(f"delete the failed int8 files from {WEB_DIR} first: {stale}")
    ckpt = RUNS / RUN / "best.pt"
    sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    if sha != EXPECTED_SHA[RUN]:
        raise SystemExit(f"{ckpt} is not the frozen checkpoint ({sha}); nothing exported")
    cfg = json.loads((RUNS / RUN / "config.json").read_text())
    val = [i for i in load_items(cfg["labels_csv"], cfg["params_csv"], cfg["set_name"]) if i.split == "val"]
    ds = DefinitionDataset(val, cfg["images_dir"], cfg["set_name"], cfg["height"])
    x_val = np.stack([unit_tensor(ds.image(i)) for i in val])

    model = DefinitionNet8(pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net = FrontHeads(model.eval()).eval()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    work = WORK_DIR / FILE_NAME
    torch.onnx.export(net, torch.zeros(1, 3, x_val.shape[2], x_val.shape[3]), work, input_names=["x"],
                      output_names=list(OUTPUTS), dynamic_axes={"x": {0: "n"}, **{o: {0: "n"} for o in OUTPUTS}},
                      opset_version=17, dynamo=False)  # fmt: skip
    size_mb = work.stat().st_size / 1e6

    sess = session(work)
    report, parity_ok, fixture = {"checkpoint_sha256": sha, "size_mb": size_mb}, True, {}
    with torch.no_grad():
        batches = [net(torch.from_numpy(x_val[i : i + 64])) for i in range(0, len(x_val), 64)]
    for k, name in enumerate(OUTPUTS):
        expected = np.concatenate([b[k].numpy() for b in batches])
        got = np.concatenate([sess.run([name], {"x": x_val[i : i + 64]})[0] for i in range(0, len(x_val), 64)])
        diff = float(np.abs(got - expected).max())
        ok = diff <= FP32_ATOL
        parity_ok &= ok
        m = head_metrics(classes(got), np.array([getattr(i, name) - 1 for i in val]))
        report[name] = {"max_abs_diff": diff, "parity_pass": ok, "val": m}
        fixture[f"{name}_logits"] = got[:FIXTURE_IMAGES].tolist()
        print(f"{name:<10} max|onnx - torch| {diff:.2e} (<= {FP32_ATOL:.0e}): {'PASS' if ok else 'FAIL'}   "
              f"val rho {m['spearman']:.3f} MAE {m['mae']:.3f}")  # fmt: skip

    times = []
    for k in range(WARMUP + RUNS_TIMED):
        t0 = time.perf_counter()
        sess.run(None, {"x": unit_tensor(ds.image(val[k % len(val)]))[None]})
        if k >= WARMUP:
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    p50, p95 = statistics.median(times), times[int(0.95 * len(times)) - 1]
    size_ok, lat_ok = size_mb <= MAX_COMBINED_MB, p50 <= MAX_COMBINED_MS and p95 <= MAX_COMBINED_MS
    report |= {"size_pass": size_ok, "latency_ms_p50_p95_1thread": [p50, p95], "latency_pass": lat_ok,
               "latency_note": "local Python ORT proxy; in-browser number is separate"}  # fmt: skip
    print(f"size {size_mb:.2f} MB (<= {MAX_COMBINED_MB}): {'PASS' if size_ok else 'FAIL'}")
    print(
        f"latency (1 thread, proxy) p50 {p50:.1f} ms  p95 {p95:.1f} ms "
        f"(<= {MAX_COMBINED_MS}): {'PASS' if lat_ok else 'FAIL'}"
    )

    passed = parity_ok and size_ok and lat_ok
    if passed:
        WEB_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(work, WEB_DIR / FILE_NAME)
        report["onnx_sha256"] = hashlib.sha256(work.read_bytes()).hexdigest()
        x_u8 = np.round(x_val[:FIXTURE_IMAGES] * 255).astype(np.uint8)
        x_u8.tofile(WORK_DIR / "fixture.bin")  # raw (16, 3, H, W) uint8 for the ONNX Runtime Web parity script
        (WORK_DIR / "fixture.json").write_text(json.dumps({"shape": list(x_u8.shape), **fixture}))
        print(f"onnx sha256 {report['onnx_sha256']}")
        print("next: `cd apps/web && npm install onnxruntime-web@1.22.0 && node scripts/ort_web_parity.mjs`")
    (WORK_DIR / "fallback_export.json").write_text(json.dumps(report, indent=1, default=float))
    print("ALL CHECKS PASS: copied to the served folder" if passed else "CHECK FAILED: not copied; stop and report")


if __name__ == "__main__":
    main()
