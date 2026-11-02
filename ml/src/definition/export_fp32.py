"""Export the hybrid demo pair in fp32 (ADR-053 Amendment 6: owner cap 8 MB combined). Reads val only, never test.

    uv run --group train python -m ml.src.definition.export_fp32

Definition from the 8-head model, adiposity from the 2-head model. Each graph holds only its one used head: [0, 1] RGB
(n, 3, 256, 192) in, `logits` (n, 4) out, ImageNet normalisation inside the graph.
Fixed before the first run: each graph
matches its scored checkpoint within 2e-4 on val; combined size <= 8 MB. The files reach the served folder only after
both pass. Next checks: ONNX Runtime Web parity on the fixture, then combined latency in a real browser tab.
"""

import hashlib
import json
import shutil

import numpy as np
import torch

from ml.src.definition.data import DefinitionDataset
from ml.src.definition.evaluate import head_metrics
from ml.src.definition.export_onnx import (
    EXPECTED_SHA,
    FIXTURE_IMAGES,
    FP32_ATOL,
    SPECS,
    WEB_DIR,
    WORK_DIR,
    SingleHead,
    classes,
    run_batches,  # fmt: skip
    session,
    unit_tensor,
)
from ml.src.definition.items import load_items
from ml.src.definition.train import RUNS

MAX_COMBINED_MB = 8.0  # owner decision, raised from 5 MB after the int8 variants failed
STALE = (
    "definition.int8.onnx",
    "adiposity.int8.onnx",
    "definition.onnx",
)  # failed int8 files, the fallback's graph


def main() -> None:
    stale = [n for n in STALE if (WEB_DIR / n).exists()]
    if stale:
        raise SystemExit(f"delete from {WEB_DIR} first: {stale}")
    cfg = json.loads((RUNS / "definition_finetune" / "config.json").read_text())
    val = [i for i in load_items(cfg["labels_csv"], cfg["params_csv"], cfg["set_name"]) if i.split == "val"]
    ds = DefinitionDataset(val, cfg["images_dir"], cfg["set_name"], cfg["height"])
    x_val = np.stack([unit_tensor(ds.image(i)) for i in val])
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    report, fixture, files, passed, total_mb = {}, {}, [], True, 0.0
    for name, run, cls, head in SPECS:
        ckpt = RUNS / run / "best.pt"
        sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        if sha != EXPECTED_SHA[run]:
            raise SystemExit(f"{ckpt} is not the frozen checkpoint ({sha}); nothing exported")
        model = cls(pretrained=False)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        net = SingleHead(model.eval(), head).eval()
        path = WORK_DIR / f"{name}.fp32.onnx"
        torch.onnx.export(net, torch.zeros(1, 3, x_val.shape[2], x_val.shape[3]), path, input_names=["x"],
                          output_names=["logits"], dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}},
                          opset_version=17, dynamo=False)  # fmt: skip
        with torch.no_grad():
            expected = np.concatenate(
                [net(torch.from_numpy(x_val[i : i + 64])).numpy() for i in range(0, len(x_val), 64)]
            )
        got = run_batches(session(path), x_val)
        diff = float(np.abs(got - expected).max())
        ok = diff <= FP32_ATOL
        passed &= ok
        size_mb = path.stat().st_size / 1e6
        total_mb += size_mb
        onnx_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        m = head_metrics(classes(got), np.array([getattr(i, head) - 1 for i in val]))
        report[name] = {"run": run, "head": head, "checkpoint_sha256": sha, "onnx_sha256": onnx_sha, "size_mb": size_mb,
                        "max_abs_diff": diff, "parity_pass": ok, "val": m}  # fmt: skip
        fixture[f"{name}_logits"] = got[:FIXTURE_IMAGES].tolist()
        files.append(path)
        print(f"{name:<10} {run}: {size_mb:.2f} MB  max|onnx - torch| {diff:.2e} (<= {FP32_ATOL:.0e}): "
              f"{'PASS' if ok else 'FAIL'}  val rho {m['spearman']:.3f} MAE {m['mae']:.3f}"
              f"\n           onnx sha256 {onnx_sha}")  # fmt: skip

    size_ok = total_mb <= MAX_COMBINED_MB
    report["combined_mb"], report["size_pass"] = total_mb, size_ok
    (WORK_DIR / "fp32_export.json").write_text(json.dumps(report, indent=1, default=float))
    print(f"combined {total_mb:.2f} MB (<= {MAX_COMBINED_MB}): {'PASS' if size_ok else 'FAIL'}")
    if not (passed and size_ok):
        print("CHECK FAILED: nothing copied; the pre-registered fallback applies (ADR-053 Amendment 6)")
        return
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    for path in files:
        shutil.copyfile(path, WEB_DIR / path.name)
    x_u8 = np.round(x_val[:FIXTURE_IMAGES] * 255).astype(np.uint8)
    x_u8.tofile(WORK_DIR / "fixture.bin")  # raw (16, 3, H, W) uint8 for the ONNX Runtime Web parity script
    (WORK_DIR / "fixture.json").write_text(json.dumps({"shape": list(x_u8.shape), **fixture}))
    print("ALL CHECKS PASS: copied to the served folder; next `node scripts/ort_web_parity.mjs` from apps/web")


if __name__ == "__main__":
    main()
