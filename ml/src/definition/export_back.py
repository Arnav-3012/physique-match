"""Export the 8-head model's `back` head as a third web graph (ADR-061, owner override 2026-11-03).
Reads val only, never test.

    uv run --group train python -m ml.src.definition.export_back

The head did NOT beat the overall score in the incremental gate (ADR-053 Amendment 5 item 8), so the owner
asked for it as an experimental, clearly labelled comparison. Same graph shape as the other two:
[0, 1] RGB (n, 3, 256, 192) in, `logits` (n, 4) out, ImageNet normalisation inside.
Fixed before the first run: the graph matches its checkpoint within FP32_ATOL on the val back-view images,
and the file reaches the served folder only after that passes. The head's own val metrics are printed
next to the P4 gate (rho >= 0.8, MAE <= 0.6) and are NOT a blocker: the owner decided to ship it experimental.
"""

import hashlib
import json
import shutil

import numpy as np
import torch

from ml.src.definition.data8 import DefinitionDataset8
from ml.src.definition.evaluate import gate, head_metrics
from ml.src.definition.export_onnx import (
    EXPECTED_SHA,
    FP32_ATOL,
    WEB_DIR,
    WORK_DIR,
    SingleHead,
    classes,
    run_batches,
    session,
    unit_tensor,
)
from ml.src.definition.items8 import load_items8, samples
from ml.src.definition.model8 import DefinitionNet8
from ml.src.definition.train import RUNS
from ml.src.definition.views import HEADS8

RUN, HEAD, NAME = "definition8_finetune", "back", "back"


def main() -> None:
    # the data paths, read the way export_fp32 reads them
    cfg = json.loads((RUNS / "definition_finetune" / "config.json").read_text())
    val = [i for i in load_items8(cfg["labels_csv"], cfg["params_csv"], cfg["set_name"]) if i.split == "val"]
    back = samples(val, ("back",))
    ds = DefinitionDataset8(back, cfg["images_dir"], cfg["set_name"], cfg["height"])
    x_val = np.stack([unit_tensor(ds.image(s)) for s in back])
    y_val = np.array([s.item.labels[HEADS8.index(HEAD)] - 1 for s in back])
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    ckpt = RUNS / RUN / "best.pt"
    sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    if sha != EXPECTED_SHA[RUN]:
        raise SystemExit(f"{ckpt} is not the frozen checkpoint ({sha}); nothing exported")
    model = DefinitionNet8(pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net = SingleHead(model.eval(), HEAD).eval()
    path = WORK_DIR / f"{NAME}.fp32.onnx"
    torch.onnx.export(
        net,
        torch.zeros(1, 3, x_val.shape[2], x_val.shape[3]),
        path,
        input_names=["x"],
        output_names=["logits"],
        dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}},
        opset_version=17,
        dynamo=False,
    )
    with torch.no_grad():
        expected = np.concatenate([net(torch.from_numpy(x_val[i : i + 64])).numpy() for i in range(0, len(x_val), 64)])
    got = run_batches(session(path), x_val)
    diff = float(np.abs(got - expected).max())
    parity = diff <= FP32_ATOL
    metrics = head_metrics(classes(got), y_val)
    onnx_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    size_mb = path.stat().st_size / 1e6
    report = {
        "run": RUN,
        "head": HEAD,
        "n_val_back_images": len(back),
        "checkpoint_sha256": sha,
        "onnx_sha256": onnx_sha,
        "size_mb": size_mb,
        "max_abs_diff": diff,
        "parity_pass": parity,
        "val": metrics,
        "p4_gate_pass": bool(gate(metrics)),
    }
    (WORK_DIR / "back_export.json").write_text(json.dumps(report, indent=1, default=float))
    print(
        f"{NAME}: {size_mb:.2f} MB  max|onnx - torch| {diff:.2e} (<= {FP32_ATOL:.0e}): {'PASS' if parity else 'FAIL'}"
    )
    rho, mae, exact = metrics["spearman"], metrics["mae"], metrics["exact"]
    print(f"val back images n={len(back)}  rho {rho:.3f}  MAE {mae:.3f}  exact {exact:.3f}")
    verdict = "PASS" if report["p4_gate_pass"] else "FAIL"
    print(f"P4 gate (rho >= 0.8, MAE <= 0.6): {verdict} (reported, not a blocker: owner override)")
    print(f"onnx sha256 {onnx_sha}")
    if not parity:
        print("PARITY FAILED: nothing copied")
        return
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, WEB_DIR / path.name)
    print(f"copied to {WEB_DIR / path.name}; add a .gitignore exception for it, then wire the web app")


if __name__ == "__main__":
    main()
