"""Export the hybrid demo pair to int8 ONNX and check it (ADR-053 Amendment 6).

    uv run --group train python -m ml.src.definition.export_onnx

Definition comes from the 8-head model's overall head, adiposity from the 2-head model's adiposity head. Each is its own
graph: [0, 1] RGB (n, 3, 256, 192) in, 4 CORAL logits out, ImageNet normalisation inside the graph, so the browser only
composites on the grey background, resizes and divides by 255. Only the used head is exported: no regional output can
reach the app. Each model is quantised separately (static QDQ, QUInt8, per-channel weights).

Pre-registered (ADR-053 Amendment 6; the first run, MinMax calibration, failed parity: agreement 0.37 / 0.23):
  variants, fixed, nothing added later, run in this order and stopped at the first that passes for BOTH models:
    v1  per-channel weights + percentile calibration (ORT default percentile)
    v2  v1 with entropy calibration and the four overflow-warned convs (bias onnx::Conv_412/415/421/424) left in fp32
    v3  percentile calibration, all depthwise convs plus the first and last conv left in fp32
  calibration   384 TRAINING identities, stratified over sex x training rig; never val, never test
  scoring       val identities, front view, against torch: fp32 graph max |logit diff| <= 2e-4 (the Siamese export's
                tolerance); int8 predicted-class agreement >= 0.95, no sample off by > 1 class, and metric drift within
                0.01 (rho >= torch - 0.01, MAE <= torch + 0.01: the tolerance of ADR-053 Amendment 5 item 7)
  time box      45 minutes of wall clock, checked before each variant starts
  size          combined int8 <= 5 MB (the fp32 file sizes are measured and printed)
  latency       combined p50 and p95 <= 300 ms. Python ORT, 1 CPU thread, PNG decode + composite + resize + both
                forward passes: a LOCAL PROXY. The in-browser number is measured after the web wiring.
No variant passing, or a failed size or latency, means the pre-registered fallback
(8-head model alone, owner override of
the adiposity rule). The test split is never read here.
"""

import argparse
import hashlib
import json
import shutil
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from torch import nn

from ml.src.definition.data import MEAN, STD, DefinitionDataset
from ml.src.definition.evaluate import head_metrics
from ml.src.definition.evaluate8 import AB_TOLERANCE
from ml.src.definition.items import load_items
from ml.src.definition.model import DefinitionNet
from ml.src.definition.model8 import DefinitionNet8
from ml.src.definition.train import RUNS

ROOT = Path(__file__).resolve().parents[3]
WORK_DIR = ROOT / "ml" / "data" / "export"  # gitignored: fp32 graphs, variants, fixture, report
WEB_DIR = ROOT / "apps" / "web" / "public" / "models"
EXPECTED_SHA = {  # the frozen checkpoints (dev-log, ADR-053 Amendment 6); a different file is refused
    "definition8_finetune": "f0a1314d886324ca4e078226543d533db5377d8e49e303ece9fcfe1a9ca2661f",
    "definition_finetune": "c1142f758fe62333c35278ada3fefaf1b7cc49abadf847881c72f2846a91a123",
}
SPECS = (  # (output name, run folder, model class, head)
    ("definition", "definition8_finetune", DefinitionNet8, "definition"),
    ("adiposity", "definition_finetune", DefinitionNet, "adiposity"),
)
VARIANTS = ("v1", "v2", "v3")
OVERFLOW_BIASES = (
    "onnx::Conv_412",
    "onnx::Conv_415",
    "onnx::Conv_421",
    "onnx::Conv_424",
)  # warned in the first run
FP32_ATOL = 2e-4
INT8_AGREEMENT_MIN, INT8_MAX_STEP, DRIFT_TOL = 0.95, 1, AB_TOLERANCE
MAX_COMBINED_MB, MAX_COMBINED_MS = 5.0, 300.0
TIME_BOX_MIN = 45
CALIBRATION_IDENTITIES, FIXTURE_IMAGES = 384, 16
WARMUP, RUNS_TIMED = 10, 100


class SingleHead(nn.Module):
    """[0, 1] RGB (n, 3, H, W) -> (n, 4) logits of one head. Eval mode: no dropout."""

    def __init__(self, model: nn.Module, head: str):
        super().__init__()
        self.trunk, self.head = model.trunk, model.heads[head]
        self.register_buffer("mean", torch.from_numpy(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.from_numpy(STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk((x - self.mean) / self.std))


def unit_tensor(img) -> np.ndarray:
    """PIL RGB -> float32 (3, H, W) in [0, 1]: all the preprocessing the browser has to reproduce."""
    return np.ascontiguousarray(np.asarray(img, np.float32).transpose(2, 0, 1) / 255)


def classes(logits: np.ndarray) -> np.ndarray:
    return (logits > 0).sum(-1)


def agreement(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """(share of identical predicted classes, largest class difference) between two logit arrays."""
    ca, cb = classes(a), classes(b)
    return float((ca == cb).mean()), int(np.abs(ca - cb).max())


def within_drift(m: dict, reference: dict) -> bool:
    return bool(m["spearman"] >= reference["spearman"] - DRIFT_TOL and m["mae"] <= reference["mae"] + DRIFT_TOL)


def excluded_nodes(model, variant: str) -> list[str]:
    """Names of the Conv nodes a variant leaves in fp32 (`model` is an onnx ModelProto)."""
    convs = [n for n in model.graph.node if n.op_type == "Conv"]
    if variant == "v1":
        return []
    if variant == "v2":
        names = [n.name for n in convs if set(n.input) & set(OVERFLOW_BIASES)]
        if len(names) != len(OVERFLOW_BIASES):
            raise SystemExit(f"v2: expected {len(OVERFLOW_BIASES)} convs using {OVERFLOW_BIASES}, found {names}")
        return names
    depthwise = [n for n in convs if next((a.i for a in n.attribute if a.name == "group"), 1) > 1]
    return sorted({n.name for n in depthwise} | {convs[0].name, convs[-1].name})


class Reader(CalibrationDataReader):
    def __init__(self, xs: np.ndarray):
        self._it = iter({"x": x[None]} for x in xs)

    def get_next(self):
        return next(self._it, None)


def quantize(pre: Path, out: Path, variant: str, xs: np.ndarray) -> int:
    """Quantise `pre` into `out`; returns how many nodes stayed in fp32."""
    import onnx

    exclude = excluded_nodes(onnx.load(str(pre)), variant)
    if any(not n for n in exclude):
        raise SystemExit(f"{variant}: a node to exclude has no name; the quantiser cannot skip it")
    method = CalibrationMethod.Entropy if variant == "v2" else CalibrationMethod.Percentile
    quantize_static(str(pre), str(out), Reader(xs), quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QUInt8, per_channel=True,
                    calibrate_method=method, nodes_to_exclude=exclude)  # fmt: skip
    return len(exclude)


def session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def run_batches(sess: ort.InferenceSession, xs: np.ndarray, batch: int = 64) -> np.ndarray:
    return np.concatenate([sess.run(None, {"x": xs[i : i + batch]})[0] for i in range(0, len(xs), batch)])


def calibration_items(train: list, n: int, seed: int = 0) -> list:
    """n training identities spread evenly over sex x training rig (all groups present)."""
    groups: dict[tuple, list] = {}
    for it in train:
        groups.setdefault((it.sex, it.rig), []).append(it)
    rng = np.random.default_rng(seed)
    per = -(-n // len(groups))
    picked = [g[k] for g in groups.values() for k in rng.choice(len(g), min(per, len(g)), replace=False)]
    return picked[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = ap.parse_args()
    t_start = time.monotonic()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    cfgs = {run: json.loads((RUNS / run / "config.json").read_text()) for _, run, _, _ in SPECS}
    for run, sha in EXPECTED_SHA.items():
        got = hashlib.sha256((RUNS / run / "best.pt").read_bytes()).hexdigest()
        if got != sha:
            raise SystemExit(f"{run}/best.pt is not the frozen checkpoint ({got} != {sha}); nothing exported")

    cfg = cfgs["definition_finetune"]  # both runs share these (checked below)
    assert all(cfgs[r][k] == cfg[k] for r in cfgs for k in ("labels_csv", "params_csv", "images_dir", "set_name",
                                        "height")), \
        "the two runs must read the same data at the same size"  # fmt: skip
    items = load_items(cfg["labels_csv"], cfg["params_csv"], cfg["set_name"])
    train = [i for i in items if i.split == "train"]
    val = [i for i in items if i.split == "val"]
    calib = calibration_items(train, CALIBRATION_IDENTITIES)
    aug = DefinitionDataset(calib, cfg["images_dir"], cfg["set_name"], cfg["height"], train=True, seed=0)
    ev = DefinitionDataset(val, cfg["images_dir"], cfg["set_name"], cfg["height"])
    x_calib = np.stack([unit_tensor(aug.image(i)) for i in calib])
    x_val = np.stack([unit_tensor(ev.image(i)) for i in val])
    sexes = sorted({i.sex for i in calib})
    print(f"calibration {len(x_calib)} train identities ({', '.join(sexes)}; rigs "
          f"{sorted({i.rig for i in calib})}),"
          f" training augmentation; parity on {len(x_val)} val identities")  # fmt: skip

    stage = {}  # per model: fp32 graph, torch reference
    for name, run, cls, head in SPECS:
        model = cls(pretrained=False)
        model.load_state_dict(torch.load(RUNS / run / "best.pt", map_location="cpu"))
        net = SingleHead(model.eval(), head).eval()
        fp32, pre = WORK_DIR / f"{name}.fp32.onnx", WORK_DIR / f"{name}.pre.onnx"
        torch.onnx.export(net, torch.zeros(1, 3, x_val.shape[2], x_val.shape[3]), fp32, input_names=["x"],
                          output_names=["logits"], dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}},
                          opset_version=17, dynamo=False)  # fmt: skip
        quant_pre_process(str(fp32), str(pre))
        with torch.no_grad():
            expected = np.concatenate(
                [net(torch.from_numpy(x_val[i : i + 64])).numpy() for i in range(0, len(x_val), 64)]
            )
        fp32_diff = float(np.abs(run_batches(session(fp32), x_val) - expected).max())
        true = np.array([getattr(i, head) - 1 for i in val])
        stage[name] = {"run": run, "head": head, "pre": pre, "expected": expected, "true": true,
                       "torch": head_metrics(classes(expected), true),
                       "fp32_mb": fp32.stat().st_size / 1e6}  # fmt: skip
        print(f"{name:<10} fp32 graph {stage[name]['fp32_mb']:.2f} MB  max|onnx - torch| {fp32_diff:.2e} "
              f"(<= {FP32_ATOL:.0e}): {'PASS' if fp32_diff <= FP32_ATOL else 'FAIL'}   "
              f"val torch rho {stage[name]['torch']['spearman']:.3f} "
              f"MAE {stage[name]['torch']['mae']:.3f}")  # fmt: skip
        if fp32_diff > FP32_ATOL:
            raise SystemExit("fp32 PARITY FAILED: stop and report; no quantisation")
    fp32_8 = stage["definition"]["fp32_mb"]
    print(f"fp32 combined {sum(s['fp32_mb'] for s in stage.values()):.2f} MB; the lone fp32 8-head graph is "
          f"{fp32_8:.2f} MB "
          f"({'within' if fp32_8 <= MAX_COMBINED_MB else 'OVER'} the {MAX_COMBINED_MB} MB cap)")  # fmt: skip

    report: dict = {"fp32_mb": {n: s["fp32_mb"] for n, s in stage.items()}, "variants": {}}
    chosen = None
    for variant in args.variants:
        elapsed = (time.monotonic() - t_start) / 60
        if elapsed > TIME_BOX_MIN:
            report["variants"][variant] = {"not_run": f"time box: {elapsed:.1f} min elapsed"}
            print(f"{variant}: not run, time box ({elapsed:.1f} min > {TIME_BOX_MIN})")
            continue
        t0, passed = time.monotonic(), True
        report["variants"][variant] = {}
        for name, s in stage.items():
            out = WORK_DIR / f"{name}.{variant}.int8.onnx"
            n_fp32 = quantize(s["pre"], out, variant, x_calib)
            got = run_batches(session(out), x_val)
            agree, step = agreement(got, s["expected"])
            m = head_metrics(classes(got), s["true"])
            ok = agree >= INT8_AGREEMENT_MIN and step <= INT8_MAX_STEP and within_drift(m, s["torch"])
            passed &= ok
            report["variants"][variant][name] = {"nodes_left_fp32": n_fp32, "size_mb": out.stat().st_size / 1e6,
                                                 "agreement": agree, "max_class_step": step,
                                                 "val_int8": m, "pass": ok}  # fmt: skip
            print(f"{variant} {name:<10} int8 {out.stat().st_size / 1e6:.2f} MB ({n_fp32} convs fp32)  "
            f"agreement {agree:.3f}  "
                  f"max step {step}  val rho {m['spearman']:.3f} (torch {s['torch']['spearman']:.3f})  "
                  f"MAE {m['mae']:.3f} (torch {s['torch']['mae']:.3f})  {'PASS' if ok else 'FAIL'}")  # fmt: skip
        print(f"{variant}: {'PASS for both models' if passed else 'FAIL'}  ({(time.monotonic() - t0) / 60:.1f} min)")
        if passed:
            chosen = variant
            break
    report["chosen"] = chosen
    (WORK_DIR / "hybrid_export.json").write_text(json.dumps(report, indent=1, default=float))
    if chosen is None:
        print("NO VARIANT PASSED: log all results in dev-log and use the pre-registered fallback (8-head model alone, "
              "owner override of the adiposity rule, ADR-053 Amendment 6). No threshold changes.")  # fmt: skip
        return

    sessions, total_mb, fixture = [], 0.0, {}
    for name, s in stage.items():
        final = WEB_DIR / f"{name}.int8.onnx"
        shutil.copyfile(WORK_DIR / f"{name}.{chosen}.int8.onnx", final)
        total_mb += final.stat().st_size / 1e6
        sessions.append(session(final))
        got = run_batches(sessions[-1], x_val[:FIXTURE_IMAGES])
        fixture[f"{name}_logits_torch"], fixture[f"{name}_logits_int8"] = (
            s["expected"][:FIXTURE_IMAGES],
            got,
        )
        report["variants"][chosen][name]["int8_sha256"] = hashlib.sha256(final.read_bytes()).hexdigest()
        report["variants"][chosen][name]["checkpoint_sha256"] = EXPECTED_SHA[s["run"]]
    x_u8 = np.round(x_val[:FIXTURE_IMAGES] * 255).astype(np.uint8)
    x_u8.tofile(WORK_DIR / "fixture.bin")  # raw (16, 3, H, W) uint8 for the ONNX Runtime Web parity script
    (WORK_DIR / "fixture.json").write_text(json.dumps(
        {"shape": list(x_u8.shape), **{k: v.tolist() for k, v in fixture.items()}}))  # fmt: skip

    times = []  # latency of the pair on one front image: PNG decode + composite + resize + both forward passes
    for k in range(WARMUP + RUNS_TIMED):
        t0 = time.perf_counter()
        x = unit_tensor(ev.image(val[k % len(val)]))[None]
        for sess in sessions:
            sess.run(None, {"x": x})
        if k >= WARMUP:
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    p50, p95 = statistics.median(times), times[int(0.95 * len(times)) - 1]
    size_ok, lat_ok = total_mb <= MAX_COMBINED_MB, p50 <= MAX_COMBINED_MS and p95 <= MAX_COMBINED_MS
    report["combined"] = {"variant": chosen, "size_mb": total_mb, "size_pass": size_ok,
                          "latency_ms_p50_p95_1thread": [p50, p95], "latency_pass": lat_ok,
                          "latency_note": "local Python ORT proxy; in-browser number is separate"}  # fmt: skip
    (WORK_DIR / "hybrid_export.json").write_text(json.dumps(report, indent=1, default=float))
    print(f"chosen {chosen}: combined int8 {total_mb:.2f} MB (<= {MAX_COMBINED_MB}): {'PASS' if size_ok else 'FAIL'}")
    print(f"combined latency (1 thread, proxy) p50 {p50:.1f} ms  p95 {p95:.1f} ms (<= {MAX_COMBINED_MS}): "
          f"{'PASS' if lat_ok else 'FAIL'}")  # fmt: skip
    print(f"files in {WEB_DIR}; report, fixture.bin and fixture.json in {WORK_DIR}")
    print("next: ONNX Runtime Web parity, `cd apps/web && node scripts/ort_web_parity.mjs`")
    if not (size_ok and lat_ok):
        print("FALLBACK RULE: serve the 8-head model alone and record the owner override (ADR-053 Amendment 6)")


if __name__ == "__main__":
    main()
