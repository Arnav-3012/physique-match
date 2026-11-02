"""Reliability block for the frame V-taper (ADR-038): residual and prior std of the index
shoulders / waist on val, written as a `v_taper` key into gap_regressor.json. The ONNX is not
re-exported; the block records what it was computed from. Commit this file first, so the recorded
commit contains it and `tree_clean` is true. Run:
  uv run --group train python -m ml.src.v_taper_block --ckpt ml/data/runs/baseline_b_full/best.pt
Reads val only, never Test-B. Re-running ml.src.export_onnx rewrites the JSON and drops the block.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from ml.src.calibrate import index_sigmas, reliability
from ml.src.data.dataset import Views
from ml.src.export_onnx import META_PATH, ONNX_PATH
from ml.src.predict import load_checkpoint, person_values

ROOT = Path(__file__).resolve().parents[2]
VERSION = 1


def git(*args: str) -> str:
    done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return done.stdout.strip()


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    args = ap.parse_args()

    meta = json.loads(META_PATH.read_text())
    model_hash = sha256(args.ckpt)
    if meta["model_hash"] != model_hash:
        raise SystemExit(f"{args.ckpt} ({model_hash[:12]}) is not the exported model")

    val = Views("val")
    model = load_checkpoint(args.ckpt, "cpu")
    full, front = person_values(model, val.pick(np.arange(len(val))), "cpu")
    block = {
        "version": VERSION,
        "script_commit": git("rev-parse", "HEAD"),
        "tree_clean": not git("status", "--porcelain"),
        "model_hash": model_hash,
        "onnx_sha256": sha256(ONNX_PATH),
        "n_val_subjects": len(val),
    } | index_sigmas(full, front, val.y.numpy())
    META_PATH.write_text(json.dumps(meta | {"v_taper": block}, indent=1))

    rel_full = reliability(block["sigma_full"], block["sigma_full"], block["sigma_prior"])
    rel_front = reliability(block["sigma_front"], block["sigma_full"], block["sigma_prior"])
    print(f"v_taper block v{VERSION}: model {model_hash[:12]}, {len(val)} val subjects")
    tree = "clean" if block["tree_clean"] else "DIRTY"
    print(f"commit {block['script_commit'][:12]}, tree {tree}")
    print(", ".join(f"{k} {block[k]:.4f}" for k in ("sigma_full", "sigma_front", "sigma_prior")))
    print(f"reliability: full-input goal {rel_full:.2f}, front-only goal {rel_front:.2f}")


if __name__ == "__main__":
    main()
