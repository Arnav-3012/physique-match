"""Metrics, the incremental gate and the A/B rule for the 8-head model (ADR-053 Amendment 5).

    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --group train python -m ml.src.definition.evaluate8 \\
      --run8 definition8_finetune --run2 definition_finetune

Each head is scored over the samples (identity, view) it is valid on, with the sample count. The incremental gate
works per identity on the same held-out identities. Test is scored once per final candidate (ledger in the run
folder). The functions above `main` need only numpy and scipy.
"""

import argparse
import hashlib
import json

import numpy as np
from scipy.stats import spearmanr

from ml.src.definition.evaluate import GATE_MAE, GATE_RHO, MIN_GROUP, gate, head_metrics
from ml.src.definition.views import HEADS8, REGIONS, valid_mask

INCREMENTAL_MARGIN = 0.05  # placeholder fixed in ADR-053 Amendment 5, never tuned after results
AB_TOLERANCE = 0.01  # same: how much worse the 8-head overall / abs heads may be than the 2-head one
COMPARED = tuple(r for r in REGIONS if r != "abs")  # abs carries the overall label: nothing to compare
REASON_NO_GATE = "regional head did not pass the primary gate"
REASON_NOT_BETTER = "regional head did not beat the overall score"
REASON_ABS = "the abs label is the overall label"


def sample_arrays(samples) -> tuple[np.ndarray, np.ndarray]:
    """(true classes 0-4, valid mask), both (N, 8)."""
    true = np.array([[c - 1 for c in s.item.labels] for s in samples])
    mask = np.array([valid_mask(s.view) for s in samples])
    return true, mask


def masked_head_metrics(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> dict[str, dict]:
    """Per head, over the samples it is valid on; `n` is that sample count."""
    out = {}
    for k, h in enumerate(HEADS8):
        v = mask[:, k]
        out[h] = head_metrics(pred[v, k], true[v, k]) if v.any() else {"n": 0, "spearman": float("nan"),
                                                                       "mae": float("nan"),
                                                                       "exact": float("nan")}  # fmt: skip
    return out


def identity_level(samples, pred: np.ndarray):
    """(ids, predictions (I, 8), true (I, 8)): per identity, a head's prediction is the mean over the views it is
    valid on (NaN when it has none). Identities keep their order of first appearance."""
    total, count, true, order = {}, {}, {}, []
    for s, p in zip(samples, pred, strict=False):
        if s.id not in total:
            order.append(s.id)
            total[s.id], count[s.id], true[s.id] = (
                np.zeros(len(HEADS8)),
                np.zeros(len(HEADS8)),
                np.array(s.item.labels) - 1,
            )
        m = np.array(valid_mask(s.view))
        total[s.id] += np.where(m, p, 0)
        count[s.id] += m
    mean = np.array([np.where(count[i] > 0, total[i] / np.maximum(count[i], 1), np.nan) for i in order])
    return order, mean, np.array([true[i] for i in order])


def _rho(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    return float(spearmanr(a, b).statistic) if len(a) > 2 and len(set(a)) > 1 and len(set(b)) > 1 else float("nan")


def decide(primary_pass: bool, rho_region: float, rho_overall: float) -> tuple[str, str]:
    """The display rule of ADR-053 Amendment 5 item 8: ('regional' | 'overall' | 'na', reason). NaN fails."""
    if not primary_pass:
        return "na", REASON_NO_GATE
    if not rho_region - rho_overall >= INCREMENTAL_MARGIN:
        return "overall", REASON_NOT_BETTER
    return "regional", ""


def regional_decisions(samples, pred: np.ndarray, metrics: dict[str, dict]) -> dict[str, dict]:
    """Per region head: its identity-level Spearman, the overall head's against the same label, the decision."""
    _, mean, true = identity_level(samples, pred)
    overall = mean[:, HEADS8.index("definition")]
    out = {"abs": {"decision": "overall", "reason": REASON_ABS}}
    for r in COMPARED:
        k = HEADS8.index(r)
        rho_r, rho_o = _rho(mean[:, k], true[:, k]), _rho(overall, true[:, k])
        decision, reason = decide(gate(metrics[r]), rho_r, rho_o)
        out[r] = {"decision": decision, "reason": reason, "rho_region": rho_r, "rho_overall": rho_o}
    return out


def not_worse(m8: dict, m2: dict) -> bool:
    """The pre-registered rule (Amendment 5 item 7): rho_8 >= rho_2 - tolerance and MAE_8 <= MAE_2 + tolerance."""
    return bool(m8["spearman"] >= m2["spearman"] - AB_TOLERANCE and m8["mae"] <= m2["mae"] + AB_TOLERANCE)


def ab_check(m8: dict[str, dict], m2: dict) -> dict[str, bool]:
    """The 8-head `definition` and `abs` heads each not worse than the 2-head `definition` head, same identities."""
    ok = {h: not_worse(m8[h], m2) for h in ("definition", "abs")}
    return {**ok, "ship_8_head": all(ok.values())}


def front_only(samples, pred: np.ndarray, true: np.ndarray, head: str) -> dict:
    """A head over the front-view samples only: the like-for-like comparison with the front-only 2-head model."""
    front = np.array([s.view == "front" for s in samples])
    k = HEADS8.index(head)
    return head_metrics(pred[front, k], true[front, k])


def sex_metrics(samples, pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> dict[str, dict[str, dict]]:
    """Per sex with at least MIN_GROUP identities (ADR-045 item 5), each head over its valid samples."""
    out = {}
    for sex in sorted({s.item.sex for s in samples}):
        idx = np.array([s.item.sex == sex for s in samples])
        if len({s.id for s, i in zip(samples, idx, strict=False) if i}) >= MIN_GROUP:
            out[sex] = masked_head_metrics(pred[idx], true[idx], mask[idx])
    return out


def predict_samples(model, dataset, device: str, batch_size: int = 64) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader

    from ml.src.definition.model import predict_class

    model.eval()
    out = []
    with torch.no_grad():
        for x, _, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            out.append(predict_class(model(x.to(device))).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    import torch

    from ml.src.definition.data import DefinitionDataset
    from ml.src.definition.data8 import DefinitionDataset8
    from ml.src.definition.evaluate import predict
    from ml.src.definition.items import load_items
    from ml.src.definition.items8 import load_items8, samples
    from ml.src.definition.model import DefinitionNet
    from ml.src.definition.model8 import DefinitionNet8
    from ml.src.definition.train import RUNS, device_name

    ap = argparse.ArgumentParser()
    ap.add_argument("--run8", required=True, help="a folder under ml/data/runs with best.pt and config.json")
    ap.add_argument("--run2", required=True, help="the 2-head demo model's run folder (read, not reselected)")
    ap.add_argument("--again", action="store_true", help="score the test split a second time (log why)")
    args = ap.parse_args()
    run8, run2 = RUNS / args.run8, RUNS / args.run2
    cfg8, cfg2 = (
        json.loads((run8 / "config.json").read_text()),
        json.loads((run2 / "config.json").read_text()),
    )
    sha8, sha2 = (hashlib.sha256((r / "best.pt").read_bytes()).hexdigest() for r in (run8, run2))
    ledger = run8 / "test_scored.json"
    if ledger.exists() and not args.again:
        raise SystemExit(f"{ledger} exists: the test split is scored once per final candidate (ADR-045)")

    device = device_name()
    model8 = DefinitionNet8(pretrained=False).to(device)
    model8.load_state_dict(torch.load(run8 / "best.pt", map_location=device))
    model2 = DefinitionNet(pretrained=False).to(device)
    model2.load_state_dict(torch.load(run2 / "best.pt", map_location=device))

    test = [i for i in load_items8(cfg8["labels_csv"], cfg8["params_csv"], cfg8["set_name"]) if i.split == "test"]
    test_samples = samples(test)
    pred = predict_samples(
        model8,
        DefinitionDataset8(test_samples, cfg8["images_dir"], cfg8["set_name"], cfg8["height"]),
        device,
    )
    true, mask = sample_arrays(test_samples)
    metrics = masked_head_metrics(pred, true, mask)
    print(f"test identities {len(test)}  samples {len(test_samples)}  8-head sha256 {sha8}  2-head sha256 {sha2}")
    for h, m in metrics.items():
        print(f"{h:<11} n {m['n']:>5}  rho {m['spearman']:.3f}  MAE {m['mae']:.3f}  exact {m['exact']:.3f}  "
              f"{'PASS' if gate(m) else 'FAIL'}")  # fmt: skip
    print(f"primary gate per head: rho >= {GATE_RHO}, MAE <= {GATE_MAE}")

    decisions = regional_decisions(test_samples, pred, metrics)
    for r, d in decisions.items():
        extra = f"  rho region {d['rho_region']:.3f} vs overall {d['rho_overall']:.3f}" if "rho_region" in d else ""
        print(f"display {r:<10}{d['decision']:<9}{d['reason']}{extra}")

    items2 = [i for i in load_items(cfg2["labels_csv"], cfg2["params_csv"], cfg2["set_name"]) if i.split == "test"]
    assert [i.id for i in items2] == [i.id for i in test], "the two runs must score the same held-out identities"
    pred2 = predict(
        model2,
        DefinitionDataset(items2, cfg2["images_dir"], cfg2["set_name"], cfg2["height"]),
        device,
    )
    m2 = head_metrics(pred2[:, 0], np.array([i.definition - 1 for i in items2]))
    ab = ab_check(metrics, m2)
    print(f"2-head definition: rho {m2['spearman']:.3f} MAE {m2['mae']:.3f}   A/B (tolerance {AB_TOLERANCE}): {ab}")

    # Ship conditions (ADR-053 Amendment 6, user 2026-11-02). (1) adiposity on the SAME front-only samples.
    adip8 = front_only(test_samples, pred, true, "adiposity")
    adip2 = head_metrics(pred2[:, 1], np.array([i.adiposity - 1 for i in items2]))
    cond1 = not_worse(adip8, adip2)
    print(
        f"(1) adiposity front-only n {adip8['n']}: 8-head rho {adip8['spearman']:.3f} MAE {adip8['mae']:.3f}  "
        f"2-head rho {adip2['spearman']:.3f} MAE {adip2['mae']:.3f}  not worse (tolerance {AB_TOLERANCE}): {cond1}"
    )

    # (2a) per sex, every head over its valid samples against the primary gate; front-only adiposity beside it.
    cond2a = True
    for sex, per_head in sex_metrics(test_samples, pred, true, mask).items():
        for h, m in per_head.items():
            cond2a &= gate(m)
            print(f"(2a) sex {sex:<7}{h:<11} n {m['n']:>5}  rho {m['spearman']:.3f}  MAE {m['mae']:.3f}  "
                  f"{'PASS' if gate(m) else 'FAIL'}")  # fmt: skip
        idx = np.array([s.item.sex == sex for s in test_samples])
        sub = [s for s, i in zip(test_samples, idx, strict=False) if i]
        fa = front_only(sub, pred[idx], true[idx], "adiposity")
        print(f"(2a) sex {sex:<7}adiposity front-only n {fa['n']}  rho {fa['spearman']:.3f}  MAE {fa['mae']:.3f}")
    print(f"(2a) every head passes the primary gate for each sex: {bool(cond2a)}")

    # (2b) lighting slice (rig `top`), both models on the same 50 identities, front view for the comparison.
    cond2b = None
    slice_items = [s.item for s in test_samples if s.item.slice_top and s.view == "front"]
    if cfg8["set_name"] == "bump" and slice_items:
        sl = samples(slice_items)
        sl_pred = predict_samples(
            model8,
            DefinitionDataset8(sl, cfg8["images_dir"], "bump", cfg8["height"], rig="top"),
            device,
        )
        sl_true, sl_mask = sample_arrays(sl)
        for h, m in masked_head_metrics(sl_pred, sl_true, sl_mask).items():
            print(f"(2b) lighting slice {h:<11} n {m['n']:>3}  rho {m['spearman']:.3f}  MAE {m['mae']:.3f}")
        items2_sl = [i for i in items2 if i.slice_top]
        sl_pred2 = predict(
            model2,
            DefinitionDataset(items2_sl, cfg2["images_dir"], "bump", cfg2["height"], rig="top"),
            device,
        )
        sl2 = {"definition": head_metrics(sl_pred2[:, 0], np.array([i.definition - 1 for i in items2_sl])),
               "adiposity": head_metrics(sl_pred2[:, 1], np.array([i.adiposity - 1 for i in items2_sl]))}  # fmt: skip
        sl8 = {h: front_only(sl, sl_pred, sl_true, h) for h in sl2}
        cond2b = {h: not_worse(sl8[h], sl2[h]) for h in sl2}
        for h in sl2:
            print(
                f"(2b) slice front-only {h:<10} 8-head rho {sl8[h]['spearman']:.3f} MAE {sl8[h]['mae']:.3f}  "
                f"2-head rho {sl2[h]['spearman']:.3f} MAE {sl2[h]['mae']:.3f}  not worse: {cond2b[h]}"
            )
    print(f"CONDITIONS (Amendment 6): (1) {cond1}  (2a) {bool(cond2a)}  (2b) {cond2b}; any False: report before export")

    ledger.write_text(json.dumps({"checkpoint_sha256": sha8, "two_head_sha256": sha2, "ab": ab,
                                  "display": {r: d["decision"] for r, d in decisions.items()},
                                  "conditions": {"adiposity_front_only": cond1, "per_sex_gate": bool(cond2a),
                                                 "slice_not_worse": cond2b}}))  # fmt: skip


if __name__ == "__main__":
    main()
