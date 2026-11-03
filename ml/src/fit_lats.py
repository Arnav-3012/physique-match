"""Lats ridge on ANSUR II with the additive sex term: ADR-036 + ADR-037 Amendment 2 (step 3 only).

y = interscyei / stature. Inputs chest / stature, biacromial / stature, waist / stature, is_male.
5 outer folds stratified by sex (one row per subject, so a row split is a subject split); alpha from
an inner 5-fold CV on noisy inputs; train rows get 20 noisy copies; held-out rows are scored on
noisy inputs (50 draws). Models: `chest` ([chest, is_male]) and `full` (all four). All numbers and
the gate are fixed in the ADRs; no tuning after seeing the result.
No pooled no-sex ridge: dropped by ADR-037 Amendment 3 (`lats` needs both sexes set and equal).
Noise-free columns and the shoulder coefficient are diagnostics only; they never gate.
Run (after ansur_check): uv run python -m ml.src.fit_lats
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from ml.src.data.ansur import ROOT, load

SEED, N_FOLDS, N_COPIES, N_DRAWS, N_BOOT = 0, 5, 20, 50, 2000
ALPHAS = np.logspace(-6, 2, 17)
STRESS = 1.5
GATE_DR2 = 0.02  # min over sexes of R2_full - R2_chest
TIERS = (0.7, 0.5)  # on min over sexes of R2_full
STOP_R2 = 0.999  # in-sample clean R2 above this means y is a linear function of the inputs
NAMES = ["chest", "shoulders", "waist", "is_male"]
MODELS = {"chest": [0, 3], "full": [0, 1, 2, 3]}  # column indices into NAMES
OWN_SEX_COLS = [0, 1, 2]
SCENARIOS = ("noisy", "front", "stress")  # sigma_full, sigma_front, 1.5 x sigma_full
GAP_JSON = Path(__file__).resolve().parents[2] / "services/api/app/analyzer/gap_regressor.json"
OUT = GAP_JSON.with_name("lats_coef.json")


def rng_for(*key: int) -> np.random.Generator:
    return np.random.default_rng([SEED, *key])


def prepare() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = load()
    x = np.column_stack([df.chest / df.stature, df.biacromial / df.stature,
                         df.waist / df.stature, df.is_male])  # fmt: skip
    return x, (df.interscye / df.stature).to_numpy(), df.is_male.to_numpy().astype(int)


def noise_sigmas() -> dict[str, np.ndarray]:
    """Per-column noise std (B's val residual std, ADR-036); is_male gets none."""
    cal = json.loads(GAP_JSON.read_text())["calibration"]
    return {k: np.array([cal[r][f"sigma_{k}"] for r in NAMES[:3]] + [0.0])
            for k in ("full", "front")}  # fmt: skip


def assign_folds(male: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    fold = np.empty(len(male), int)
    for s in (0, 1):
        idx = np.flatnonzero(male == s)
        fold[rng.permutation(idx)] = np.arange(len(idx)) % k
    return fold


def ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Raw-unit coefficients and intercept; features standardised with x's own statistics."""
    mu, sd = x.mean(0), x.std(0)
    z = (x - mu) / sd
    w = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - y.mean())) / sd
    return w, y.mean() - w @ mu


def noisy_copies(x, y, sigma, rng, copies=N_COPIES):
    z = rng.standard_normal((copies, *x.shape))
    return (x + z * sigma).reshape(-1, x.shape[1]), np.tile(y, copies)


def fit(x, y, male, cols, sigma, rng):
    """Alpha from an inner CV (noisy copies of the validation rows), then refit on all rows."""
    inner = assign_folds(male, N_FOLDS, rng)
    mse = np.zeros(len(ALPHAS))
    for k in range(N_FOLDS):
        xt, yt = noisy_copies(x[inner != k], y[inner != k], sigma, rng)
        xv, yv = noisy_copies(x[inner == k], y[inner == k], sigma, rng, copies=5)
        for i, a in enumerate(ALPHAS):
            w, b = ridge(xt[:, cols], yt, a)
            mse[i] += np.mean((xv[:, cols] @ w + b - yv) ** 2)
    alpha = ALPHAS[mse.argmin()]
    xt, yt = noisy_copies(x, y, sigma, rng)
    return (cols, *ridge(xt[:, cols], yt, alpha)), float(alpha)


def refit(data, rows, cols, sigma, *seed_key):
    x, y, male = data
    return fit(x[rows], y[rows], male[rows], cols, sigma, rng_for(*seed_key))


def residuals(model, x, y, sigma, z) -> np.ndarray:
    """(draws, n) prediction minus clean target; z = standard-normal draws, shape (draws, n, 4)."""
    cols, w, b = model
    return (x + z * sigma)[..., cols] @ w + b - y


def clean_resid(model, x, y) -> np.ndarray:
    return residuals(model, x, y, np.zeros(4), np.zeros((1, *x.shape)))[0]


def cross_validate(data, sig):
    x, y, male = data
    outer = assign_folds(male, N_FOLDS, rng_for(0))
    scenarios = {"noisy": sig["full"], "front": sig["front"], "stress": STRESS * sig["full"]}
    n = len(y)
    res = {m: {s: np.zeros((N_DRAWS, n)) for s in scenarios} for m in MODELS}
    # noise-free diagnostics (Amendment 3): noise-fitted ("shipped") and clean-fitted ("ceiling")
    clean = {f"{kind}_{m}": np.zeros(n) for kind in ("shipped", "ceiling") for m in MODELS}
    own = np.zeros((N_DRAWS, n))  # separate fit per sex, diagnostic only
    sst, sst_pooled, alphas = np.zeros(n), np.zeros(n), {m: [] for m in MODELS}
    for k in range(N_FOLDS):
        tr, te = np.flatnonzero(outer != k), np.flatnonzero(outer == k)
        z = rng_for(k, 2).standard_normal((N_DRAWS, len(te), 4))
        fits = {}
        for m, cols in MODELS.items():  # same seed per model: same inner folds and training noise
            fits[m], a = refit(data, tr, cols, sig["full"], k, 1)
            alphas[m].append(a)
            for s, sigma in scenarios.items():
                res[m][s][:, te] = residuals(fits[m], x[te], y[te], sigma, z)
        for m, cols in MODELS.items():
            ceiling, _ = refit(data, tr, cols, np.zeros(4), k, 1)
            clean[f"shipped_{m}"][te] = clean_resid(fits[m], x[te], y[te])
            clean[f"ceiling_{m}"][te] = clean_resid(ceiling, x[te], y[te])
        for s in (0, 1):
            tr_s, te_s = tr[male[tr] == s], te[male[te] == s]
            sst[te_s] = (y[te_s] - y[tr_s].mean()) ** 2  # reference = training mean of the same sex
            m_s, _ = refit(data, tr_s, OWN_SEX_COLS, sig["full"], k, 3)
            own[:, te_s] = residuals(m_s, x[te_s], y[te_s], sig["full"], z[:, male[te] == s])
        sst_pooled[te] = (y[te] - y[tr].mean()) ** 2
    return res, clean, own, sst, sst_pooled, alphas


def r2(sq_err_row: np.ndarray, sst_row: np.ndarray, mask: np.ndarray) -> float:
    return float(1 - sq_err_row[mask].sum() / sst_row[mask].sum())


def bootstrap_min_dr2(diff, sst, male, rng) -> np.ndarray:
    """Per resample (subjects, stratified by sex, reference means fixed): min over sexes of dR2."""
    stat = np.empty((N_BOOT, 2))
    for s in (0, 1):
        idx = np.flatnonzero(male == s)
        pick = idx[rng.integers(0, len(idx), (N_BOOT, len(idx)))]
        stat[:, s] = diff[pick].sum(1) / sst[pick].sum(1)
    return stat.min(1)


def tier(gate_passed: bool, r2_min: float) -> str:
    if not gate_passed or r2_min < TIERS[1]:
        return "unranked"
    return "ranked" if r2_min >= TIERS[0] else "ranked_low_confidence"


def sigma_prior(y, male) -> float:
    """Std of the true gap over ordered same-sex pairs (the only pairs this route serves)."""
    groups = [y[male == s] for s in (0, 1)]
    sum_sq = sum(2 * len(g) * ((g - g.mean()) ** 2).sum() for g in groups)  # sum (yi - yj)^2, i != j
    return float(np.sqrt(sum_sq / sum(len(g) * (len(g) - 1) for g in groups)))


def per_sex_table(res, clean, own, sst, sexes) -> dict[str, dict[str, float]]:
    sq = {(m, s): (r**2).mean(0) for m in MODELS for s, r in res[m].items()}
    table = {}
    for name, mask in sexes.items():
        t = {f"r2_{m}_{s}": r2(sq[m, s], sst, mask) for m in MODELS for s in SCENARIOS}
        t["dr2_noisy"] = t["r2_full_noisy"] - t["r2_chest_noisy"]
        for kind in ("shipped", "ceiling"):
            c = {m: r2(clean[f"{kind}_{m}"] ** 2, sst, mask) for m in MODELS}
            t[f"r2_clean_{kind}"], t[f"dr2_clean_{kind}"] = c["full"], c["full"] - c["chest"]
        t["r2_own_sex_fit_noisy"] = r2((own**2).mean(0), sst, mask)
        t["mae_full_noisy"] = float(np.abs(res["full"]["noisy"]).mean(0)[mask].mean())
        t["mae_full_noisy_cm_at_175"] = t["mae_full_noisy"] * 175
        table[name] = t
    return table


def slopes_raw(data, sig, sexes, pooled_w) -> dict[str, dict[str, float]]:
    """Raw-unit slopes: pooled additive fit vs a separate fit per sex (diagnostic)."""
    slopes = {"pooled_additive": dict(zip(NAMES, map(float, pooled_w[:3]), strict=False))}
    for name, mask in sexes.items():
        (_, w, _), _ = refit(data, mask, OWN_SEX_COLS, sig["full"], 99, 3)
        slopes[f"{name}_only"] = dict(zip(NAMES, map(float, w), strict=False))
    return slopes


def print_report(table, pooled, slopes, alpha, alphas, summary, reliability, diag) -> None:
    head = ["R2 chest", "R2 full", "dR2", "clean", "ceiling", "dR2 cln", "dR2 ceil",
            "own-sex", "MAE", "cm@175"]  # fmt: skip
    print(f"{'':8}" + "".join(f"{h:>9}" for h in head))
    for name, t in table.items():
        row = [t["r2_chest_noisy"], t["r2_full_noisy"], t["dr2_noisy"], t["r2_clean_shipped"],
               t["r2_clean_ceiling"], t["dr2_clean_shipped"], t["dr2_clean_ceiling"],
               t["r2_own_sex_fit_noisy"], t["mae_full_noisy"],
               t["mae_full_noisy_cm_at_175"]]  # fmt: skip
        print(f"{name:8}" + "".join(f"{v:9.3f}" for v in row))
    print(
        "diagnostic (not a gate): shoulder coef noise-fit "
        f"{diag['shoulder_coef_noise_fit']:.3f}, clean-fit {diag['shoulder_coef_clean_fit']:.3f}"
    )
    print(f"pooled R2 (secondary): chest {pooled['chest']:.3f}, full {pooled['full']:.3f}")
    for name, t in table.items():
        gain = t["r2_own_sex_fit_noisy"] - t["r2_full_noisy"]
        print(f"{name}: full R2 at sigma_front {t['r2_full_front']:.3f}, at 1.5 x sigma_full "
              f"{t['r2_full_stress']:.3f}; own-sex fit minus pooled R2 {gain:+.3f}"
              + ("  FLAG (>= 0.02)" if gain >= 0.02 else ""))  # fmt: skip
    print("raw slopes:", json.dumps(slopes))
    print(f"alpha (all rows) {alpha:.2g}, outer alphas (full) {alphas['full']}")
    s = summary
    print("--- summary ---")
    print(
        f"1. dR2_min {s['dR2_min']:.3f} (gate >= {GATE_DR2}), bootstrap 95% CI of the min "
        f"[{s['dR2_min_ci95'][0]:.3f}, {s['dR2_min_ci95'][1]:.3f}]"
    )
    print(f"2. gate: {'PASS' if s['gate_passed'] else 'FAIL'}")
    print(f"3. R2_min (worse sex, noisy inputs) {s['R2_min']:.3f}")
    print(f"4. verdict: {s['verdict']}")
    print(
        f"5. sigma_prior {reliability['sigma_prior']:.4f}, residual std full "
        f"{reliability['resid_std_full']:.4f}, front {reliability['resid_std_front']:.4f}"
    )


def main() -> None:
    data = prepare()
    x, y, male = data
    sig = noise_sigmas()
    res, clean, own, sst, sst_pooled, alphas = cross_validate(data, sig)
    sexes = {"male": male == 1, "female": male == 0}
    table = per_sex_table(res, clean, own, sst, sexes)
    pooled = {m: r2((res[m]["noisy"] ** 2).mean(0), sst_pooled, male >= 0) for m in MODELS}

    r2_min = min(t["r2_full_noisy"] for t in table.values())
    dr2_min = min(t["dr2_noisy"] for t in table.values())
    diff = ((res["chest"]["noisy"] ** 2) - (res["full"]["noisy"] ** 2)).mean(0)
    boots = bootstrap_min_dr2(diff, sst, male, rng_for(0))
    lo, hi = (float(v) for v in np.percentile(boots, [2.5, 97.5]))
    passed = bool(dr2_min >= GATE_DR2 and lo > 0)
    summary = {
        "verdict": tier(passed, r2_min), "gate_passed": passed,
        "R2_min": r2_min, "dR2_min": dr2_min, "dR2_min_ci95": [lo, hi],
        "rule": f"dR2_min >= {GATE_DR2} and CI lower bound > 0; tiers on R2_min {TIERS}",
    }  # fmt: skip

    final, alpha = fit(x, y, male, MODELS["full"], sig["full"], rng_for(99, 1))
    ceiling, _ = fit(x, y, male, MODELS["full"], np.zeros(4), rng_for(99, 1))
    in_sample = float(1 - np.mean(clean_resid(ceiling, x, y) ** 2) / y.var())
    if in_sample > STOP_R2:
        raise SystemExit(f"stop rule: y is a linear function of the inputs (R2 {in_sample:.5f})")
    slopes = slopes_raw(data, sig, sexes, final[1])
    diag = {"shoulder_coef_noise_fit": float(final[1][1]),
            "shoulder_coef_clean_fit": float(ceiling[1][1])}  # fmt: skip
    reliability = {
        "resid_std_full": float(res["full"]["noisy"].std()),
        "resid_std_front": float(res["full"]["front"].std()),
        "sigma_prior": sigma_prior(y, male),
    }
    _, w, b = final
    out = {"additive_sex": {
        **summary,
        "target": "interscyei / stature", "shoulder_column": "biacromialbreadth",
        "inputs": NAMES, "intercept": float(b),
        "coef": dict(zip(NAMES, map(float, w), strict=True)),
        "alpha": alpha, "outer_alphas": alphas,
        "noise": {"sigma_full": sig["full"][:3].tolist(), "sigma_front": sig["front"][:3].tolist()},
        "reliability": reliability, "per_sex": table, "pooled_r2": pooled,
        "slopes_raw": slopes, "in_sample_clean_r2": in_sample, "diagnostics": diag,
        "data": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                 for f in sorted(ROOT.glob("*.csv"))},
        "seed": SEED, "n": {"male": int(male.sum()), "female": int((1 - male).sum())},
    }}  # fmt: skip
    OUT.write_text(json.dumps(out, indent=1))
    print_report(table, pooled, slopes, alpha, alphas, summary, reliability, diag)
    print("->", OUT)


if __name__ == "__main__":
    main()
