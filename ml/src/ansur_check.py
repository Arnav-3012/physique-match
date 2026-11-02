"""ADR-036 shoulder-column rule, run before any ridge is fitted. Inputs only, never the lats target.

Per sex: mean of BodyM `col / height_cm` (train split, one row per subject) next to ANSUR
`col / stature`, for shoulders (both candidates), chest and waist. The shoulder column switches
from biacromial to bideltoid only if BodyM's mean is closer to bideltoid's for both sexes.
Chest and waist have one ANSUR candidate each: reported, not acted on.
Run: uv run python -m ml.src.ansur_check
"""

from ml.src.data.ansur import load
from ml.src.data.split import load_split_table

PAIRS = {"shoulder_breadth": ("biacromial", "bideltoid"), "chest": ("chest",), "waist": ("waist",)}


def main() -> None:
    ansur = load()
    ansur["sex"] = ansur.is_male.map({1.0: "male", 0.0: "female"})
    bodym = load_split_table("train").drop_duplicates("subject_id")
    closer_to_bideltoid = []
    print(
        f"{'sex':7}{'BodyM column':18}{'BodyM mean':>11}  ANSUR candidates (mean of col / stature)"
    )
    for sex in ("male", "female"):
        a, b = ansur[ansur.sex == sex], bodym[bodym.sex == sex]
        print(f"{sex}: ANSUR n={len(a)}, BodyM n={len(b)}")
        for col, candidates in PAIRS.items():
            bm = (b[col] / b.height_cm).mean()
            ans = {c: (a[c] / a.stature).mean() for c in candidates}
            print(f"{'':7}{col:18}{bm:11.4f}  " + "  ".join(f"{c} {v:.4f}" for c, v in ans.items()))
            if col == "shoulder_breadth":
                closer_to_bideltoid.append(abs(bm - ans["bideltoid"]) < abs(bm - ans["biacromial"]))
    choice = "bideltoid" if all(closer_to_bideltoid) else "biacromial"
    print(f"closer to bideltoid per sex (male, female): {closer_to_bideltoid} -> column: {choice}")


if __name__ == "__main__":
    main()
