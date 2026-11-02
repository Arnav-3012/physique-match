"""ANSUR II public files (ADR-036): one row per participant, millimetres, `is_male` from the file.
Header casing differs between the two files (`subjectid` / `SubjectId`), so columns are lower-cased.
Files: ml/data/ansur/ANSUR_II_{MALE,FEMALE}_Public.csv (gitignored, not redistributed)."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "data" / "ansur"
COLUMNS = {
    "subjectid": "subject_id", "stature": "stature", "chestcircumference": "chest",
    "biacromialbreadth": "biacromial", "bideltoidbreadth": "bideltoid",
    "waistcircumference": "waist", "interscyei": "interscye",
}  # fmt: skip


def load(root: Path = ROOT) -> pd.DataFrame:
    """Both sexes stacked. Columns: subject_id, stature, chest, biacromial, bideltoid, waist,
    interscye (all mm) and is_male."""
    parts = []
    for sex in ("male", "female"):
        df = pd.read_csv(root / f"ANSUR_II_{sex.upper()}_Public.csv", encoding="latin-1")
        df.columns = df.columns.str.lower()
        if not df.gender.str.lower().eq(sex).all():
            raise ValueError(f"{sex} file has a Gender value that is not {sex}")
        parts.append(df[list(COLUMNS)].rename(columns=COLUMNS).assign(is_male=float(sex == "male")))
    out = pd.concat(parts, ignore_index=True)
    if not out.subject_id.is_unique:
        raise ValueError("subjectid is not unique across the two files")
    if out.isna().any().any():
        raise ValueError(f"missing values:\n{out.isna().sum()}")
    return out
