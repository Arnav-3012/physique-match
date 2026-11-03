import json

import pandas as pd
import pytest

from ml.src.data.bodym import ROOT
from ml.src.data.regions import LABELLED_REGIONS
from ml.src.data.split import SPLITS_JSON, assert_disjoint, load_split_table, make_split
from physique_schemas.models import REGIONS


def _table(prefix: str, n: int, shared: tuple[str, ...] = ()) -> pd.DataFrame:
    """n subjects with 3 photos each, alternating sex."""
    ids = [f"{prefix}{i}" for i in range(n)] + list(shared)
    rows = [(s, f"{s}-p{k}", "male" if i % 2 else "female") for i, s in enumerate(ids) for k in range(3)]
    return pd.DataFrame(rows, columns=["subject_id", "photo_id", "sex"])


@pytest.fixture
def tables():
    return {
        "train": _table("tr", 100),
        "testA": _table("a", 10, ("x",)),
        "testB": _table("b", 10, ("x",)),
    }


def test_no_subject_in_two_splits(tables):
    split = make_split(tables)
    assert_disjoint(split)
    assert set(split["train"]) | set(split["val"]) == set(tables["train"].subject_id)


def test_subject_in_both_tests_dropped_from_testB(tables):
    split = make_split(tables)
    assert "x" in split["testA"] and "x" not in split["testB"]


def test_split_is_deterministic_and_seeded(tables):
    assert make_split(tables) == make_split(tables)
    assert make_split(tables, seed=1)["val"] != make_split(tables)["val"]


def test_val_is_stratified_by_sex(tables):
    sex = tables["train"].drop_duplicates("subject_id").set_index("subject_id").sex
    val_sex = sex[make_split(tables)["val"]].value_counts()
    assert val_sex["male"] == val_sex["female"]  # 50/50 train -> equal val per sex


def test_overlap_is_detected():
    with pytest.raises(ValueError):
        assert_disjoint({"a": ["s1", "s2"], "b": ["s2"]})


def test_labelled_regions_are_schema_regions():
    assert set(LABELLED_REGIONS) <= set(REGIONS)
    assert set(REGIONS) - set(LABELLED_REGIONS) == {"upper_back", "lats"}


@pytest.mark.skipif(not SPLITS_JSON.exists(), reason="run python -m ml.src.data.split first")
def test_real_splits_json_is_disjoint():
    assert_disjoint(json.loads(SPLITS_JSON.read_text()))


@pytest.mark.skipif(not ROOT.exists(), reason="BodyM not downloaded")
def test_real_train_val_share_no_subject():
    tr, va = load_split_table("train"), load_split_table("val")
    assert not set(tr.subject_id) & set(va.subject_id)
