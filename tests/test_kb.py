"""Corpus checks (ADR-048). The counts pin the 2026-11-02 snapshot of free-exercise-db that was
checked by hand; a changed dump should fail here, not pass silently."""

import copy
import json
import re

import pytest

from kb import baselines as baselines_mod
from kb.ingest_exercises import load
from physique_schemas.models import REGIONS, Level

MUSCLES = {  # the dataset's own 17 names: `shoulders` is one group, not front/rear
    "abdominals", "abductors", "adductors", "biceps", "calves", "chest", "forearms", "glutes",
    "hamstrings", "lats", "lower back", "middle back", "neck", "quadriceps", "shoulders",
    "traps", "triceps",
}  # fmt: skip


def test_exercise_ids_are_unique_and_counts_match_the_snapshot():
    chunks = load()
    assert len({c["id"] for c in chunks}) == len(chunks) == 876
    assert sum(c["plannable"] for c in chunks) == 584 + 38  # strength + powerlifting


def test_every_chunk_has_the_metadata_retrieval_filters_on():
    for c in load():
        assert c["muscle"], c["id"]
        assert c["level"] in Level.__args__, c["id"]
        assert isinstance(c["equipment"], str) and c["equipment"], c["id"]
        assert c["text"].startswith(c["name"]), c["id"]


def test_muscle_vocabulary_is_the_checked_one():
    seen = {m for c in load() for m in (*c["muscle"], *c["secondary"])}
    assert seen == MUSCLES


def test_no_image_fields_are_kept():
    assert all("images" not in c and ".jpg" not in c["text"] for c in load())


# --- baselines and the region map (ADR-039 item 7) ---------------------------------------
def _kb():
    baselines, region_map = baselines_mod.load()
    return baselines, region_map, baselines_mod.source_ids()


def test_cited_baselines_cover_every_region_with_a_listed_source():
    baselines, region_map, sources = _kb()
    baselines_mod.validate(baselines, region_map, sources, REGIONS)  # raises on any problem
    assert {"pelland-2024-preprint", "remmert-2025-preprint"} <= sources


def test_groups_are_the_datasets_own_names():
    baselines, region_map, _ = _kb()
    mapped = {g for r in region_map["regions"].values() for g in r["groups"]}
    assert mapped <= MUSCLES and baselines["groups"].keys() == mapped


def test_counting_rule_is_primary_one_secondary_half():
    baselines, _, _ = _kb()
    assert baselines["counting"]["primary"] == 1.0 and baselines["counting"]["secondary"] == 0.5


def test_a_baseline_without_a_listed_source_fails_loudly():
    baselines, region_map, sources = _kb()
    broken = copy.deepcopy(baselines)
    broken["groups"]["chest"]["source_id"] = "made-up"
    with pytest.raises(ValueError, match="made-up"):
        baselines_mod.validate(broken, region_map, sources, REGIONS)


def test_a_region_without_a_baseline_fails_loudly():
    baselines, region_map, sources = _kb()
    broken = copy.deepcopy(baselines)
    del broken["groups"]["calves"]
    with pytest.raises(ValueError, match="calves"):
        baselines_mod.validate(broken, region_map, sources, REGIONS)


def test_a_region_missing_from_the_map_fails_loudly():
    baselines, region_map, sources = _kb()
    broken = copy.deepcopy(region_map)
    del broken["regions"]["lats"]
    with pytest.raises(ValueError, match="lats"):
        baselines_mod.validate(baselines, broken, sources, REGIONS)


def test_single_group_regions_map_one_to_one():
    _, region_map, _ = _kb()
    assert set(region_map["one_to_one"]) == {"shoulders", "chest", "lats", "forearm", "calf"}
    for region in region_map["one_to_one"]:
        assert len(region_map["regions"][region]["groups"]) == 1, region


def test_lower_back_and_neck_are_untracked_not_prescribed():
    baselines, region_map, _ = _kb()
    mapped = {g for r in region_map["regions"].values() for g in r["groups"]}
    assert set(region_map["untracked"]) == {"lower back", "neck"} <= MUSCLES
    for group, note in region_map["untracked"].items():
        assert group not in baselines["groups"] and group not in mapped
        assert "not tracked, not prescribed directly" in note.lower()
        assert "does not forbid compound lifts" in note


def test_wording_is_typical_minimum_and_the_cap_is_a_soft_constraint():
    baselines, _, _ = _kb()
    cap = baselines["session_cap"]
    assert baselines["label"] == "typical minimum"
    assert "typical minimum" in baselines["agent_text"]
    assert "your minimum" not in baselines["agent_text"].lower()
    assert cap["kind"] == "soft planner constraint" and cap["value"] == 11
    assert "safety" not in json.dumps(cap).lower()  # never described as a safety limit
    assert "OUR ASSUMPTION" in baselines["counting"]["assumption"]


# --- protein and energy (ADR-048) ---------------------------------------------------------
def _nutrition():
    return json.loads((baselines_mod.KB / "nutrition.json").read_text())


def test_nutrition_numbers_cite_listed_sources_and_carry_the_medical_line():
    n, sources = _nutrition(), baselines_mod.source_ids()
    assert n["protein"]["g_per_kg_day"] == [1.6, 2.2] and n["protein"]["label"] == "typical range"
    assert "body weight is known" in n["protein"]["applies_when"]
    cited = {*n["protein"]["source_ids"], *n["energy"]["source_ids"]}
    cited |= {s for x in n["not_used"] for s in x["source_ids"]}
    assert cited <= sources, cited - sources
    assert "kidney disease" in n["safety_text"] and "doctor or dietitian" in n["safety_text"]
    assert "not medical advice" in n["safety_text"].lower()


def test_nutrition_json_states_unit_scope_planner_rules_and_the_deferred_items():
    n = _nutrition()
    assert n["protein"]["unit"] == "g per kg of total body weight per day"
    assert n["protein"]["also_listed"][0]["g_per_kg_day"] == [1.4, 2.0]
    assert "total body weight" in n["protein"]["agent_text"] and "1.4" in n["protein"]["agent_text"]
    scope = {k: v["verified_scope"] for k, v in n["sources"].items()}
    assert scope["helms-2014"] == scope["iraki-2019"] == "partial"
    for partial in ("helms-2014", "iraki-2019"):
        assert "tbd_after_demo" in n["sources"][partial]  # reading the tail stays in the plan
    assert set(n["sources"]) <= baselines_mod.source_ids()
    assert set(n["planner_rules"]) == {"protein_point_pick", "loss_rate_scaling_by_leanness"}
    assert all("OUR rule" in rule for rule in n["planner_rules"].values())
    bmi = [x for x in n["tbd_after_demo"] if x["item"] == "High-BMI handling"]
    assert len(bmi) == 1 and "OUR rule" in bmi[0]["what"]
    assert "kidney" not in n["protein"]["agent_text"].lower()  # neutral referral, no claims


def test_protein_energy_doc_matches_the_json_and_cites_listed_sources():
    text = (baselines_mod.KB / "docs" / "protein-energy.md").read_text()
    front = re.search(r"^source_ids: \[(.*?)\]$", text, re.M).group(1)
    assert {s.strip() for s in front.split(",")} <= baselines_mod.source_ids()
    phrases = ("1.6–2.2 grams", "10–20% above maintenance", "0.25–0.5%", "0.5–1%", "kidney disease")
    for phrase in phrases:
        assert phrase in text, phrase
    assert "not medical advice" in text.lower()
