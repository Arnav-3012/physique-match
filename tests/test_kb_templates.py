import json
import re

import pytest

from kb.baselines import KB, load, source_ids
from kb.templates import check, load_exercises

TEMPLATES = json.loads((KB / "templates.json").read_text())["templates"]


@pytest.mark.parametrize("name", TEMPLATES)
def test_template_ids_exist_and_meet_every_floor(name):
    baselines, _ = load()
    template = TEMPLATES[name]
    exercises = load_exercises()
    unknown = {i for day in template["days"].values() for i in day} - exercises.keys()
    assert not unknown
    week = check(template, exercises, baselines)
    assert set(baselines["groups"]) <= week.keys()


def test_floor_failure_is_loud():
    baselines, _ = load()
    thin = {"days": {"day_1": {"Barbell_Bench_Press_-_Medium_Grip": 3}}}
    with pytest.raises(ValueError, match="floor"):
        check(thin, load_exercises(), baselines)


def test_session_cap_failure_is_loud():
    baselines, _ = load()
    heavy = {"days": {"day_1": {"Barbell_Bench_Press_-_Medium_Grip": 12}}}
    with pytest.raises(ValueError, match="session cap"):
        check(heavy, load_exercises(), baselines)


def test_doc_source_ids_are_in_sources_md():
    known = source_ids()
    for doc in (KB / "docs").glob("*.md"):
        front = re.search(r"^source_ids: \[(.*?)\]", doc.read_text(), re.M)
        assert front, f"{doc.name} has no source_ids line"
        cited = {s.strip() for s in front.group(1).split(",") if s.strip()}
        assert cited <= known, f"{doc.name}: {cited - known}"
