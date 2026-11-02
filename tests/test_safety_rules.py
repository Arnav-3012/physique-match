"""app/safety/rules.py is an independent re-check: it must not import the body-composition code, and every
limit it restates is pinned to that code here so drift fails loudly."""

import ast
from pathlib import Path

import pytest

from app import bodycomp
from app.safety import rules
from app.safety.rules import SafetyBreach, check_plan_safety
from physique_schemas.models import BodyComp, BodyFatRange, EnergyTarget, UserProfile

PROFILE = dict(
    sex="male", height_cm=175, weight_kg=75, age=30, activity="moderate", level="beginner",
    goal="cut", days_per_week=3,
)  # fmt: skip


def profile(**kw) -> UserProfile:
    return UserProfile(**{**PROFILE, **kw})


def target(goal="maintain", low=2300, high=2450) -> EnergyTarget:
    return EnergyTarget(goal=goal, kcal_low=low, kcal_high=high, protein_g_low=120, protein_g_high=165)


def comp(energy, **kw) -> BodyComp:
    base = dict(
        body_fat=BodyFatRange(low=15, high=20, method="test", confidence="low"),
        rmr_kcal=1700, tdee_kcal=2600, energy=energy,
    )  # fmt: skip
    # model_construct: BodyComp's own validator already blocks most breaches; the rules must still catch them
    return BodyComp.model_construct(**{**base, **kw})


def test_rules_do_not_import_the_body_composition_code():
    tree = ast.parse(Path(rules.__file__).read_text())
    modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
    }
    assert not {m for m in modules if m.startswith(("app.bodycomp", "app.nutrition"))}


def test_restated_limits_equal_the_body_composition_code():
    assert rules.MAX_LOSS_PER_WEEK == bodycomp.MAX_LOSS_PER_WEEK
    assert rules.KCAL_PER_KG == bodycomp.KCAL_PER_KG
    assert rules.MIN_BMI == bodycomp.MIN_BMI
    assert rules.BF_FLOOR == bodycomp.BF_FLOOR
    assert rules.ADULT_AGE == bodycomp.MIN_AGE
    assert round(1 - bodycomp.ENERGY["cut"][0], 9) == rules.MAX_DEFICIT
    assert round(bodycomp.ENERGY["bulk"][1] - 1, 9) == rules.MAX_SURPLUS


def test_a_normal_body_comp_passes():
    p = profile()
    check_plan_safety(p, bodycomp.body_comp(p.model_copy(update={"tape_waist_cm": 85.0})))
    check_plan_safety(p, None)


@pytest.mark.parametrize(
    ("p", "c", "why"),
    [
        (profile(ed_history=True), comp([target()]), "calorie numbers"),
        (profile(sex="not_specified"), comp([target()]), "calorie numbers"),
        (profile(), comp([target("cut", 1900, 2100)]), "deficit cap"),
        (profile(), comp([target("cut", 1400, 1500)], rmr_kcal=1700, tdee_kcal=1750), "deficit cap"),
        (profile(), comp([target("bulk", 2700, 3000)]), "surplus cap"),
        # 600 kcal/day below 3000 is 0.55 kg/week, over 1% of 50 kg, though exactly the 20% deficit
        (profile(weight_kg=50), comp([target("cut", 2400, 2500)], tdee_kcal=3000), "over the cap"),
        (profile(goal="bulk"), comp([target("cut", 2080, 2200)]), "own choice"),
        (
            profile(weight_kg=50),  # BMI 16.3: a deficit target is below the floor
            comp([target("cut", 2300, 2400)], tdee_kcal=2600, rmr_kcal=1500),
            "floor",
        ),
    ],
)
def test_each_breach_is_caught(p, c, why):
    with pytest.raises(SafetyBreach, match=why):
        check_plan_safety(p, c)
