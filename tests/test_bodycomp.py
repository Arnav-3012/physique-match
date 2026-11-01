from itertools import product

import pytest

from app.bodycomp import body_comp, body_fat, rfm, rmr
from physique_schemas.models import UserProfile


def profile(**kw) -> UserProfile:
    base = dict(height_cm=175, weight_kg=75, age=30, sex="male", activity="moderate",
                level="beginner", goal="cut", days_per_week=4, tape_waist_cm=85)  # fmt: skip
    return UserProfile(**(base | kw))


def test_rfm_known_values():
    assert rfm(180, 90, "male") == pytest.approx(24.0)
    assert rfm(180, 90, "female") == pytest.approx(36.0)


def test_mifflin_known_values():
    assert rmr(profile(weight_kg=80, height_cm=180, age=30)) == pytest.approx(1780)
    female = profile(sex="female", weight_kg=60, height_cm=165, age=30)
    assert rmr(female) == pytest.approx(1320.25)


def test_body_fat_is_a_range_and_silhouette_is_wider_than_tape():
    p = profile()
    tape, sil = body_fat(p, 90, 1.0, "tape"), body_fat(p, 90, 4.05, "silhouette")
    assert tape.low < tape.high and sil.low < sil.high
    assert sil.high - sil.low > tape.high - tape.low
    assert sil.confidence == "low"


def test_every_combination_respects_the_safety_floors():
    """BodyComp's own validator re-checks floor and cap; here we also check the loss rate."""
    for sex, activity, goal, weight, height, waist in product(
        ("male", "female"), ("sedentary", "moderate", "very_active"), ("cut", "recomp", "bulk"),
        (45, 60, 80, 120), (155, 175, 195), (60, 90, 120),
    ):  # fmt: skip
        p = profile(sex=sex, activity=activity, goal=goal, weight_kg=weight,
                    height_cm=height, tape_waist_cm=waist)  # fmt: skip
        bc = body_comp(p)
        assert 3 <= bc.body_fat.low < bc.body_fat.high <= 60
        for t in bc.energy:
            assert t.kcal_low >= bc.rmr_kcal - 1  # never below RMR (rounding slack)
            assert t.kcal_high <= 1.1 * bc.tdee_kcal + 1  # surplus <= 10%
            assert bc.tdee_kcal - t.kcal_low <= 0.01 * weight * 7700 / 7 + 1  # <= 1% bw / week


def test_eating_disorder_history_removes_every_calorie_number():
    bc = body_comp(profile(ed_history=True))
    assert bc.energy == [] and bc.rmr_kcal is None and bc.tdee_kcal is None
    assert bc.note and bc.body_fat.low < bc.body_fat.high


def test_underweight_gets_maintenance_only():
    bc = body_comp(profile(weight_kg=50, height_cm=180, goal="cut", tape_waist_cm=70))
    assert [t.goal for t in bc.energy] == ["maintain"] and bc.note


def test_lean_male_gets_no_deficit_even_at_a_healthy_bmi():
    bc = body_comp(profile(weight_kg=85, height_cm=195, goal="cut", tape_waist_cm=66))
    assert "cut" not in [t.goal for t in bc.energy] and bc.note


def test_bulk_is_still_offered_to_a_lean_user():
    bc = body_comp(profile(weight_kg=50, height_cm=180, goal="bulk", tape_waist_cm=70))
    assert [t.goal for t in bc.energy] == ["maintain", "bulk"]


def test_normal_cut_gives_maintain_and_cut():
    assert [t.goal for t in body_comp(profile()).energy] == ["maintain", "cut"]


def test_silhouette_waist_is_used_without_a_tape_and_marked():
    bc = body_comp(profile(tape_waist_cm=None), waist_g=0.5)
    assert "silhouette" in bc.body_fat.method and bc.body_fat.confidence == "low"


def test_no_waist_at_all_is_an_error_not_a_guess():
    with pytest.raises(ValueError, match="no waist"):
        body_comp(profile(tape_waist_cm=None))
    with pytest.raises(ValueError, match="no waist"):
        body_comp(profile(tape_waist_cm=None), waist_g=float("nan"))
