import copy
from itertools import product

import pytest
from pydantic import ValidationError

from app import nutrition
from app.bodycomp import (
    ENERGY,
    MAX_LOSS_PER_WEEK,
    NOTE_ED,
    NOTE_MINOR,
    NOTE_NO_SEX,
    body_comp,
    body_fat,
    rfm,
    rmr,
)
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


def test_unset_sex_gives_no_body_fat_and_no_calories_and_no_error():
    """ADR-037 Amendment 4: RFM and Mifflin both carry a sex term, so neither is guessed."""
    bc = body_comp(profile(sex="not_specified"))
    assert bc.body_fat is None and bc.energy == []
    assert bc.rmr_kcal is None and bc.tdee_kcal is None and bc.note == NOTE_NO_SEX


def test_unset_sex_needs_no_waist_and_keeps_the_eating_disorder_note():
    bc = body_comp(profile(sex="not_specified", tape_waist_cm=None, ed_history=True))
    assert bc.body_fat is None and bc.note == f"{NOTE_ED} {NOTE_NO_SEX}"


def test_rmr_never_guesses_a_sex_term():
    with pytest.raises(KeyError):
        rmr(profile(sex="not_specified"))


@pytest.mark.parametrize("goal", ["cut", "recomp", "bulk"])
def test_body_fat_never_changes_the_mode_only_the_deficit_opt_out(goal):
    """The mode is the user's `goal` field (ADR-037 Amendment 4). Across lean to heavy waists the
    targets offered are maintain and that goal, never another mode, and the profile is untouched."""
    for waist in (55, 66, 75, 90, 110, 140):
        p = profile(goal=goal, weight_kg=85, height_cm=195, tape_waist_cm=waist)
        offered = {t.goal for t in body_comp(p).energy}
        assert offered <= {"maintain", goal} and "maintain" in offered
        assert p.goal == goal


def test_no_waist_at_all_is_an_error_not_a_guess():
    with pytest.raises(ValueError, match="no waist"):
        body_comp(profile(tape_waist_cm=None))
    with pytest.raises(ValueError, match="no waist"):
        body_comp(profile(tape_waist_cm=None), waist_g=float("nan"))


# --- protein, calories and the cited ranges (ADR-048 Amendment 2) ----------------------------
def test_protein_shows_when_sex_is_unset_but_calories_are_not_measured():
    bc = body_comp(profile(sex="not_specified", weight_kg=80))
    assert bc.energy_measured is False and bc.energy == []
    assert bc.rmr_kcal is None and bc.tdee_kcal is None and bc.body_fat is None
    assert (bc.protein.g_per_kg_low, bc.protein.g_per_kg_high) == (1.6, 2.2)
    assert (bc.protein.g_low, bc.protein.g_high) == (128, 176)
    assert bc.protein.unit == "g per kg of total body weight per day"
    assert "1.4" in bc.protein.also_listed and "2.0" in bc.protein.also_listed


def test_protein_is_the_whole_cited_range_and_matches_every_energy_target():
    bc = body_comp(profile(weight_kg=70))
    low, high = round(1.6 * 70), round(2.2 * 70)
    assert bc.energy_measured is True and (bc.protein.g_low, bc.protein.g_high) == (low, high)
    assert {(t.protein_g_low, t.protein_g_high) for t in bc.energy} == {(low, high)}


def test_protein_and_calories_are_withheld_together_for_an_eating_disorder_history():
    bc = body_comp(profile(ed_history=True))
    assert bc.protein is None and bc.energy_measured is False and bc.energy == []


def test_every_calorie_band_is_inside_the_cited_range_or_a_declared_override():
    nutrition.check_energy(ENERGY, MAX_LOSS_PER_WEEK)


def test_a_band_without_a_cited_range_or_declared_override_fails_loudly():
    with pytest.raises(ValueError, match="bulk"):
        nutrition.check_energy({**ENERGY, "bulk": (1.15, 1.25)}, MAX_LOSS_PER_WEEK)
    with pytest.raises(ValueError, match="crash"):
        nutrition.check_energy({**ENERGY, "crash": (0.6, 0.7)}, MAX_LOSS_PER_WEEK)
    with pytest.raises(ValueError, match="loss cap"):
        nutrition.check_energy(ENERGY, 0.02)


def test_app_floors_override_the_cited_ranges(monkeypatch):
    p = profile(weight_kg=60, height_cm=165, goal="cut", tape_waist_cm=80)
    before = body_comp(p)
    wide = copy.deepcopy(nutrition.NUTRITION)
    wide["energy"]["loss_pct_bodyweight_per_week"] = [0.5, 3.0]
    wide["energy"]["gain_surplus_pct_above_maintenance"] = [10, 50]
    monkeypatch.setattr(nutrition, "NUTRITION", wide)
    after = body_comp(p)
    assert after == before  # calories come from the app's own bands and cap, not the cited file
    assert "cut" in [t.goal for t in after.energy]
    for t in after.energy:
        assert t.kcal_low >= after.rmr_kcal - 1
        assert after.tdee_kcal - t.kcal_low <= MAX_LOSS_PER_WEEK * 60 * 7700 / 7 + 1


def test_under_18_gets_no_targets_and_a_short_message():
    with pytest.raises(ValidationError):
        profile(age=17)  # the schema's 18+ gate
    bc = body_comp(profile().model_copy(update={"age": 17}))  # body_comp checks it again
    assert bc.note == NOTE_MINOR and len(NOTE_MINOR) < 160
    assert bc.protein is None and bc.energy == [] and bc.body_fat is None
    assert bc.energy_measured is False and bc.rmr_kcal is None


def test_protein_and_loss_cap_do_not_scale_with_leanness():
    """No point is picked inside the protein range and the loss cap does not move with body fat;
    only the deficit opt-out depends on it (kb/nutrition.json planner_rules, user 2026-11-02)."""
    outs = [
        body_comp(profile(weight_kg=85, height_cm=175, goal="cut", tape_waist_cm=w))
        for w in (75, 85, 95, 110)
    ]
    assert all([t.goal for t in o.energy] == ["maintain", "cut"] for o in outs)
    assert all(o.energy == outs[0].energy and o.protein == outs[0].protein for o in outs)
