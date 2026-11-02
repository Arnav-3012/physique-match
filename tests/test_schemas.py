import pytest
from pydantic import ValidationError

from physique_schemas import (
    BodyComp,
    BodyFatRange,
    DefinitionScores,
    EnergyTarget,
    GapProfile,
    GoalProfile,
    Scan,
    UserProfile,
)
from physique_schemas.models import REGIONS, AnalyzeRequest


def view():
    kp = {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0}
    return {"mask_png_b64": "AAAA", "keypoints": [kp] * 33}


def gap_regions():
    return {r: {"delta": 0.02, "confidence": 0.8} for r in REGIONS}


def bf():
    return {"low": 18, "high": 23, "method": "RFM", "confidence": "medium"}


# --- Scan ---
def test_scan_needs_a_front_view_only():
    Scan(views={"front": view()})  # a goal photo may have nothing else (ADR-025)
    Scan(views={"front": view(), "left": view()})
    with pytest.raises(ValidationError, match="missing required views"):
        Scan(views={"left": view()})


def test_goal_may_be_front_only_but_the_current_scan_needs_a_side_view():
    front, both = {"views": {"front": view()}}, {"views": {"front": view(), "left": view()}}
    right = {"views": {"front": view(), "right": view()}}  # a right profile is mirrored later
    AnalyzeRequest(goal=front, current=both, profile=profile())
    AnalyzeRequest(goal=front, current=right, profile=profile())
    with pytest.raises(ValidationError, match="current scan needs a side view"):
        AnalyzeRequest(goal=both, current=front, profile=profile())


def test_scan_rejects_wrong_keypoint_count():
    bad = view() | {"keypoints": view()["keypoints"][:5]}
    with pytest.raises(ValidationError):
        Scan(views={"front": bad, "left": view()})


def test_scan_rejects_unexpected_image_field():
    with pytest.raises(ValidationError):
        Scan(views={"front": view(), "left": view()}, rgb_image="xxxx")


def test_scan_rejects_oversized_payload():
    big = view() | {"mask_png_b64": "A" * 200_000}  # 4 x 200k > 700k total cap
    with pytest.raises(ValidationError, match="512 KB"):
        Scan(views={v: big for v in ("front", "left", "back", "right")})


# --- GapProfile ---
def test_gap_profile_needs_all_ten_regions():
    GapProfile(analyzer="geometric", regions=gap_regions())
    partial = dict(list(gap_regions().items())[:9])
    with pytest.raises(ValidationError, match="missing regions"):
        GapProfile(analyzer="siamese_subtract", regions=partial)


# --- BodyComp ---
def test_body_fat_is_a_range_not_a_point():
    BodyFatRange(**bf())
    with pytest.raises(ValidationError, match="range"):
        BodyFatRange(**(bf() | {"low": 20, "high": 20}))


def energy(kcal_low, kcal_high):
    return EnergyTarget(
        goal="cut", kcal_low=kcal_low, kcal_high=kcal_high, protein_g_low=140, protein_g_high=170
    )


def test_energy_within_floor_and_cap_is_accepted():
    # TDEE 2500 -> deficit floor 2000 (20%), RMR 1700 is lower so 20% rule binds
    BodyComp(body_fat=bf(), rmr_kcal=1700, tdee_kcal=2500, energy=[energy(2000, 2200)])


def test_cut_below_20_percent_deficit_is_rejected():
    with pytest.raises(ValidationError, match="safety"):
        BodyComp(body_fat=bf(), rmr_kcal=1700, tdee_kcal=2500, energy=[energy(1800, 2000)])


def test_cut_below_rmr_is_rejected():
    with pytest.raises(ValidationError, match="safety"):
        BodyComp(body_fat=bf(), rmr_kcal=2100, tdee_kcal=2500, energy=[energy(2000, 2200)])


def test_surplus_over_10_percent_is_rejected():
    bulk = EnergyTarget(
        goal="bulk", kcal_low=2600, kcal_high=2900, protein_g_low=140, protein_g_high=170
    )
    with pytest.raises(ValidationError, match="safety"):
        BodyComp(body_fat=bf(), rmr_kcal=1700, tdee_kcal=2500, energy=[bulk])


def test_opt_out_has_no_energy_numbers():
    assert BodyComp(body_fat=bf()).energy == []


# --- DefinitionScores / UserProfile ---
def test_definition_scores_are_1_to_5_and_experimental_by_default():
    ok = dict.fromkeys(("abs", "chest", "shoulders", "arms", "back", "legs", "adiposity"), 3)
    assert DefinitionScores(**ok).experimental is True
    with pytest.raises(ValidationError):
        DefinitionScores(**(ok | {"abs": 6}))


def profile(**kw):
    base = dict(
        height_cm=178, weight_kg=75, age=25, sex="male", activity="moderate",
        level="beginner", goal="recomp", days_per_week=4,
    )  # fmt: skip
    return UserProfile(**(base | kw))


def test_user_must_be_18_plus():
    profile(age=18)
    with pytest.raises(ValidationError):
        profile(age=17)


def test_sex_is_optional_and_never_defaulted_from_the_other_profile():
    """ADR-037: a missing sex is `not_specified`, not an error and not copied from the user."""
    assert UserProfile(**profile().model_dump(exclude={"sex"})).sex == "not_specified"
    assert GoalProfile().sex == "not_specified"
    assert profile(sex="female").sex == "female"
    with pytest.raises(ValidationError):
        profile(sex="other")
    with pytest.raises(ValidationError):
        GoalProfile(sex="other")
    with pytest.raises(ValidationError):  # extra="forbid" holds for the new model too
        GoalProfile(sex="male", height_cm=180)


def test_analyze_request_goal_profile_defaults_to_not_specified():
    field = AnalyzeRequest.model_fields["goal_profile"]
    assert field.get_default(call_default_factory=True).sex == "not_specified"
