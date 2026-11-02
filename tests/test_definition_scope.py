import pytest
from pydantic import ValidationError

from app.definition_scope import (
    REASON_AGE,
    REASON_BACK,
    REASON_CLOTHED,
    REASON_GOAL,
    REASON_NO_ABDOMEN,
    REASON_REGION,
    definition_statuses,
)
from physique_schemas.models import (
    DEFINITION_REGIONS,
    AnalyzeRequest,
    DefinitionRegionStatus,
    GoalProfile,
    Keypoint,
    Scan,
    UserProfile,
    ViewData,
)

ADULT = 30
OTHER_REGIONS = set(DEFINITION_REGIONS) - {"abs"}


def statuses(scope, age=ADULT, not_shown=()):
    return definition_statuses(scope, age, not_shown)


def test_the_clothed_scan_scores_nothing_with_the_clothed_reason():
    out = statuses("none")
    assert set(out) == set(DEFINITION_REGIONS)
    assert all(s.status == "not_measured_covered" and s.reason == REASON_CLOTHED for s in out.values())


def test_scope_a_does_not_show_the_abdomen_so_definition_is_na_with_that_reason():
    out = statuses("arms_shoulders_legs")
    assert all(s.status == "not_measured_covered" and s.reason == REASON_NO_ABDOMEN for s in out.values())
    assert "abdomen" in REASON_NO_ABDOMEN


def test_scope_b_scores_the_overall_definition_through_abs_and_nothing_else():
    out = statuses("full_body")
    assert out["abs"].status == "measured" and out["abs"].reason is None
    for region in OTHER_REGIONS:
        assert out[region].status == "not_measured_not_in_demo"
    assert out["back"].reason == REASON_BACK
    assert {out[r].reason for r in OTHER_REGIONS - {"back"}} == {REASON_REGION}


def test_a_goal_photo_that_covers_the_abdomen_gives_na_for_the_definition_gap():
    out = statuses("full_body", not_shown=["abs"])
    assert out["abs"].status == "not_measured_goal_covers"
    assert out["abs"].reason == REASON_GOAL


def test_other_regions_in_goal_not_shown_do_not_change_the_overall_score():
    assert statuses("full_body", not_shown=["chest", "legs"])["abs"].status == "measured"


@pytest.mark.parametrize("age", [None, 17, 0])
def test_full_body_under_18_or_unset_is_gated(age):
    out = statuses("full_body", age=age)
    assert all(s.status == "not_measured_age_gate" and s.reason == REASON_AGE for s in out.values())


def test_scope_a_has_no_age_gate():
    out = statuses("arms_shoulders_legs", age=17)
    assert all(s.reason == REASON_NO_ABDOMEN for s in out.values())


def test_the_reason_texts_are_all_different_so_the_ui_can_tell_them_apart():
    reasons = [
        REASON_CLOTHED,
        REASON_NO_ABDOMEN,
        REASON_AGE,
        REASON_GOAL,
        REASON_REGION,
        REASON_BACK,
    ]
    assert len(set(reasons)) == len(reasons)


def test_na_never_carries_a_number_and_always_a_reason():
    for scope in ("none", "arms_shoulders_legs", "full_body"):
        for s in statuses(scope, not_shown=["abs"]).values():
            assert set(s.model_dump()) == {"status", "reason"}  # no score or delta field exists
            assert (s.status == "measured") == (s.reason is None)


def test_status_validator_rejects_na_without_a_reason_and_measured_with_one():
    with pytest.raises(ValidationError):
        DefinitionRegionStatus(status="not_measured_not_in_demo")
    with pytest.raises(ValidationError):
        DefinitionRegionStatus(status="measured", reason="why")


# --- the request ------------------------------------------------------------
def _scan():
    kp = Keypoint(x=0.5, y=0.5, z=0.0, visibility=1.0)
    view = ViewData(mask_png_b64="AAAA", keypoints=[kp] * 33)
    return Scan(views={"front": view, "left": view})


def _profile(age):
    return UserProfile(
        height_cm=175, weight_kg=75, age=age, activity="moderate", level="beginner",
        goal="recomp", days_per_week=3,
    )  # fmt: skip


def _request(scope, profile):
    return AnalyzeRequest(goal=_scan(), current=_scan(), profile=profile, definition_scope=scope)


def test_the_default_scope_is_the_clothed_scan():
    assert _request("none", _profile(ADULT)).definition_scope == "none"
    req = AnalyzeRequest(goal=_scan(), current=_scan(), profile=_profile(ADULT))
    assert req.definition_scope == "none" and req.goal_profile.not_shown == []


def test_a_request_claiming_full_body_for_a_minor_is_rejected_server_side():
    minor = UserProfile.model_construct(**(_profile(ADULT).model_dump() | {"age": 17}))
    with pytest.raises(ValidationError, match="adults"):
        _request("full_body", minor)


def test_an_adult_can_request_every_scope():
    for scope in ("none", "arms_shoulders_legs", "full_body"):
        assert _request(scope, _profile(ADULT)).definition_scope == scope


def test_goal_not_shown_takes_only_definition_regions():
    assert GoalProfile(not_shown=["abs"]).not_shown == ["abs"]
    with pytest.raises(ValidationError):
        GoalProfile(not_shown=["lats"])
