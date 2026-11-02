"""The definition payload carries two numbers and no per-region definition (ADR-053 Amendment 6)."""

import pytest
from pydantic import ValidationError

from physique_schemas.models import (
    DEFINITION_REGIONS,
    AnalyzeRequest,
    DefinitionScores,
    Keypoint,
    Report,
    Scan,
    UserProfile,
    ViewData,
)


def _request(scope, definition=None):
    kp = Keypoint(x=0.5, y=0.5, z=0.0, visibility=1.0)
    view = ViewData(mask_png_b64="AAAA", keypoints=[kp] * 33)
    scan = Scan(views={"front": view, "left": view})
    profile = UserProfile(
        height_cm=175, weight_kg=75, age=30, activity="moderate", level="beginner",
        goal="recomp", days_per_week=3,
    )  # fmt: skip
    return AnalyzeRequest(goal=scan, current=scan, profile=profile, definition_scope=scope, definition=definition)


def test_the_scores_are_the_overall_definition_the_adiposity_and_the_badge_only():
    assert set(DefinitionScores.model_fields) == {"definition", "adiposity", "experimental"}
    assert set(DefinitionScores.model_json_schema()["properties"]).isdisjoint(DEFINITION_REGIONS)


@pytest.mark.parametrize("region", DEFINITION_REGIONS)
def test_a_regional_score_is_rejected_not_ignored(region):
    with pytest.raises(ValidationError):
        DefinitionScores(definition=3, adiposity=2, **{region: 3})
    with pytest.raises(ValidationError):
        DefinitionScores.model_validate({"definition": 3, "adiposity": 2, region: 3})


def test_the_scores_are_classes_one_to_five_and_experimental_by_default():
    assert DefinitionScores(definition=1, adiposity=5).experimental is True
    for bad in (0, 6):
        with pytest.raises(ValidationError):
            DefinitionScores(definition=bad, adiposity=3)


def test_the_request_and_the_report_use_the_two_number_model():
    assert AnalyzeRequest.model_fields["definition"].annotation == DefinitionScores | None
    assert Report.model_fields["definition"].annotation == DefinitionScores | None
    assert _request("full_body", DefinitionScores(definition=4, adiposity=2)).definition.definition == 4


@pytest.mark.parametrize("scope", ["none", "arms_shoulders_legs"])
def test_scores_with_a_scope_that_does_not_show_the_abdomen_are_rejected(scope):
    with pytest.raises(ValidationError, match="full-body"):
        _request(scope, DefinitionScores(definition=3, adiposity=3))


def test_no_scores_is_the_default_for_every_scope():
    for scope in ("none", "arms_shoulders_legs", "full_body"):
        assert _request(scope).definition is None
