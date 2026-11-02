"""Contracts between browser, API and agent. TS types are generated from these (export_schema.py).

Privacy by construction: `extra="forbid"` means an unexpected field (e.g. an `image`) is rejected,
and a Scan can only carry 1-bit mask PNGs + keypoints (ADR-003).
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

View = Literal["front", "left", "back", "right"]
Region = Literal[
    "shoulders", "chest", "upper_back", "lats", "waist",
    "hips", "upper_arm", "forearm", "thigh", "calf",
]  # fmt: skip
EnergyGoal = Literal["cut", "recomp", "bulk", "maintain"]
Sex = Literal["male", "female", "not_specified"]  # user-entered, never inferred (ADR-037)
Level = Literal["beginner", "intermediate", "advanced"]
Activity = Literal["sedentary", "light", "moderate", "active", "very_active"]
# Which path wrote the plan text, and why the writer did not (ADR-058). Closed enums, no free text.
ServedBy = Literal["primary", "fallback", "deterministic"]
FallbackReason = Literal["disabled", "timeout", "rate_limited", "invalid_json", "check_failed", "unsafe_text"]
# Closed checklist, no free text (ADR-054): the `equipment` values of the planner's pool (strength +
# powerlifting) in kb/raw/exercises.json, spelled as the dataset spells them. A test pins the two.
Equipment = Literal[
    "barbell", "dumbbell", "cable", "body only", "machine", "kettlebells",
    "other", "bands", "e-z curl bar", "exercise ball", "none", "medicine ball",
]  # fmt: skip

REGIONS: tuple[str, ...] = Region.__args__
N_KEYPOINTS = 33  # MediaPipe Pose Landmarker
MAX_PAYLOAD_B64 = 700_000  # ~512 KB of PNG bytes after base64 (architecture.md section 5)
MAX_VIEW_B64 = MAX_PAYLOAD_B64 // 2  # one view may be large; the Scan total is what's capped
KCAL_TOLERANCE = 0.5
MIN_ADULT_AGE = 18  # the 18+ gate (UserProfile.age) and the full-body definition scope (ADR-051)
UPPER_BACK_REASON = (  # fixed text, ADR-039 item 2
    "Not measurable from a front photo; a back silhouette adds no size information."
)

Score = Annotated[int, Field(ge=1, le=5)]
Unit = Annotated[float, Field(ge=0, le=1)]
# Scale-free silhouette waist girth (girth / body height) that /analyze returns for /plan. A sanity
# bound, ours and not fitted: 0.2 is a 35 cm waist at 175 cm, 1.0 a waist as long as the body is tall.
WaistG = Annotated[float, Field(ge=0.2, le=1.0)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- Scan -----------------------------------------------------------------
class Keypoint(_Strict):
    x: Unit
    y: Unit
    z: float
    visibility: Unit


class ViewData(_Strict):
    mask_png_b64: str = Field(min_length=1, max_length=MAX_VIEW_B64)
    keypoints: list[Keypoint] = Field(min_length=N_KEYPOINTS, max_length=N_KEYPOINTS)


class Scan(_Strict):
    """Front is required: a goal photo may have nothing else (ADR-025). A current scan also needs a
    side view (left, or right mirrored), which AnalyzeRequest checks. Back feeds lat width."""

    views: dict[View, ViewData]

    @model_validator(mode="after")
    def _check_views(self):
        if "front" not in self.views:
            raise ValueError("missing required views: ['front']")
        if sum(len(v.mask_png_b64) for v in self.views.values()) > MAX_PAYLOAD_B64:
            raise ValueError("scan payload exceeds 512 KB")
        return self


# --- Analyzer output ------------------------------------------------------
class RegionGap(_Strict):
    delta: Annotated[float, Field(ge=-1, le=1)]  # (goal - current) / height; + = goal larger
    confidence: Unit  # UI + model card: "region reliability", fixed per region and slice (ADR-035)
    # Each person's measurement / height; siamese_subtract only, None for back-view regions.
    goal_ratio: float | None = None
    current_ratio: float | None = None
    # False = the view this region needs was missing. The report says "not measured"; delta and
    # confidence are 0 placeholders and must not be read as a result.
    measured: bool = True
    reason: str | None = None  # why it is not measured; shown in the report (ADR-037, ADR-039)

    @model_validator(mode="after")
    def _unmeasured_carries_no_numbers(self):
        blank = self.delta == 0 and self.confidence == 0
        no_ratios = self.goal_ratio is None and self.current_ratio is None
        if not self.measured and not (blank and no_ratios):
            raise ValueError("an unmeasured region carries no numbers")
        if self.measured and self.reason is not None:
            raise ValueError("only an unmeasured region has a reason")
        return self


class ZoneLabel(_Strict):
    """ADR-041 / ADR-042: the exact wording for a zone, chosen by the ANSUR support at its end.
    The report and the agent use `text` as is; case 3 carries no "reachable" claim."""

    support: Annotated[int, Field(ge=0)]  # ANSUR subjects of the user's sex at or below the end
    extrapolated: bool  # the end sits below every ANSUR subject of that sex
    label_case: Literal[1, 2, 3]
    text: str


class Ceiling(_Strict):
    """Reachable frame ceiling, ADR-040: the V-taper index the user's own shoulders allow at a
    leaner waist, from RFM inverted at `BF_ATHLETE_UPPER` and `BF_FLOOR`. Applied in the route
    (it needs the user's sex), like `lats`."""

    at_athlete_upper: float | None = None  # the lower ceiling: "reachable on frame alone"
    at_floor: float | None = None  # the higher ceiling: "reachable only near the body-fat floor"
    reachable: ZoneLabel | None = None  # zone 1 wording, support at the athlete-upper end
    near_floor: ZoneLabel | None = None  # zone 2 wording, support at the floor end
    # The goal's zone and its exact wording (ADR-040 flag): 1 reachable on frame alone, 2 only near
    # the floor, 3 needs more than frame. One-sided: it never says a goal is too small.
    zone: Literal[1, 2, 3] | None = None
    zone_text: str | None = None
    measured: bool = True
    reason: str | None = None

    @model_validator(mode="after")
    def _numbers_iff_measured(self):
        parts = (
            self.at_athlete_upper, self.at_floor, self.reachable, self.near_floor,
            self.zone, self.zone_text,
        )  # fmt: skip
        if self.measured and None in parts:
            raise ValueError("a measured ceiling needs both ends, both labels and the zone")
        if self.measured and self.reason is not None:
            raise ValueError("only an unmeasured ceiling has a reason")
        if not self.measured and any(p is not None for p in parts):
            raise ValueError("an unmeasured ceiling carries no numbers")
        return self


class VTaper(_Strict):
    """Frame V-taper (estimate), ADR-038: index = shoulders / waist, each B's measurement / height.
    About .40, not the gym ratio: no ideal line. Shown, not ranked. `delta` = goal - current."""

    goal_index: float | None = None
    current_index: float | None = None
    delta: Annotated[float, Field(ge=-1, le=1)] | None = None
    # None until the `v_taper` block (val residuals) is in gap_regressor.json (ADR-038).
    reliability: Unit | None = None
    ceiling: Ceiling | None = None  # None until the route adds it (ADR-040)
    measured: bool = True  # False = shoulders or waist was not measured; the numbers are all None
    reason: str | None = None

    @model_validator(mode="after")
    def _numbers_iff_measured(self):
        numbers = (self.goal_index, self.current_index, self.delta)
        if self.measured and None in numbers:
            raise ValueError("a measured V-taper needs both indices and the delta")
        if self.measured and self.reason is not None:
            raise ValueError("only an unmeasured V-taper has a reason")
        if not self.measured and any(n is not None for n in (*numbers, self.reliability)):
            raise ValueError("an unmeasured V-taper carries no numbers")
        return self


class GapProfile(_Strict):
    analyzer: Literal["geometric", "siamese_subtract"]
    regions: dict[Region, RegionGap]
    goal_front_only: bool = False  # the goal had no side view; the UI says so (ADR-025)
    asymmetry: dict[Region, float] | None = None  # left - right, height-normalised
    v_taper: VTaper | None = None  # None when the analyzer has no per-person ratios (geometric)

    @model_validator(mode="after")
    def _all_regions(self):
        missing = set(REGIONS) - self.regions.keys()
        if missing:
            raise ValueError(f"missing regions: {sorted(missing)}")
        return self


# --- Body composition -----------------------------------------------------
class BodyFatRange(_Strict):
    """Always a range, never a point estimate (privacy-and-safety.md)."""

    low: Annotated[float, Field(ge=3, le=60)]
    high: Annotated[float, Field(ge=3, le=60)]
    method: str
    confidence: Literal["low", "medium", "high"]

    @model_validator(mode="after")
    def _is_range(self):
        if self.low >= self.high:
            raise ValueError("body fat must be a range: low < high")
        return self


class EnergyTarget(_Strict):
    goal: EnergyGoal
    kcal_low: Annotated[float, Field(gt=0)]
    kcal_high: Annotated[float, Field(gt=0)]
    protein_g_low: Annotated[float, Field(gt=0)]
    protein_g_high: Annotated[float, Field(gt=0)]

    @model_validator(mode="after")
    def _ordered(self):
        if self.kcal_low > self.kcal_high or self.protein_g_low > self.protein_g_high:
            raise ValueError("low must be <= high")
        return self


class ProteinRange(_Strict):
    """Cited typical protein range (kb/nutrition.json, ADR-048). A range, never a point: nothing
    downstream picks a value inside it unless that is stated as our rule, not a cited number."""

    g_per_kg_low: Annotated[float, Field(gt=0)]
    g_per_kg_high: Annotated[float, Field(gt=0)]
    g_low: Annotated[int, Field(gt=0)]  # g_per_kg x weight_kg, total body weight
    g_high: Annotated[int, Field(gt=0)]
    unit: str  # "g per kg of total body weight per day"
    label: str  # "typical range"
    also_listed: str  # the ISSN 1.4-2.0 range, shown beside ours
    text: str  # the wording the report and the agent use as is

    @model_validator(mode="after")
    def _ordered(self):
        if self.g_per_kg_low > self.g_per_kg_high or self.g_low > self.g_high:
            raise ValueError("low must be <= high")
        return self


class BodyComp(_Strict):
    """Energy fields are empty when the user opts out (eating-disorder screen, BMI/BF floors).
    `body_fat` is None when the user's sex is unset: RFM has a sex term and is not guessed
    (ADR-037 Amendment 4); `note` says why. `energy_measured` is False whenever no calorie numbers
    are given (sex unset, under 18, eating-disorder opt-out). `protein` is a per-kg range that
    needs only body weight, so it still shows when sex is unset (ADR-048 Amendment 2)."""

    body_fat: BodyFatRange | None = None
    rmr_kcal: float | None = None
    tdee_kcal: float | None = None
    energy: list[EnergyTarget] = []
    protein: ProteinRange | None = None
    energy_measured: bool = True
    note: str = ""

    @model_validator(mode="after")
    def _safety_floors(self):
        """Deficit <= 20% of TDEE, never below RMR; surplus <= 10% of TDEE."""
        if not self.energy_measured and (self.energy or self.rmr_kcal is not None or self.tdee_kcal is not None):
            raise ValueError("calories not measured: no energy numbers allowed")
        if not self.energy:
            return self
        if self.rmr_kcal is None or self.tdee_kcal is None:
            raise ValueError("energy targets need rmr_kcal and tdee_kcal")
        floor = max(self.rmr_kcal, 0.8 * self.tdee_kcal) - KCAL_TOLERANCE
        cap = 1.1 * self.tdee_kcal + KCAL_TOLERANCE
        for t in self.energy:
            if t.kcal_low < floor or t.kcal_high > cap:
                raise ValueError(f"{t.goal} target outside safety floor/cap")
        return self


DefinitionRegion = Literal["abs", "chest", "shoulders", "arms", "back", "legs"]
DEFINITION_REGIONS: tuple[str, ...] = DefinitionRegion.__args__
# What the user chose to compare, asked once before capture (ADR-051).
DefinitionScope = Literal["arms_shoulders_legs", "full_body", "none"]
DefinitionStatus = Literal[
    "measured",  # eligible for an on-device score; scores are not in the request yet
    "not_measured_covered",  # clothing covers the region in this scan option
    "not_measured_age_gate",  # the full-body option is for adults
    "not_measured_goal_covers",  # the goal photo does not show the region bare, or at all
    "not_measured_not_in_demo",  # no trained head for this region in the demo (ADR-053 Amendment 4)
]


class DefinitionRegionStatus(_Strict):
    """Per-region definition status. N/A always has a reason and is never a number (ADR-051)."""

    status: DefinitionStatus
    reason: str | None = None

    @model_validator(mode="after")
    def _reason_iff_not_measured(self):
        if self.status == "measured" and self.reason is not None:
            raise ValueError("a measured region has no reason")
        if self.status != "measured" and not self.reason:
            raise ValueError("a region that is not measured needs a reason")
        return self


class DefinitionScores(_Strict):
    """On-device model output; only these two numbers leave the browser (ADR-013). There is deliberately
    no per-region field: the regional heads did not beat the overall score, so a payload, a report or an
    agent prompt built from this cannot say one region is less defined than another (ADR-053 Amendment 6)."""

    definition: Score  # "leanness & muscularity (estimate)": the 8-head model's overall head, class 1-5
    adiposity: Score  # waist-girth proxy: the 2-head model's adiposity head, class 1-5
    experimental: bool = True  # stays True until real-data rho >= 0.5 (ADR-020)


# --- User input -----------------------------------------------------------
class UserProfile(_Strict):
    height_cm: Annotated[float, Field(ge=120, le=230)]
    weight_kg: Annotated[float, Field(ge=30, le=300)]
    age: Annotated[int, Field(ge=18, le=100)]  # 18+ gate
    sex: Sex = "not_specified"
    activity: Activity
    level: Level
    goal: Literal["cut", "recomp", "bulk"]
    days_per_week: Annotated[int, Field(ge=1, le=7)]
    equipment: list[Equipment] = []
    ed_history: bool | None = None  # True -> training plan only, no calorie numbers
    tape_waist_cm: Annotated[float, Field(ge=40, le=200)] | None = None


class GoalProfile(_Strict):
    """What the user says about the goal physique. Never filled from the photo or defaulted to
    the user's sex (ADR-037 Amendment 3): `lats` needs both sexes set and equal."""

    sex: Sex = "not_specified"
    # Regions the goal photo does not show bare, or at all: typed by the user, never inferred.
    not_shown: list[DefinitionRegion] = []


# --- Report ---------------------------------------------------------------
class PlannedExercise(_Strict):
    name: str
    muscle: str
    sets: Annotated[int, Field(ge=1, le=10)]
    reps: str
    cue: str
    citations: list[str] = Field(min_length=1)  # retrieved chunk ids


class PlanDay(_Strict):
    label: str
    exercises: list[PlannedExercise] = Field(min_length=1)


class Report(_Strict):
    summary: str
    priorities: list[Region] = Field(max_length=3)
    gap: GapProfile
    body_comp: BodyComp | None = None  # None when no waist was available (/plan says why)
    definition: DefinitionScores | None = None
    plan: list[PlanDay] = Field(min_length=1)
    realism_note: str | None = None
    disclaimer: str

    @model_validator(mode="after")
    def _disclaimer(self):
        if "not medical advice" not in self.disclaimer.lower():
            raise ValueError("disclaimer must state it is not medical advice")
        return self


# --- /analyze -------------------------------------------------------------
class AnalyzeRequest(_Strict):
    """Goal and current scans are silhouettes only; the profile is numbers (ADR-003)."""

    goal: Scan
    current: Scan
    profile: UserProfile
    goal_profile: GoalProfile = Field(default_factory=GoalProfile)
    definition_scope: DefinitionScope = "none"  # ADR-051
    definition: DefinitionScores | None = None  # the current scan's two on-device scores, only if the user opted in

    @model_validator(mode="after")
    def _scores_need_the_full_body_scope(self):
        if self.definition is not None and self.definition_scope != "full_body":
            raise ValueError("definition scores are only accepted for the full-body scope")
        return self

    @model_validator(mode="after")
    def _current_has_a_side_view(self):
        if not {"left", "right"} & self.current.views.keys():
            raise ValueError("the current scan needs a side view: left or right")
        return self

    @model_validator(mode="after")
    def _full_body_is_for_adults(self):
        if self.definition_scope == "full_body" and self.profile.age < MIN_ADULT_AGE:
            raise ValueError("the full-body definition scope is for adults")
        return self


class AnalyzeResponse(_Strict):
    gap: GapProfile
    body_comp: BodyComp
    warnings: list[str] = []  # report-level, e.g. a different-sex goal (ADR-037 Amendment 3)
    # One entry per definition region, set by the route from the scope; N/A always carries a reason.
    definition_status: dict[DefinitionRegion, DefinitionRegionStatus] = {}
    # The current scan's silhouette waist girth, handed back so /plan can compute body composition
    # without the scans. None when a tape value or an unset sex made it unnecessary. Not a person's
    # waist in cm: a scale-free number, bound-checked when it comes back in a PlanRequest.
    waist_g: float | None = None


# --- /plan ----------------------------------------------------------------
class PlanRequest(_Strict):
    """No scans: the profile, the gap /analyze returned and, without a tape waist, its waist_g.
    The server recomputes body composition itself and never trusts a client-supplied BodyComp."""

    profile: UserProfile
    gap: GapProfile
    waist_g: WaistG | None = None
    goal_sex: Sex = "not_specified"  # typed by the user, never defaulted to the profile's sex


class PlanSlot(_Strict):
    exercise_id: str  # the retrieved chunk id, also its citation
    name: str
    sets: Annotated[int, Field(ge=1, le=10)]


class PlanShortfall(_Strict):
    """A muscle group that ended under its cited weekly floor. Never silent, never lowered."""

    group: str
    planned: Annotated[float, Field(ge=0)]
    floor: Annotated[float, Field(gt=0)]


class PlanResponse(_Strict):
    priorities: list[Region] = Field(max_length=3)
    days: list[list[PlanSlot]]
    weekly: dict[str, float]  # recomputed fractional sets per tracked muscle group
    shortfalls: list[PlanShortfall]
    note: str = ""  # fixed sentence for weeks of 2 days or fewer with a shortfall (ADR-056)
    body_comp: BodyComp | None = None  # None when no waist was available (see body_comp_note)
    body_comp_note: str = ""
    retrieval: Literal["hybrid", "bm25"]  # which retriever picked the candidates
    served_by: ServedBy
    # primary: none. fallback: why the primary failed. deterministic: the last failure.
    fallback_reason: FallbackReason | None = None
    warnings: list[str] = []  # fixed, code-written sentences only
    report: Report | None = None  # None when a day has no exercise (nothing to write about)

    @model_validator(mode="after")
    def _reason_matches_path(self):
        if (self.served_by == "primary") != (self.fallback_reason is None):
            raise ValueError("fallback_reason is set exactly when the primary did not serve")
        return self
