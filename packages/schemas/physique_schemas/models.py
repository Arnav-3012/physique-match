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

REGIONS: tuple[str, ...] = Region.__args__
N_KEYPOINTS = 33  # MediaPipe Pose Landmarker
MAX_PAYLOAD_B64 = 700_000  # ~512 KB of PNG bytes after base64 (architecture.md section 5)
MAX_VIEW_B64 = MAX_PAYLOAD_B64 // 2  # one view may be large; the Scan total is what's capped
KCAL_TOLERANCE = 0.5
UPPER_BACK_REASON = (  # fixed text, ADR-039 item 2
    "Not measurable from a front photo; a back silhouette adds no size information."
)

Score = Annotated[int, Field(ge=1, le=5)]
Unit = Annotated[float, Field(ge=0, le=1)]


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


class VTaper(_Strict):
    """Frame V-taper (estimate), ADR-038: index = shoulders / waist, each B's measurement / height.
    About .40, not the gym ratio: no ideal line. Shown, not ranked. `delta` = goal - current."""

    goal_index: float | None = None
    current_index: float | None = None
    delta: Annotated[float, Field(ge=-1, le=1)] | None = None
    # None until the `v_taper` block (val residuals) is in gap_regressor.json (ADR-038).
    reliability: Unit | None = None
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


class BodyComp(_Strict):
    """Energy fields are empty when the user opts out (eating-disorder screen, BMI/BF floors).
    `body_fat` is None when the user's sex is unset: RFM has a sex term and is not guessed
    (ADR-037 Amendment 4); `note` says why."""

    body_fat: BodyFatRange | None = None
    rmr_kcal: float | None = None
    tdee_kcal: float | None = None
    energy: list[EnergyTarget] = []
    note: str = ""

    @model_validator(mode="after")
    def _safety_floors(self):
        """Deficit <= 20% of TDEE, never below RMR; surplus <= 10% of TDEE."""
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


class DefinitionScores(_Strict):
    """On-device model output; only these numbers leave the browser (ADR-013)."""

    abs: Score
    chest: Score
    shoulders: Score
    arms: Score
    back: Score
    legs: Score
    adiposity: Score
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
    equipment: list[str] = []
    ed_history: bool | None = None  # True -> training plan only, no calorie numbers
    tape_waist_cm: Annotated[float, Field(ge=40, le=200)] | None = None


class GoalProfile(_Strict):
    """What the user says about the goal physique. Never filled from the photo or defaulted to
    the user's sex (ADR-037 Amendment 3): `lats` needs both sexes set and equal."""

    sex: Sex = "not_specified"


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
    body_comp: BodyComp
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

    @model_validator(mode="after")
    def _current_has_a_side_view(self):
        if not {"left", "right"} & self.current.views.keys():
            raise ValueError("the current scan needs a side view: left or right")
        return self


class AnalyzeResponse(_Strict):
    gap: GapProfile
    body_comp: BodyComp
    warnings: list[str] = []  # report-level, e.g. a different-sex goal (ADR-037 Amendment 3)
