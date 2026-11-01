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
Sex = Literal["male", "female"]
Level = Literal["beginner", "intermediate", "advanced"]
Activity = Literal["sedentary", "light", "moderate", "active", "very_active"]

REGIONS: tuple[str, ...] = Region.__args__
N_KEYPOINTS = 33  # MediaPipe Pose Landmarker
MAX_PAYLOAD_B64 = 700_000  # ~512 KB of PNG bytes after base64 (architecture.md section 5)
MAX_VIEW_B64 = MAX_PAYLOAD_B64 // 2  # one view may be large; the Scan total is what's capped
KCAL_TOLERANCE = 0.5

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
    """Front + left are required (Siamese input); back/right feed asymmetry and lat width."""

    views: dict[View, ViewData]

    @model_validator(mode="after")
    def _check_views(self):
        missing = {"front", "left"} - self.views.keys()
        if missing:
            raise ValueError(f"missing required views: {sorted(missing)}")
        if sum(len(v.mask_png_b64) for v in self.views.values()) > MAX_PAYLOAD_B64:
            raise ValueError("scan payload exceeds 512 KB")
        return self


# --- Analyzer output ------------------------------------------------------
class RegionGap(_Strict):
    delta: Annotated[float, Field(ge=-1, le=1)]  # (goal - current) / height; + = goal larger
    confidence: Unit


class GapProfile(_Strict):
    analyzer: Literal["geometric", "siamese"]
    regions: dict[Region, RegionGap]
    asymmetry: dict[Region, float] | None = None  # left - right, height-normalised

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
    """Energy fields are empty when the user opts out (eating-disorder screen, BMI/BF floors)."""

    body_fat: BodyFatRange
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
    sex: Sex
    activity: Activity
    level: Level
    goal: Literal["cut", "recomp", "bulk"]
    days_per_week: Annotated[int, Field(ge=1, le=7)]
    equipment: list[str] = []
    ed_history: bool | None = None  # True -> training plan only, no calorie numbers
    tape_waist_cm: Annotated[float, Field(ge=40, le=200)] | None = None


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


class AnalyzeResponse(_Strict):
    gap: GapProfile
    body_comp: BodyComp
