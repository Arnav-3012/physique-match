"""What the model may return (ADR-058). There is deliberately NO sets field: sets, caps and floors come
from the planner only, so the model cannot raise or lower any volume."""

from pydantic import BaseModel, ConfigDict, Field

from app.planner.params import MAX_EXERCISES_PER_DAY


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WriterItem(_Strict):
    exercise_id: str = Field(max_length=80)  # must be one of the candidates shown for that day
    reps: str = Field(pattern=r"^\d{1,2}(-\d{1,2})?$")
    cue: str = Field(max_length=160)


class WriterDay(_Strict):
    label: str = Field(max_length=40)
    items: list[WriterItem] = Field(min_length=1, max_length=MAX_EXERCISES_PER_DAY)


class WriterOutput(_Strict):
    summary: str = Field(max_length=500)
    days: list[WriterDay] = Field(min_length=1, max_length=7)
