"""Write packages/schemas/schema.json; `npm run types` in apps/web turns it into types.gen.ts."""

import json
from pathlib import Path

from pydantic import BaseModel

from .models import (
    AnalyzeRequest,
    AnalyzeResponse,
    BodyComp,
    DefinitionScores,
    GapProfile,
    PlanRequest,
    PlanResponse,
    Report,
    Scan,
    UserProfile,
)


class Contracts(BaseModel):
    analyze_request: AnalyzeRequest
    analyze_response: AnalyzeResponse
    scan: Scan
    gap_profile: GapProfile
    body_comp: BodyComp
    definition_scores: DefinitionScores
    user_profile: UserProfile
    report: Report
    plan_request: PlanRequest
    plan_response: PlanResponse


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "schema.json"
    out.write_text(json.dumps(Contracts.model_json_schema(), indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
