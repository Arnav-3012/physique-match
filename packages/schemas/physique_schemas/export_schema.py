"""Write packages/schemas/schema.json; `npm run types` in apps/web turns it into types.gen.ts."""

import json
from pathlib import Path

from pydantic import BaseModel

from .models import BodyComp, DefinitionScores, GapProfile, Report, Scan, UserProfile


class Contracts(BaseModel):
    scan: Scan
    gap_profile: GapProfile
    body_comp: BodyComp
    definition_scores: DefinitionScores
    user_profile: UserProfile
    report: Report


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "schema.json"
    out.write_text(json.dumps(Contracts.model_json_schema(), indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
