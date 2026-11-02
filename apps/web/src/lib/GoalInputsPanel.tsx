import { useState } from "react";
import type { Sex } from "../capture/scope";
import {
  BANDS_LABEL,
  bandsFor,
  bandText,
  bandVsCurrent,
  GOAL_REGIONS,
  RATING_ANCHORS,
  ratingNote,
  REGION_LABEL,
  type Band,
  type GoalRegion,
  type Range,
} from "./goalInputs";

export interface GoalChoices {
  band?: Band;
  highPriority: GoalRegion[];
  selfRating?: number;
}

interface Props {
  sex: Sex;
  currentRange?: Range; // the user's RFM range, once the analysis exists (P7)
  modelLeanness?: number; // the definition model's estimate, once it exists (P4 model step)
  onChange?: (c: GoalChoices) => void;
  onBack: () => void;
}

/** Goal body-fat band, region priorities and an optional leanness self-rating (ADR-052). Client state only. */
export default function GoalInputsPanel({ sex, currentRange, modelLeanness, onChange, onBack }: Props) {
  const [band, setBand] = useState<Band>();
  const [high, setHigh] = useState<GoalRegion[]>([]);
  const [rating, setRating] = useState<number>();
  const bands = bandsFor(sex);

  const update = (next: GoalChoices) => onChange?.(next);
  const pickBand = (b: Band) => {
    setBand(b);
    update({ band: b, highPriority: high, selfRating: rating });
  };
  const toggle = (r: GoalRegion) => {
    const next = high.includes(r) ? high.filter((x) => x !== r) : [...high, r];
    setHigh(next);
    update({ band, highPriority: next, selfRating: rating });
  };
  const rate = (n: number) => {
    setRating(n);
    update({ band, highPriority: high, selfRating: n });
  };
  const note = ratingNote(rating, modelLeanness);

  return (
    <section className="mx-auto max-w-xl p-4" aria-labelledby="goal-h">
      <h1 id="goal-h" className="text-xl font-bold">Your goal</h1>
      <p className="mt-1 text-gray-600">
        The plan follows the measured size gaps, the body-fat band you choose and the regions you prioritise. The
        muscle-definition score is an experimental estimate that informs only.
      </p>

      <fieldset className="mt-4">
        <legend className="font-semibold">Goal body-fat band</legend>
        {bands ? (
          <>
            {bands.map((b) => (
              <label key={b.label} className="mt-2 flex items-start gap-2">
                <input type="radio" name="band" className="mt-1" checked={band === b} onChange={() => pickBand(b)} />
                <span>
                  {bandText(b)}
                  <span className="block text-sm text-gray-500">{bandVsCurrent(b, currentRange)}</span>
                </span>
              </label>
            ))}
            <p className="mt-2 text-sm text-gray-500">
              Bands are {BANDS_LABEL}, not a medical target. Nothing below the safety floor is offered.
            </p>
          </>
        ) : (
          <p className="mt-1 text-amber-700">Set your sex to pick a goal body-fat band: the formulas differ by sex.</p>
        )}
      </fieldset>

      <fieldset className="mt-4">
        <legend className="font-semibold">Regions to prioritise</legend>
        <div className="mt-1 grid grid-cols-2 gap-1">
          {GOAL_REGIONS.map((r) => (
            <label key={r} className="flex items-center gap-2">
              <input type="checkbox" checked={high.includes(r)} onChange={() => toggle(r)} />
              {REGION_LABEL[r]}
            </label>
          ))}
        </div>
      </fieldset>

      <fieldset className="mt-4">
        <legend className="font-semibold">How lean is the goal photo? (optional)</legend>
        <div className="mt-1 flex flex-wrap gap-3" role="radiogroup" aria-label="Goal leanness, 1 to 5">
          {RATING_ANCHORS.map((a, i) => (
            <label key={a} className="flex items-center gap-1">
              <input type="radio" name="rating" checked={rating === i + 1} onChange={() => rate(i + 1)} />
              {a}
            </label>
          ))}
        </div>
        {note && <p className="mt-1 text-sm text-amber-700">{note}</p>}
      </fieldset>

      <button className="mt-4 rounded border px-4 py-2" onClick={onBack}>Back</button>
    </section>
  );
}
