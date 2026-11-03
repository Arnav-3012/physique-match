import { useState, type ReactNode } from "react";
import type { Sex } from "../capture/scope";
import Choice from "../ui/Choice";
import { Heading, Item, Notice, Panel, Screen } from "../ui/Layout";
import SideFigure from "../ui/SideFigure";
import {
  BANDS_LABEL,
  bandsFor,
  bandText,
  bandVsCurrent,
  MAX_PRIORITY_REGIONS,
  PRIORITY_OPTIONS,
  RATING_ANCHORS,
  ratingNote,
  regionsOf,
  type Band,
  type GoalRegion,
  type Range,
} from "./goalInputs";

export interface GoalChoices {
  band?: Band;
  highPriority: GoalRegion[]; // the gap regions the picked options map to; sent to /plan as `focus`
  selfRating?: number;
}

interface Props {
  sex: Sex;
  currentRange?: Range; // the user's RFM range, once the analysis exists (P7)
  modelLeanness?: number; // the definition model's estimate, once it exists (P4 model step)
  onChange?: (c: GoalChoices) => void;
  onBack: () => void;
  onContinue?: () => void;
  top?: ReactNode; // the goal photo upload, shown before the band and region choices
  canContinue?: boolean;
}

/** Goal body-fat band, region priorities and an optional leanness self-rating (ADR-052). Client state only. */
export default function GoalInputsPanel({ sex, currentRange, modelLeanness, onChange, onBack, onContinue, top, canContinue = true }: Props) {
  const [band, setBand] = useState<Band>();
  const [picked, setPicked] = useState<string[]>([]);
  const [rating, setRating] = useState<number>();
  const bands = bandsFor(sex);
  const used = regionsOf(picked).length;

  const update = (next: GoalChoices) => onChange?.(next);
  const pickBand = (b: Band) => {
    setBand(b);
    update({ band: b, highPriority: regionsOf(picked), selfRating: rating });
  };
  const toggle = (id: string) => {
    const next = picked.includes(id) ? picked.filter((x) => x !== id) : [...picked, id];
    setPicked(next);
    update({ band, highPriority: regionsOf(next), selfRating: rating });
  };
  const rate = (n: number) => {
    setRating(n);
    update({ band, highPriority: regionsOf(picked), selfRating: n });
  };
  const note = ratingNote(rating, modelLeanness);

  return (
    <Screen labelledBy="goal-h" aside={<SideFigure caption="Your goal sets the direction" />}>
      <Heading eyebrow="Step 5 · Your goal" title="Where are you headed?" id="goal-h">
        The plan follows the measured size gaps, the body-fat band you choose and the regions you prioritise. The
        muscle-definition score is an experimental estimate that informs only.
      </Heading>

      <div className="space-y-5">
        {top}
        <Panel>
          <fieldset>
            <legend>Goal body-fat band</legend>
            {bands ? (
              <>
                <div className="mt-5 space-y-3">
                  {bands.map((b) => (
                    <Choice key={b.label} type="radio" name="band" checked={band === b} onChange={() => pickBand(b)} title={bandText(b)} description={bandVsCurrent(b, currentRange)} />
                  ))}
                </div>
                <p className="mt-4 text-sm text-ink-3">
                  Bands are {BANDS_LABEL}, not a medical target. Nothing below the safety floor is offered.
                </p>
              </>
            ) : (
              <div className="mt-5">
                <Notice tone="caution">Set your sex to pick a goal body-fat band: the formulas differ by sex.</Notice>
              </div>
            )}
          </fieldset>
        </Panel>

        <Panel>
          <fieldset>
            <legend>Regions to prioritise</legend>
            <p className="mt-2 text-sm leading-relaxed text-ink-2">
              Pick up to {MAX_PRIORITY_REGIONS} areas for extra weekly volume. Legs use two slots (thighs and calves). If you pick
              none, we use your largest measured gaps. Back, abs and waist always get their weekly minimum.
            </p>
            <p className="mt-2 text-xs uppercase tracking-[0.16em] text-accent-2" aria-live="polite">
              {used} of {MAX_PRIORITY_REGIONS} slots used
            </p>
            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              {PRIORITY_OPTIONS.map((o) => {
                const on = picked.includes(o.id);
                return (
                  <Choice
                    key={o.id}
                    compact
                    type="checkbox"
                    checked={on}
                    disabled={!on && used + o.regions.length > MAX_PRIORITY_REGIONS}
                    onChange={() => toggle(o.id)}
                    title={o.label}
                  />
                );
              })}
            </div>
          </fieldset>
        </Panel>

        <Panel>
          <fieldset>
            <legend>How lean is the goal photo? (optional)</legend>
            <div className="mt-5 grid gap-3 sm:grid-cols-2 sm:[&>*:last-child]:col-span-2" role="radiogroup" aria-label="Goal leanness, 1 to 5">
              {RATING_ANCHORS.map((a, i) => (
                <Choice key={a} compact type="radio" name="rating" checked={rating === i + 1} onChange={() => rate(i + 1)} title={a} />
              ))}
            </div>
            {note && <p className="mt-3 text-sm text-caution">{note}</p>}
          </fieldset>
        </Panel>
      </div>

      <Item className="mt-8 flex flex-wrap gap-3">
        {onContinue && (
          <button className="btn btn-primary disabled:opacity-40" disabled={!canContinue} onClick={onContinue}>
            Continue
          </button>
        )}
        <button className="btn btn-ghost" onClick={onBack}>
          Back
        </button>
      </Item>
    </Screen>
  );
}
