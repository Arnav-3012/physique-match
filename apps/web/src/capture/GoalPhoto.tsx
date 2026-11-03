import { useState } from "react";
import Choice from "../ui/Choice";
import { Notice, Panel } from "../ui/Layout";
import { processUpload } from "./pipeline";
import { scoreFrontAndClear, type Scores } from "../definition/score";
import { rgbStore } from "./rgbStore";
import type { Scope, Sex } from "./scope";
import { torsoLooksCovered } from "./torso";
import type { ViewCapture } from "./types";

/** The goal views the API accepts: a front (required) and one side, left or right (mirrored by the model). */
export interface GoalViews {
  front?: ViewCapture;
  side?: ViewCapture;
}

const SEXES: { value: Sex; label: string }[] = [
  { value: "male", label: "Male" },
  { value: "female", label: "Female" },
  { value: "not_specified", label: "Prefer not to say" },
];

interface SlotProps {
  title: string;
  badge: string;
  hint: string;
  view?: ViewCapture;
  busy: boolean;
  disabled?: boolean;
  disabledNote?: string;
  onFile: (f?: File) => void;
}

function Slot({ title, badge, hint, view, busy, disabled, disabledNote, onFile }: SlotProps) {
  return (
    <div className={disabled ? "opacity-60" : ""}>
      <div className="flex items-baseline justify-between">
        <p className="display text-2xl">{title}</p>
        <span className="text-[0.62rem] uppercase tracking-[0.16em] text-ink-3">{badge}</span>
      </div>
      <label
        className={`group relative mt-2 block aspect-[3/4] overflow-hidden rounded-xl border bg-canvas transition duration-300 ${
          disabled ? "cursor-not-allowed border-line" : "cursor-pointer border-dashed border-line hover:border-accent/60"
        } ${view?.maskPngB64 ? "border-solid border-accent/50" : ""}`}
      >
        <input type="file" accept="image/*" className="sr-only" disabled={busy || disabled} onChange={(e) => onFile(e.target.files?.[0])} />
        {view?.maskPngB64 ? (
          <>
            <img src={`data:image/png;base64,${view.maskPngB64}`} alt={`${title} outline`} className="h-full w-full object-contain" />
            <span className="scanline absolute inset-x-0 h-px bg-accent shadow-[0_0_12px_2px_rgb(200_168_106_/_0.6)]" aria-hidden />
            <span className="absolute inset-x-0 bottom-0 bg-canvas/80 py-1.5 text-center text-[0.68rem] uppercase tracking-[0.14em] text-accent-2 backdrop-blur">
              Replace photo
            </span>
          </>
        ) : (
          <span className="absolute inset-0 grid place-items-center px-3 text-center">
            <span>
              <span className="block text-sm text-ink">{busy ? "Reading…" : disabled ? "Not used yet" : "Choose a photo"}</span>
              <span className="mt-1 block text-xs leading-relaxed text-ink-3">{disabled ? disabledNote : hint}</span>
            </span>
          </span>
        )}
      </label>
    </div>
  );
}

interface Props {
  views: GoalViews;
  onViews: (v: GoalViews) => void;
  goalSex: Sex | undefined;
  onGoalSex: (s: Sex) => void;
  scoreWanted?: boolean; // the on-device leanness opt-in (full body only)
  scope?: Scope;
  onGoalScores?: (s: Scores | null) => void;
}

/** The goal photos (Mode 1, ADR-025). Each is read in the browser like the scan; only its outline is kept.
 * Only the FRONT photo can be scored: the definition model was trained on front and back renders and its overall head
 * on front images only (ml/src/definition/views.py), and never saw a side view. */
export default function GoalPhoto({ views, onViews, goalSex, onGoalSex, scoreWanted, scope, onGoalScores }: Props) {
  const [busy, setBusy] = useState<"front" | "side" | null>(null);
  const [error, setError] = useState("");
  const [side, setSide] = useState<"left" | "right">("left");
  const [scoreNote, setScoreNote] = useState("");

  async function pick(slot: "front" | "side", file?: File) {
    if (!file) return;
    setBusy(slot);
    setError("");
    try {
      const v = await processUpload(file, slot === "front" ? "front" : side);
      let scores: Scores | null = null;
      if (slot === "front") {
        const covered = !!v && torsoLooksCovered(scope, v.torsoSkinShare);
        if (scoreWanted && v?.maskPngB64 && !covered) {
          scores = await scoreFrontAndClear(() => import("../definition/models").then((m) => m.loadRunners())); // scores, then clears
          setScoreNote(scores ? "Leanness estimate scored on this device. The photo was released." : "The leanness estimate could not be scored for this photo.");
        } else {
          rgbStore.clear();
          setScoreNote(scoreWanted && covered ? "The torso in this goal photo looks covered, so its leanness is not scored." : "");
        }
        onGoalScores?.(scores);
      } else rgbStore.clear(); // a side photo's pixels are never kept, and it is never scored
      if (!v || !v.maskPngB64) setError("We could not find a person in this photo. Try one with the whole body in frame.");
      else onViews({ ...views, [slot]: v });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not read this image.");
    } finally {
      setBusy(null);
    }
  }

  const pickSide = (s: "left" | "right") => {
    setSide(s);
    if (views.side) onViews({ ...views, side: { ...views.side, name: s } });
  };

  const notes = [views.front, views.side].flatMap((v) => (v ? [...v.issues, ...v.warnings] : []));

  return (
    <Panel>
      <fieldset>
        <legend>Your goal photos</legend>
        <p className="mt-2 max-w-lg text-sm leading-relaxed text-ink-2">
          Photos of the physique you are aiming for. A front view is enough; a side view makes the comparison sharper. They
          are read in your browser and never uploaded; only their outlines are kept. Use photos you have the right to use.
        </p>

        <div className="mt-5 grid grid-cols-3 gap-3 sm:gap-4">
          <Slot title="Front" badge="Required" hint="Whole body, facing the camera" view={views.front} busy={busy === "front"} onFile={(f) => pick("front", f)} />
          <Slot title="Side" badge="Optional" hint="Sharpens waist, thigh and other girths" view={views.side} busy={busy === "side"} onFile={(f) => pick("side", f)} />
          <Slot
            title="Back"
            badge="Later"
            hint=""
            busy={false}
            disabled
            disabledNote="A back outline adds no size information, and the leanness model reads only the front photo for now."
            onFile={() => {}}
          />
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-3" role="radiogroup" aria-label="Which side the side photo shows">
          <span className="text-[0.7rem] font-medium uppercase tracking-[0.18em] text-ink-3">Side photo shows</span>
          <div className="grid w-56 grid-cols-2 gap-2">
            <Choice compact type="radio" name="goal-side" checked={side === "left"} onChange={() => pickSide("left")} title="Left" />
            <Choice compact type="radio" name="goal-side" checked={side === "right"} onChange={() => pickSide("right")} title="Right" />
          </div>
        </div>

        {scoreWanted && (
          <p className="mt-4 text-sm text-ink-2">
            {scoreNote || "Leanness is read from the front photo only, because the model was never trained on side or back views."}
          </p>
        )}
        {error && <p className="mt-4 text-sm text-[#e58b7b]">{error}</p>}
        {notes.length > 0 && (
          <div className="mt-4">
            <Notice tone="caution" title="Check these photos">
              {[...new Set(notes)].join(" ")} You can continue, but the comparison will be less reliable.
            </Notice>
          </div>
        )}

        <p className="mt-6 text-[0.7rem] font-medium uppercase tracking-[0.18em] text-ink-3" id="goal-sex-label">
          Whose body is in the goal photos?
        </p>
        <div className="mt-2 grid gap-3 sm:grid-cols-3" role="radiogroup" aria-labelledby="goal-sex-label">
          {SEXES.map((s) => (
            <Choice key={s.value} compact type="radio" name="goal-sex" checked={goalSex === s.value} onChange={() => onGoalSex(s.value)} title={s.label} />
          ))}
        </div>
        <p className="mt-3 text-sm text-ink-3">Asked separately from your own. It is never copied from yours or guessed from the photo.</p>
        <p className="mt-3 text-sm text-ink-3">
          Goal photos are often posed, pumped, lit or edited, so anything we read from one is an appearance-based estimate.
        </p>
      </fieldset>
    </Panel>
  );
}
