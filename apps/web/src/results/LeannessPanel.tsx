import { motion, useReducedMotion } from "motion/react";
import { BAND_LABELS, bandOf, EXPERIMENTAL_BADGE } from "../definition/bands";
import type { Scores } from "../definition/score";
import { Panel } from "../ui/Layout";

const STEPS = ["low", "mid", "high"];

function Track({ band, tone }: { band: string; tone: "you" | "goal" }) {
  const reduce = useReducedMotion();
  const at = Math.max(0, STEPS.indexOf(band));
  return (
    <div>
      <div className="relative grid grid-cols-3 gap-1.5" aria-hidden>
        {STEPS.map((s) => (
          <div key={s} className="h-1.5 rounded-full bg-line" />
        ))}
        <motion.span
          className={`absolute -top-1 size-3.5 rounded-full border-2 border-canvas ${tone === "goal" ? "bg-accent" : "bg-ink-3"}`}
          initial={reduce ? false : { left: "0%" }}
          animate={{ left: `${(at / 3) * 100 + 100 / 6}%` }}
          style={{ x: "-50%" }}
          transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
        />
      </div>
      <p className="mt-3 text-lg capitalize text-ink">{band}</p>
    </div>
  );
}

/** Leanness and adiposity as goal against current, never a stand-alone score (docs/phase-plan.md P7). Computed in the
 * browser; nothing here is sent anywhere. Always carries the experimental label and the limits. */
export default function LeannessPanel({ you, goal }: { you: Scores; goal: Scores }) {
  const rows = [
    { label: BAND_LABELS.definition, you: bandOf(you.definition), goal: bandOf(goal.definition) },
    { label: BAND_LABELS.adiposity, you: bandOf(you.adiposity), goal: bandOf(goal.adiposity) },
  ];
  return (
    <Panel className="mb-5">
      <div className="flex flex-wrap items-center gap-3">
        <h2>Leanness, you and your goal</h2>
        <span className="rounded-full border border-caution/60 px-3 py-1 text-[0.65rem] uppercase tracking-[0.14em] text-caution">{EXPERIMENTAL_BADGE}</span>
      </div>
      <p className="mt-3 text-ink-2">
        Compared with your goal photo, your {rows[0].label.toLowerCase()} is <span className="text-ink">{rows[0].you}</span> (goal: {rows[0].goal}) and your{" "}
        {rows[1].label.toLowerCase()} is <span className="text-ink">{rows[1].you}</span> (goal: {rows[1].goal}).
      </p>
      <div className="mt-6 space-y-6">
        {rows.map((r) => (
          <div key={r.label} className="grid gap-4 sm:grid-cols-[10rem_1fr_1fr] sm:items-end">
            <p className="text-sm text-ink-3">{r.label}</p>
            <div>
              <p className="mb-2 text-[0.65rem] uppercase tracking-[0.16em] text-ink-3">You</p>
              <Track band={r.you} tone="you" />
            </div>
            <div>
              <p className="mb-2 text-[0.65rem] uppercase tracking-[0.16em] text-accent-2">Goal</p>
              <Track band={r.goal} tone="goal" />
            </div>
          </div>
        ))}
      </div>
      <p className="mt-6 text-xs leading-relaxed text-ink-3">
        Adiposity is a waist-girth proxy, not a body-fat percentage, and is least reliable for very muscular bodies. This
        estimate is experimental and has not been validated on real photos; goal photos are often posed, pumped, lit or
        edited. It never changes your body-fat range or your plan, and it is read from the front photos only. Not medical advice.
      </p>
    </Panel>
  );
}
