import { motion, useReducedMotion } from "motion/react";
import type { PlanResponse } from "../lib/types.gen";
import { Notice } from "../ui/Layout";

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

/** The weekly plan as day cards. Uses the written report when there is one, otherwise the planner's own slots. */
export default function PlanCards({ plan }: { plan: PlanResponse }) {
  const reduce = useReducedMotion();
  const written = plan.report?.plan;
  const days = written
    ? written.map((d) => ({
        label: d.label,
        items: d.exercises.map((e) => ({ name: e.name, detail: `${e.sets} × ${e.reps}`, muscle: e.muscle, cue: e.cue, sources: e.citations.length })),
      }))
    : plan.days.map((d, i) => ({
        label: `Day ${i + 1}`,
        items: d.map((s) => ({ name: s.name, detail: `${s.sets} sets`, muscle: "", cue: "", sources: 0 })),
      }));

  return (
    <div>
      {plan.shortfalls.length > 0 && (
        <div className="mb-5">
          <Notice tone="caution" title="Short of the weekly minimum">
            {plan.shortfalls.map((s) => `${cap(s.group)}: ${s.planned.toFixed(1)} of ${s.floor} sets`).join(" · ")}. The minimum is not lowered; these
            groups are named so you can add a day or equipment.
          </Notice>
        </div>
      )}
      <div className="grid gap-4 sm:grid-cols-2">
        {days.map((d, i) => (
          <motion.article
            key={i}
            initial={reduce ? false : { opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-40px" }}
            transition={{ duration: 0.5, delay: (i % 2) * 0.08, ease: [0.22, 1, 0.36, 1] }}
            className="rounded-2xl border border-line bg-card/60 p-5"
          >
            <p className="eyebrow">Day {i + 1}</p>
            <h3 className="display mt-2 text-2xl">{d.label}</h3>
            <ul className="mt-4 space-y-4">
              {d.items.map((e, j) => (
                <li key={j} className="border-t border-line-soft pt-3 first:border-0 first:pt-0">
                  <div className="flex items-baseline justify-between gap-3">
                    <p className="text-ink">{e.name}</p>
                    <p className="shrink-0 text-sm tabular-nums text-accent-2">{e.detail}</p>
                  </div>
                  {e.muscle && <p className="mt-0.5 text-xs text-ink-3">{cap(e.muscle)}</p>}
                  {e.cue && <p className="mt-1 text-sm leading-relaxed text-ink-2">{e.cue}</p>}
                  {e.sources > 0 && <p className="mt-1 text-xs text-ink-4">{e.sources} source{e.sources === 1 ? "" : "s"}</p>}
                </li>
              ))}
            </ul>
          </motion.article>
        ))}
      </div>
      <div className="mt-5 flex flex-wrap gap-2">
        {Object.entries(plan.weekly).map(([g, sets]) => (
          <span key={g} className="rounded-full border border-line px-3 py-1 text-xs text-ink-2">
            {cap(g)} <span className="tabular-nums text-accent-2">{sets}</span>
          </span>
        ))}
      </div>
      <p className="mt-2 text-xs text-ink-3">Weekly hard sets per muscle group.</p>
    </div>
  );
}
