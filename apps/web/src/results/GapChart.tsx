import { motion, useReducedMotion } from "motion/react";
import type { GapProfile, RegionGap } from "../lib/types.gen";
import { regionLabel } from "./answers";

/** Confidence meter: three segments, split at thirds (ours, untuned; the number is in the tooltip). */
const tier = (c: number) => (c >= 0.66 ? 3 : c >= 0.33 ? 2 : 1);

function Meter({ confidence }: { confidence: number }) {
  const n = tier(confidence);
  return (
    <span className="flex items-center gap-1" title={`Confidence ${confidence.toFixed(2)}`} aria-label={`Confidence ${["low", "medium", "high"][n - 1]}`}>
      {[1, 2, 3].map((i) => (
        <span key={i} className={`h-1 w-4 rounded-full ${i <= n ? "bg-accent" : "bg-line"}`} />
      ))}
    </span>
  );
}

function Bar({ pct, className, delay }: { pct: number; className: string; delay: number }) {
  const reduce = useReducedMotion();
  return (
    <div className="h-1.5 w-full rounded-full bg-line-soft">
      <motion.div
        className={`h-full rounded-full ${className}`}
        initial={reduce ? false : { width: 0 }}
        whileInView={{ width: `${pct}%` }}
        viewport={{ once: true }}
        transition={{ duration: 0.9, delay, ease: [0.22, 1, 0.36, 1] }}
        style={reduce ? { width: `${pct}%` } : undefined}
      />
    </div>
  );
}

function gapText(delta: number, heightCm: number): { big: string; small: string } {
  const cm = delta * heightCm;
  if (Math.abs(cm) < 1) return { big: "On target", small: "within about 1 cm" };
  return { big: `${cm > 0 ? "+" : "−"}${Math.abs(cm).toFixed(1)} cm`, small: cm > 0 ? "goal is larger" : "goal is smaller" };
}

function Row({ k, r, heightCm, priority, i }: { k: string; r: RegionGap; heightCm: number; priority: boolean; i: number }) {
  if (r.measured === false)
    return (
      <li className="grid gap-1 py-4 md:grid-cols-[10rem_1fr]">
        <p className="text-ink-2">{regionLabel(k)}</p>
        <p className="text-sm text-ink-3">{r.reason ?? "Not measured."}</p>
      </li>
    );
  const g = r.goal_ratio;
  const c = r.current_ratio;
  const max = typeof g === "number" && typeof c === "number" ? Math.max(g, c) * 1.1 : null;
  const t = gapText(r.delta, heightCm);
  return (
    <li className="grid items-center gap-3 py-4 md:grid-cols-[10rem_1fr_7.5rem]">
      <div>
        <p className="flex items-center gap-2 text-ink">
          {regionLabel(k)}
          {priority && <span className="rounded-full border border-accent/50 px-2 py-0.5 text-[0.6rem] uppercase tracking-[0.14em] text-accent-2">Priority</span>}
        </p>
        <div className="mt-1.5">
          <Meter confidence={r.confidence} />
        </div>
      </div>
      {max ? (
        <div className="space-y-1.5" aria-hidden>
          <Bar pct={((c as number) / max) * 100} className="bg-ink-3" delay={i * 0.05} />
          <Bar pct={((g as number) / max) * 100} className="bg-accent" delay={i * 0.05 + 0.1} />
        </div>
      ) : (
        <p className="text-sm text-ink-3">Bars are not available for this region.</p>
      )}
      <div className="md:text-right">
        <p className="display text-2xl text-accent-2">{t.big}</p>
        <p className="text-xs text-ink-3">{t.small}</p>
      </div>
    </li>
  );
}

/** Per region: you (grey) and your goal (gold) on one axis, with the gap in centimetres at your height. */
export default function GapChart({ gap, heightCm, priorities }: { gap: GapProfile; heightCm: number; priorities: string[] }) {
  const rows = Object.entries(gap.regions).sort(([ka, a], [kb, b]) => {
    const pa = priorities.indexOf(ka);
    const pb = priorities.indexOf(kb);
    if (pa !== pb) return (pa === -1 ? 99 : pa) - (pb === -1 ? 99 : pb);
    if ((a.measured === false) !== (b.measured === false)) return a.measured === false ? 1 : -1;
    return Math.abs(b.delta) - Math.abs(a.delta);
  });
  return (
    <div>
      <div className="flex flex-wrap items-center gap-5 text-xs text-ink-3">
        <span className="flex items-center gap-2">
          <span className="h-1.5 w-6 rounded-full bg-ink-3" /> You
        </span>
        <span className="flex items-center gap-2">
          <span className="h-1.5 w-6 rounded-full bg-accent" /> Your goal
        </span>
        <span>Each row is scaled to itself. Gaps carry several cm of noise.</span>
      </div>
      <ul className="mt-2 divide-y divide-line-soft">
        {rows.map(([k, r], i) => (
          <Row key={k} k={k} r={r} heightCm={heightCm} priority={priorities.includes(k)} i={i} />
        ))}
      </ul>
    </div>
  );
}
