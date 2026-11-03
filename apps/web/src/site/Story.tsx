import { motion, useMotionValue, useReducedMotion, useScroll, useSpring, useTransform, type MotionValue } from "motion/react";
import { useRef } from "react";
import { Figure, RatioReadout } from "./Morph";

const STAGES = [
  {
    eyebrow: "01 · Where you are",
    title: "Your outline, measured against your own height.",
    text: "Four slow turns in front of your camera give an outline of your current frame. Your photo stays on your device.",
  },
  {
    eyebrow: "02 · The gap",
    title: "Every region gets its own distance, and its own honesty.",
    text: "Set against a goal photo, each region gets a size gap with a confidence tier. Where a gap is more than your frame allows, the report says so.",
  },
  {
    eyebrow: "03 · The plan",
    title: "A week that never drops a region below its minimum.",
    text: "Training follows the gaps you care about, with a cited weekly minimum for every muscle group and the sources shown.",
  },
];

function Caption({ i, progress }: { i: number; progress: MotionValue<number> }) {
  const a = i / 3;
  const b = (i + 1) / 3;
  const opacity = useTransform(progress, [a, a + 0.06, b - 0.06, b], [i === 0 ? 1 : 0, 1, 1, i === 2 ? 1 : 0]);
  const y = useTransform(progress, [a, a + 0.06, b - 0.06, b], [i === 0 ? 0 : 20, 0, 0, i === 2 ? 0 : -20]);
  const s = STAGES[i];
  return (
    <motion.div style={{ opacity, y }} className="absolute inset-x-0 top-0">
      <p className="eyebrow">{s.eyebrow}</p>
      <h3 className="display mt-4 text-4xl sm:text-5xl">{s.title}</h3>
      <p className="mt-5 max-w-md text-ink-2">{s.text}</p>
    </motion.div>
  );
}

const Legend = ({ p }: { p: MotionValue<number> }) => (
  <p className="mt-3 flex items-baseline justify-center gap-3 text-sm text-ink-3">
    Shoulder to waist
    <span className="display text-3xl text-accent-2">
      <RatioReadout p={p} />
    </span>
    <span className="text-xs">illustration, not a person or a prediction</span>
  </p>
);

function Pinned() {
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start start", "end end"] });
  const morph = useTransform(scrollYProgress, [0.08, 0.92], [0, 1]);
  const p = useSpring(morph, { stiffness: 110, damping: 28, mass: 0.4 });
  return (
    <section ref={ref} className="relative h-[340vh]" aria-label="How a result is built">
      <div className="sticky top-0 grid h-screen items-center gap-6 px-4 sm:px-6 md:grid-cols-2 mx-auto max-w-6xl">
        <div className="relative h-72 md:h-80">
          {STAGES.map((_, i) => (
            <Caption key={i} i={i} progress={scrollYProgress} />
          ))}
        </div>
        <div className="flex flex-col items-center">
          <div className="h-[62vh] max-h-[560px] w-full">
            <Figure p={p} callouts />
          </div>
          <Legend p={p} />
        </div>
      </div>
    </section>
  );
}

function Static() {
  const cur = useMotionValue(0);
  const goal = useMotionValue(1);
  return (
    <section className="mx-auto max-w-6xl px-4 py-20 sm:px-6" aria-label="How a result is built">
      <div className="grid gap-10 md:grid-cols-2">
        {[
          ["Current", cur],
          ["Goal", goal],
        ].map(([label, p]) => (
          <div key={label as string} className="text-center">
            <div className="mx-auto h-[420px]">
              <Figure p={p as MotionValue<number>} />
            </div>
            <p className="eyebrow mt-3">{label as string}</p>
          </div>
        ))}
      </div>
      <ol className="mt-12 grid gap-8 md:grid-cols-3">
        {STAGES.map((s) => (
          <li key={s.eyebrow}>
            <p className="eyebrow">{s.eyebrow}</p>
            <h3 className="display mt-3 text-3xl">{s.title}</h3>
            <p className="mt-3 text-ink-2">{s.text}</p>
          </li>
        ))}
      </ol>
    </section>
  );
}

export default function Story() {
  return useReducedMotion() ? <Static /> : <Pinned />;
}
