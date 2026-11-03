import { animate, motion, useMotionValue, useReducedMotion, useTransform, type MotionValue } from "motion/react";
import { useEffect } from "react";

/** Illustrative headless frame that morphs from a "current" shape to a "goal" shape (t = 0 .. 1).
 * Hand-built profile numbers, not a person and not a prediction; the UI says so wherever it is shown. */
const CX = 150;
const OUTER_Y = [42, 66, 100, 150, 205, 262, 330, 420, 500, 590];
const OUTER_CUR = [15, 76, 72, 66, 64, 72, 70, 42, 38, 20];
const OUTER_GOAL = [14, 92, 88, 76, 56, 66, 68, 43, 41, 21];
const INNER_Y = [590, 500, 420, 330, 300];
const INNER_CUR = [6, 8, 7, 4, 0];
const INNER_GOAL = [7, 9, 8, 5, 0];
const ARM = { sx: [76, 90], ex: [98, 114], w: [24, 30] };

type Pt = [number, number];
const mix = (a: number[], b: number[], t: number) => a.map((v, i) => v + (b[i] - v) * t);
const lerp = ([a, b]: number[], t: number) => a + (b - a) * t;

export const shoulderWaist = (t: number) => {
  const o = mix(OUTER_CUR, OUTER_GOAL, t);
  return o[1] / o[4];
};

function smooth(pts: Pt[]): string {
  let d = `M${pts[0][0].toFixed(1)} ${pts[0][1].toFixed(1)}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i - 1] ?? pts[i];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = pts[i + 2] ?? p2;
    const c1 = [p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6];
    const c2 = [p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6];
    d += `C${c1[0].toFixed(1)} ${c1[1].toFixed(1)} ${c2[0].toFixed(1)} ${c2[1].toFixed(1)} ${p2[0].toFixed(1)} ${p2[1].toFixed(1)}`;
  }
  return d + "Z";
}

export function outline(t: number): string {
  const o = mix(OUTER_CUR, OUTER_GOAL, t);
  const n = mix(INNER_CUR, INNER_GOAL, t);
  const right: Pt[] = [
    ...OUTER_Y.map((y, i) => [CX + o[i], y] as Pt),
    [CX + o[9], 598],
    [CX + n[0], 598],
    ...INNER_Y.slice(1).map((y, i) => [CX + n[i + 1], y] as Pt),
  ];
  const left = right.map(([x, y]) => [2 * CX - x, y] as Pt).reverse();
  return smooth([...right, ...left.slice(1)]);
}

const armPath = (t: number, side: 1 | -1) =>
  `M${CX + side * lerp(ARM.sx, t)} 80L${CX + side * lerp(ARM.ex, t)} 300`;

/** Loops current -> goal -> current by itself (hero). Holds the current shape under prefers-reduced-motion. */
export function useAutoMorph(): MotionValue<number> {
  const p = useMotionValue(0);
  const reduce = useReducedMotion();
  useEffect(() => {
    if (reduce) return;
    const controls = animate(p, 1, { duration: 3.6, ease: "easeInOut", repeat: Infinity, repeatType: "reverse", repeatDelay: 1.4 });
    return () => controls.stop();
  }, [p, reduce]);
  return p;
}

export function RatioReadout({ p }: { p: MotionValue<number> }) {
  const text = useTransform(p, (t) => shoulderWaist(t).toFixed(2));
  return <motion.span className="tabular-nums">{text}</motion.span>;
}

const CALLOUTS = [
  { label: "Shoulders", i: 1 },
  { label: "Waist", i: 4 },
  { label: "Thigh", i: 6 },
];

function Callout({ p, label, i }: { p: MotionValue<number>; label: string; i: number }) {
  const x = useTransform(p, (t) => CX + mix(OUTER_CUR, OUTER_GOAL, t)[i]);
  const y = OUTER_Y[i];
  return (
    <g>
      <motion.line x1={x} y1={y} x2={282} y2={y} stroke="#c8a86a" strokeOpacity="0.55" strokeWidth="1" strokeDasharray="2 3" />
      <motion.circle cx={x} cy={y} r="3.2" fill="#c8a86a" />
      <text x="286" y={y + 3.5} fontSize="10" letterSpacing="1.6" fill="#cfc7b9" style={{ textTransform: "uppercase" }}>
        {label}
      </text>
    </g>
  );
}

export function Figure({ p, callouts = false }: { p: MotionValue<number>; callouts?: boolean }) {
  const body = useTransform(p, outline);
  const armR = useTransform(p, (t) => armPath(t, 1));
  const armL = useTransform(p, (t) => armPath(t, -1));
  const armW = useTransform(p, (t) => lerp(ARM.w, t));
  const armInner = useTransform(armW, (w) => w - 2.4);
  return (
    <svg viewBox="0 0 400 620" role="img" aria-label="Illustrative silhouette changing from a current to a goal frame" className="h-full w-full">
      {[armR, armL].map((d, k) => (
        <g key={k} strokeLinecap="round">
          <motion.path d={d} fill="none" stroke="#c8a86a" strokeOpacity="0.85" style={{ strokeWidth: armW }} />
          <motion.path d={d} fill="none" stroke="#0b0a09" style={{ strokeWidth: armInner }} />
        </g>
      ))}
      <motion.path d={body} fill="#0b0a09" stroke="#c8a86a" strokeWidth="1.4" strokeLinejoin="round" />
      {callouts && CALLOUTS.map((c) => <Callout key={c.label} p={p} {...c} />)}
    </svg>
  );
}
