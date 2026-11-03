import { motion, useReducedMotion, useScroll, useTransform } from "motion/react";

/** Subtle fixed backdrop: slow hairline rings, drifting plus-markers, film grain, a scroll ruler (wide screens).
 * Decoration only (aria-hidden); motion is off under prefers-reduced-motion. */
const MARKS: [x: number, y: number, rate: number][] = [
  [7, 16, 0.25],
  [93, 11, 0.45],
  [14, 46, 0.65],
  [87, 56, 0.3],
  [6, 80, 0.5],
  [95, 88, 0.35],
  [52, 74, 0.2],
];

function Mark({ x, y, rate, reduce }: { x: number; y: number; rate: number; reduce: boolean }) {
  const { scrollY } = useScroll();
  const drift = useTransform(scrollY, [0, 4000], [0, -400 * rate]);
  return (
    <motion.svg
      width="14"
      height="14"
      viewBox="-7 -7 14 14"
      className="absolute"
      style={{ left: `${x}%`, top: `${y}%`, y: reduce ? 0 : drift }}
    >
      <path d="M-6 0H6M0 -6V6" stroke="#c8a86a" strokeOpacity="0.4" strokeWidth="1" />
    </motion.svg>
  );
}

function Ring({ r, dash, className }: { r: number; dash?: string; className?: string }) {
  return (
    <svg viewBox="-800 -800 1600 1600" className={`absolute inset-0 h-full w-full ${className ?? ""}`}>
      <circle r={r} fill="none" stroke="#c8a86a" strokeOpacity="0.1" strokeWidth="1" strokeDasharray={dash} />
    </svg>
  );
}

function Ruler() {
  const { scrollYProgress } = useScroll();
  const top = useTransform(scrollYProgress, [0, 1], ["0%", "100%"]);
  return (
    <div className="absolute left-5 top-1/2 hidden h-[38vh] w-3 -translate-y-1/2 lg:block">
      <div
        className="h-full w-full"
        style={{
          backgroundImage:
            "repeating-linear-gradient(to bottom, rgb(244 239 230 / 0.22) 0 1px, transparent 1px 12px), repeating-linear-gradient(to bottom, rgb(244 239 230 / 0.4) 0 1px, transparent 1px 60px)",
          backgroundSize: "6px 100%, 12px 100%",
          backgroundRepeat: "no-repeat",
        }}
      />
      <motion.span className="absolute -left-[3px] size-2 -translate-y-1/2 rotate-45 bg-accent" style={{ top }} />
    </div>
  );
}

export default function Backdrop() {
  const reduce = useReducedMotion() ?? false;
  return (
    <div className="no-print pointer-events-none fixed inset-0 z-0 overflow-hidden" aria-hidden>
      <div className="absolute -right-[22vw] -top-[18vw] h-[85vw] w-[85vw] max-h-[1100px] max-w-[1100px]">
        <Ring r={780} className={reduce ? "" : "spin-slow"} dash="1 9" />
        <Ring r={640} />
        <Ring r={500} className={reduce ? "" : "spin-slow-rev"} dash="3 14" />
        <Ring r={360} />
        <Ring r={220} className={reduce ? "" : "spin-slow"} dash="1 7" />
      </div>
      {MARKS.map(([x, y, rate]) => (
        <Mark key={`${x}-${y}`} x={x} y={y} rate={rate} reduce={reduce} />
      ))}
      <Ruler />
      <div className="grain absolute inset-0" />
    </div>
  );
}
