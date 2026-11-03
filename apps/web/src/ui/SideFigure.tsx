import { Figure, RatioReadout, useAutoMorph } from "../site/Morph";

/** The landing page's morphing silhouette, framed by slow rings, as a decorative side panel for the /app steps. */
export default function SideFigure({ caption = "From where you are to where you are going" }: { caption?: string }) {
  const p = useAutoMorph();
  return (
    <div className="relative mx-auto w-full max-w-[340px]">
      <svg viewBox="-200 -200 400 400" className="absolute left-1/2 top-[44%] h-[130%] w-[130%] -translate-x-1/2 -translate-y-1/2" aria-hidden>
        <circle r="190" fill="none" stroke="#c8a86a" strokeOpacity="0.12" strokeDasharray="1 9" className="spin-slow" style={{ transformOrigin: "center" }} />
        <circle r="150" fill="none" stroke="#c8a86a" strokeOpacity="0.1" />
        <circle r="110" fill="none" stroke="#c8a86a" strokeOpacity="0.12" strokeDasharray="3 12" className="spin-slow-rev" style={{ transformOrigin: "center" }} />
      </svg>
      <div className="relative h-[480px]">
        <Figure p={p} />
      </div>
      <p className="relative mt-2 text-center text-sm text-ink-3">{caption}</p>
      <p className="relative mt-1 flex items-baseline justify-center gap-2 text-xs text-ink-4">
        shoulder to waist
        <span className="display text-xl text-accent-2">
          <RatioReadout p={p} />
        </span>
        illustration
      </p>
    </div>
  );
}
