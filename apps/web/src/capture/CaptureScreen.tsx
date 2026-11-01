import { useRef, useState } from "react";
import { VIEWS, type ViewName } from "./types";
import { useTurnCapture, type Views } from "./useTurnCapture";

const PROMPT: Record<ViewName, string> = {
  front: "Face the camera, arms slightly away from your body. Hold still.",
  left: "Turn slowly to your right until your LEFT side faces the camera. Hold still.",
  back: "Keep turning until your BACK faces the camera. Hold still.",
  right: "Keep turning until your RIGHT side faces the camera. Hold still.",
};

interface Props {
  views: Views;
  retake?: ViewName;
  onDone: (views: Views, turnMs: number) => void;
  onUpload?: () => void;
}

export default function CaptureScreen({ views: initial, retake, onDone, onUpload }: Props) {
  const video = useRef<HTMLVideoElement>(null);
  const preview = useRef<HTMLCanvasElement>(null);
  const [showFeed, setShowFeed] = useState(false); // silhouette by default: you watch a shadow, not yourself
  const [showDebug, setShowDebug] = useState(false);
  const cap = useTurnCapture(video, preview, showFeed, initial);
  const busy = cap.phase === "loading" || cap.phase === "countdown" || cap.phase === "turning";

  return (
    <section className="mx-auto max-w-2xl p-4" aria-labelledby="cap-h">
      <h1 id="cap-h" className="text-xl font-bold">{retake ? `Retake: ${retake}` : "Scan"}</h1>

      {cap.phase === "idle" && (
        <p className="mt-2 text-gray-600">
          Prop your device 2-3 m away so your whole body fits. Press Start, then follow the prompts: you will turn one
          quarter at a time and it captures by itself when you hold still. Four views: front, left, back, right.
        </p>
      )}

      <div className="relative mx-auto mt-3 w-fit max-w-full overflow-hidden rounded bg-neutral-900">
        <video ref={video} playsInline muted className="absolute h-0 w-0 opacity-0" />
        {/* mirrored for display only; processing uses the unmirrored frame */}
        <canvas ref={preview} className="block max-h-[65vh] min-h-64 max-w-full -scale-x-100" />
        <svg viewBox="0 0 100 200" className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden>
          <g fill="none" stroke="white" strokeOpacity={0.4} strokeWidth={0.8} strokeDasharray="3 3">
            <circle cx="50" cy="28" r="10" />
            <path d="M50 38 V100 M50 48 L22 92 M50 48 L78 92 M50 100 L36 188 M50 100 L64 188" />
          </g>
        </svg>

        {cap.phase === "countdown" && (
          <div className="absolute inset-0 grid place-items-center bg-black/30 text-center text-white" aria-live="assertive">
            <div>
              <p className="text-7xl font-bold">{cap.count}</p>
              <p className="mt-2 px-4 text-lg">Get in position: whole body in frame, facing the camera</p>
            </div>
          </div>
        )}
        {cap.phase === "turning" && cap.stage && !cap.flash && (
          <div className="absolute inset-x-0 top-0 bg-black/70 p-3 text-center text-lg font-semibold text-white" aria-live="assertive">
            {PROMPT[cap.stage]}
          </div>
        )}
        {cap.flash && (
          <div className="absolute inset-0 grid place-items-center bg-green-600/60 text-3xl font-bold capitalize text-white" aria-live="assertive">
            ✓ {cap.flash} captured
          </div>
        )}
        {cap.phase === "turning" && (
          <div className="absolute inset-x-0 bottom-0 h-2 bg-white/20" role="progressbar" aria-valuenow={Math.round(cap.holdPct * 100)} aria-label="Hold still">
            <div className="h-full bg-green-400" style={{ width: `${cap.holdPct * 100}%` }} />
          </div>
        )}
      </div>

      <div className="mt-3 min-h-8" aria-live="polite">
        {cap.phase === "loading" && <p>Loading the on-device model…</p>}
        {cap.hint && <p className="text-amber-700">{cap.hint}</p>}
        {cap.phase === "error" && <p className="text-red-700">Could not start: {cap.error}</p>}
        {cap.phase === "done" && <p className="text-green-700">Done. Press Review to check the silhouettes.</p>}
      </div>

      {cap.phase === "turning" && !retake && (
        <ol className="mt-1 flex gap-1" aria-label="Progress">
          {VIEWS.map((v) => (
            <li key={v} className={`h-1.5 flex-1 rounded ${cap.views[v] || VIEWS.indexOf(v) < VIEWS.indexOf(cap.stage ?? "front") ? "bg-green-600" : v === cap.stage ? "bg-black" : "bg-neutral-300"}`} />
          ))}
        </ol>
      )}

      <div className="mt-3 flex flex-wrap items-center gap-3">
        {!busy && (
          <button className="rounded bg-black px-4 py-2 text-white" onClick={() => cap.phase === "done" ? onDone(cap.views, cap.turnMs) : cap.start(retake)}>
            {cap.phase === "done" ? "Review" : retake ? "Start retake" : "Start"}
          </button>
        )}
        {cap.phase === "turning" && <button className="rounded bg-green-700 px-4 py-2 text-white" onClick={cap.captureNow}>Capture now</button>}
        {busy && <button className="rounded border px-4 py-2" onClick={cap.stop}>Cancel</button>}
        {cap.phase === "done" && <button className="rounded border px-4 py-2" onClick={() => cap.start(retake)}>Redo</button>}
        {cap.phase === "idle" && onUpload && <button className="text-sm underline" onClick={onUpload}>Upload photos instead</button>}
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={showFeed} onChange={(e) => setShowFeed(e.target.checked)} />
          Show camera feed (for alignment)
        </label>
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={showDebug} onChange={(e) => setShowDebug(e.target.checked)} />
          Debug
        </label>
      </div>

      {showDebug && cap.debug && (
        <pre className="mt-2 rounded bg-neutral-100 p-2 text-xs">
          {`view=${cap.stage ?? "-"}  match=${cap.debug.match}\nyaw=${cap.debug.yaw}° yawZ=${cap.debug.yawZ}° (want ${cap.stage === "left" || cap.stage === "right" ? "yaw>=72" : "yaw<=25"})  facing=${cap.debug.facing}  side=${cap.debug.side}\nratio=${cap.debug.ratio.toFixed(3)}  ref=${cap.debug.ref.toFixed(3)}  body=${cap.debug.problems}`}
        </pre>
      )}
    </section>
  );
}
