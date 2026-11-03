import { AnimatePresence, motion } from "motion/react";
import { useRef, useState } from "react";
import { Heading, Item, Screen } from "../ui/Layout";
import Toggle from "../ui/Toggle";
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

const Corner = ({ className }: { className: string }) => (
  <span className={`pointer-events-none absolute size-5 border-accent ${className}`} aria-hidden />
);

export default function CaptureScreen({ views: initial, retake, onDone, onUpload }: Props) {
  const video = useRef<HTMLVideoElement>(null);
  const preview = useRef<HTMLCanvasElement>(null);
  const [showFeed, setShowFeed] = useState(false); // silhouette by default: you watch a shadow, not yourself
  const [showDebug, setShowDebug] = useState(false);
  const cap = useTurnCapture(video, preview, showFeed, initial);
  const busy = cap.phase === "loading" || cap.phase === "countdown" || cap.phase === "turning";
  const stageIndex = VIEWS.indexOf(cap.stage ?? "front");

  return (
    <Screen labelledBy="cap-h">
      <Heading eyebrow="Step 3 · Scan" title={retake ? `Retake: ${retake}` : "Four slow turns"} id="cap-h">
        {cap.phase === "idle" && (
          <>
            Prop your device 2-3 m away so your whole body fits. Press Start, then follow the prompts: you will turn one
            quarter at a time and it captures by itself when you hold still. Four views: front, left, back, right.
          </>
        )}
      </Heading>

      <Item>
        <div className="relative mx-auto min-h-[26rem] w-fit min-w-[min(100%,20rem)] max-w-full overflow-hidden rounded-2xl border border-line bg-raised shadow-[0_30px_80px_-30px_rgb(0_0_0_/_0.8)]">
          <video ref={video} playsInline muted className="absolute h-0 w-0 opacity-0" />
          {/* mirrored for display only; processing uses the unmirrored frame */}
          <canvas ref={preview} className="block max-h-[65vh] min-h-64 max-w-full -scale-x-100" />
          {(cap.phase === "idle" || cap.phase === "done" || cap.phase === "error") && (
            <div className="absolute inset-0 grid place-items-center bg-raised text-center">
              <p className="px-6 text-sm uppercase tracking-[0.18em] text-ink-3">
                {cap.phase === "done" ? "Scan captured" : "Camera is off"}
              </p>
            </div>
          )}
          <svg viewBox="0 0 100 200" className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden>
            <g fill="none" stroke="#dcc08a" strokeOpacity={0.45} strokeWidth={0.7} strokeDasharray="2 3">
              <circle cx="50" cy="28" r="10" />
              <path d="M50 38 V100 M50 48 L22 92 M50 48 L78 92 M50 100 L36 188 M50 100 L64 188" />
            </g>
          </svg>
          <Corner className="left-3 top-3 border-l border-t" />
          <Corner className="right-3 top-3 border-r border-t" />
          <Corner className="bottom-3 left-3 border-b border-l" />
          <Corner className="bottom-3 right-3 border-b border-r" />

          <AnimatePresence>
            {cap.phase === "countdown" && (
              <motion.div
                key="count"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="absolute inset-0 grid place-items-center bg-canvas/70 text-center backdrop-blur-sm"
                aria-live="assertive"
              >
                <div>
                  <motion.p key={cap.count} initial={{ scale: 1.25, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }} className="display text-[7rem] text-accent-2">
                    {cap.count}
                  </motion.p>
                  <p className="mt-2 px-4 text-lg text-ink">Get in position: whole body in frame, facing the camera</p>
                </div>
              </motion.div>
            )}
            {cap.phase === "turning" && cap.stage && !cap.flash && (
              <motion.div
                key={cap.stage}
                initial={{ opacity: 0, y: -12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="absolute inset-x-0 top-0 bg-canvas/80 p-4 text-center backdrop-blur"
                aria-live="assertive"
              >
                <p className="display text-2xl sm:text-3xl">{PROMPT[cap.stage]}</p>
              </motion.div>
            )}
            {cap.flash && (
              <motion.div
                key="flash"
                initial={{ opacity: 0, scale: 0.96 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0 }}
                className="absolute inset-0 grid place-items-center bg-accent/80 text-on-accent"
                aria-live="assertive"
              >
                <p className="display text-5xl capitalize">✓ {cap.flash} captured</p>
              </motion.div>
            )}
          </AnimatePresence>
          {cap.phase === "turning" && (
            <div className="absolute inset-x-0 bottom-0 h-1.5 bg-line" role="progressbar" aria-valuenow={Math.round(cap.holdPct * 100)} aria-label="Hold still">
              <div className="h-full bg-accent transition-[width] duration-150" style={{ width: `${cap.holdPct * 100}%` }} />
            </div>
          )}
        </div>
      </Item>

      <div className="mt-4 min-h-8 text-center" aria-live="polite">
        {cap.phase === "loading" && <p className="soft-pulse text-ink-2">Loading the on-device model…</p>}
        {cap.hint && <p className="text-caution">{cap.hint}</p>}
        {cap.phase === "error" && <p className="text-[#e58b7b]">Could not start: {cap.error}</p>}
        {cap.phase === "done" && <p className="text-ok">Done. Press Review to check the silhouettes.</p>}
      </div>

      {cap.phase === "turning" && !retake && (
        <ol className="mt-1 grid grid-cols-4 gap-2" aria-label="Progress">
          {VIEWS.map((v, i) => {
            const done = !!cap.views[v] || i < stageIndex;
            const now = v === cap.stage;
            return (
              <li key={v}>
                <div className={`h-[3px] rounded-full transition-colors duration-500 ${done ? "bg-accent" : now ? "soft-pulse bg-accent-2" : "bg-line"}`} />
                <p className={`mt-2 text-[0.68rem] uppercase tracking-[0.16em] ${done || now ? "text-accent-2" : "text-ink-4"}`}>{v}</p>
              </li>
            );
          })}
        </ol>
      )}

      <Item className="mt-5 flex flex-wrap items-center gap-3">
        {!busy && (
          <button className="btn btn-primary" onClick={() => (cap.phase === "done" ? onDone(cap.views, cap.turnMs) : cap.start(retake))}>
            {cap.phase === "done" ? "Review" : retake ? "Start retake" : "Start"}
          </button>
        )}
        {cap.phase === "turning" && (
          <button className="btn btn-ghost" onClick={cap.captureNow}>
            Capture now
          </button>
        )}
        {busy && (
          <button className="btn btn-ghost" onClick={cap.stop}>
            Cancel
          </button>
        )}
        {cap.phase === "done" && (
          <button className="btn btn-ghost" onClick={() => cap.start(retake)}>
            Redo
          </button>
        )}
        {cap.phase === "idle" && onUpload && (
          <button className="text-sm text-accent-2 underline-offset-4 hover:underline" onClick={onUpload}>
            Upload photos instead
          </button>
        )}
      </Item>

      <Item className="mt-5 flex flex-wrap gap-x-7 gap-y-3">
        <Toggle checked={showFeed} onChange={setShowFeed} label="Show camera feed (for alignment)" />
        <Toggle checked={showDebug} onChange={setShowDebug} label="Debug" />
      </Item>

      {showDebug && cap.debug && (
        <pre className="mt-3 overflow-x-auto rounded-xl border border-line bg-raised p-3 text-xs text-ink-2">
          {`view=${cap.stage ?? "-"}  match=${cap.debug.match}\nyaw=${cap.debug.yaw}° yawZ=${cap.debug.yawZ}° (want ${cap.stage === "left" || cap.stage === "right" ? "yaw>=72" : "yaw<=25"})  facing=${cap.debug.facing}  side=${cap.debug.side}\nratio=${cap.debug.ratio.toFixed(3)}  ref=${cap.debug.ref.toFixed(3)}  body=${cap.debug.problems}`}
        </pre>
      )}
    </Screen>
  );
}
