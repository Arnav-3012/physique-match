import { useCallback, useEffect, useRef, useState, type RefObject } from "react";
import { CFG } from "./config";
import { detectPose, getLandmarker } from "./landmarker";
import { drawScaled, finalizeView, paintPreview, scoreFrame, type Candidate } from "./pipeline";
import { checkFullBody } from "./quality";
import { VIEWS, type Kp, type ViewCapture, type ViewName } from "./types";
import { absYawDeg, facingCamera, matchesView, shoulderRatio, visibleSide, zYawDeg } from "./yaw";

export type Phase = "idle" | "loading" | "countdown" | "turning" | "done" | "error";
export type Views = Partial<Record<ViewName, ViewCapture>>;

export interface Debug {
  yaw: number;
  yawZ: number;
  facing: boolean;
  side: string;
  ratio: number;
  ref: number;
  match: boolean;
  problems: string;
}

/**
 * Shoulder ratio when square to the camera: median of recent samples, so one bad frame cannot skew it.
 * Module-level so a retake (new component instance) keeps it; reset at the start of a full scan.
 */
const frontal = { samples: [] as number[], value: 0 };

function learnFrontal(k: Kp[], aspect: number) {
  const r = shoulderRatio(k, aspect);
  if (!(r > 0.4 && r < 2.5)) return; // not a plausible shoulder/torso ratio
  frontal.samples.push(r);
  if (frontal.samples.length > 90) frontal.samples.shift();
  const sorted = [...frontal.samples].sort((a, b) => a - b);
  frontal.value = sorted[Math.floor(sorted.length / 2)];
}

const SLOW_HINT_MS = 6000;
const RETAKE_CALIBRATION_MS = 1500;

const nextFrame = () => new Promise<void>((r) => requestAnimationFrame(() => r()));

function beep() {
  try {
    const ctx = new AudioContext();
    const osc = ctx.createOscillator();
    osc.frequency.value = 880;
    osc.connect(ctx.destination);
    osc.onended = () => void ctx.close();
    osc.start();
    osc.stop(ctx.currentTime + 0.12);
  } catch {
    /* no audio is fine; the visual flash still shows */
  }
}

/**
 * Guided turn, paced by the user: for each view the body angle must match for a hold time,
 * then the sharpest matching frame is kept and the next view is asked for.
 * "Capture now" takes the current frame when the angle estimate is wrong. `only` retakes one view.
 */
export function useTurnCapture(
  video: RefObject<HTMLVideoElement | null>,
  preview: RefObject<HTMLCanvasElement | null>,
  showFeed: boolean,
  initial: Views,
) {
  const [phase, setPhase] = useState<Phase>("idle");
  const [stage, setStage] = useState<ViewName | null>(null);
  const [count, setCount] = useState(0);
  const [hint, setHint] = useState("");
  const [error, setError] = useState("");
  const [views, setViews] = useState<Views>(initial);
  const [turnMs, setTurnMs] = useState(0);
  const [holdPct, setHoldPct] = useState(0);
  const [flash, setFlash] = useState<ViewName | null>(null);
  const [debug, setDebug] = useState<Debug>();
  const showFeedRef = useRef(showFeed);
  const manual = useRef(false);
  const cancel = useRef(() => {});

  useEffect(() => void (showFeedRef.current = showFeed), [showFeed]);
  useEffect(() => () => cancel.current(), []);

  const start = useCallback(
    async (only?: ViewName) => {
      let cancelled = false;
      cancel.current = () => void (cancelled = true);
      const targets: ViewName[] = only ? [only] : [...VIEWS];
      const v = video.current!;
      if (!only) {
        frontal.samples = [];
        frontal.value = 0;
      }
      setPhase("loading");
      setError("");
      setFlash(null);
      let stream: MediaStream | undefined;
      try {
        await getLandmarker();
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        v.srcObject = stream;
        await v.play();

        const frame = async () => {
          await nextFrame();
          const canvas = drawScaled(v, v.videoWidth, v.videoHeight);
          const det = await detectPose(canvas, "VIDEO", performance.now());
          if (preview.current) paintPreview(preview.current, canvas, det, showFeedRef.current);
          return { canvas, det };
        };
        const keepLive = async (ms: number) => {
          const t0 = performance.now();
          while (!cancelled && performance.now() - t0 < ms) await frame();
        };

        // Calibrate the "square to the camera" reference: full scan always, a retake only if none is stored.
        if (!only || frontal.value === 0) {
          const ms = only ? RETAKE_CALIBRATION_MS : CFG.countdownMs;
          setPhase("countdown");
          const t0 = performance.now();
          while (!cancelled && performance.now() - t0 < ms) {
            setCount(Math.ceil((ms - (performance.now() - t0)) / 1000));
            const { canvas, det } = await frame();
            if (det) {
              learnFrontal(det.keypoints, canvas.width / canvas.height);
              setHint(checkFullBody(det.keypoints)[0] ?? "");
            }
          }
        }

        setPhase("turning");
        const turnStart = performance.now();
        const picked: Partial<Record<ViewName, Candidate>> = {};
        for (const name of targets) {
          if (cancelled) break;
          setStage(name);
          setHoldPct(0);
          manual.current = false;
          const hold = name === "left" || name === "right" ? CFG.sideHoldMs : CFG.holdMs;
          const t0 = performance.now();
          let best: Candidate | undefined;
          let since: number | null = null; // start of the current run of matching frames
          let lastMatch = 0;
          for (;;) {
            const { canvas, det } = await frame();
            const now = performance.now();
            if (det) {
              const k = det.keypoints;
              const aspect = canvas.width / canvas.height;
              if (name === "front") learnFrontal(k, aspect);
              const ref = frontal.value;
              const problems = checkFullBody(k);
              const slow = since === null && now - t0 > SLOW_HINT_MS;
              setHint(problems[0] ?? (slow ? "Not detecting this pose - match the prompt, or press Capture now" : ""));
              const match = !problems.length && ref > 0 && matchesView(name, k, aspect, ref);
              setDebug({
                yaw: ref > 0 ? Math.round(absYawDeg(k, aspect, ref)) : -1,
                yawZ: Math.round(zYawDeg(k)),
                facing: facingCamera(k),
                side: visibleSide(k) ?? "-",
                ratio: shoulderRatio(k, aspect),
                ref,
                match,
                problems: problems[0] ?? "ok",
              });
              if (match) {
                lastMatch = now;
                since ??= now;
              } else if (since !== null && now - lastMatch > CFG.matchGraceMs) {
                since = null;
                best = undefined;
              }
              if (match || manual.current) {
                const c = scoreFrame(canvas, det);
                if (!best || c.sharpness > best.sharpness) best = c;
              }
            }
            setHoldPct(since === null ? 0 : Math.min(1, (now - since) / hold));
            if (cancelled || now - t0 >= CFG.viewTimeoutMs) break;
            if (best && (manual.current || (since !== null && now - since >= hold))) break;
          }
          if (best && !cancelled) {
            picked[name] = best;
            setFlash(name);
            beep();
            await keepLive(CFG.flashMs);
            setFlash(null);
          }
        }

        if (cancelled) return;
        const fresh: Views = {};
        for (const name of targets) {
          const c = picked[name];
          if (c) fresh[name] = finalizeView(name, c);
        }
        setViews((prev) => {
          const next = { ...prev, ...fresh };
          for (const name of targets) if (!fresh[name]) delete next[name];
          return next;
        });
        if (!only) setTurnMs(performance.now() - turnStart);
        setStage(null);
        setPhase("done");
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setPhase("error");
      } finally {
        stream?.getTracks().forEach((t) => t.stop());
        v.srcObject = null;
      }
    },
    [video, preview],
  );

  const stop = useCallback(() => {
    cancel.current();
    setPhase("idle");
    setStage(null);
  }, []);

  const captureNow = useCallback(() => void (manual.current = true), []);

  return { phase, stage, count, hint, error, views, turnMs, holdPct, flash, debug, start, stop, captureNow };
}
