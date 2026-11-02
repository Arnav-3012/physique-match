import { useEffect, useState } from "react";
import CaptureScreen from "./capture/CaptureScreen";
import Consent from "./capture/Consent";
import ReviewScreen from "./capture/ReviewScreen";
import { rgbStore } from "./capture/rgbStore";
import { chestApproximate, SCOPE_TEXT, type ScopeChoice } from "./capture/scope";
import ScopeStep from "./capture/ScopeStep";
import UploadCapture from "./capture/UploadCapture";
import type { ViewName } from "./capture/types";
import type { Views } from "./capture/useTurnCapture";
import { getHealth, type Health } from "./lib/api";
import GoalInputs from "./lib/GoalInputsPanel";
import DevPreview, { withDiag, type Diag } from "./definition/DevPreview";
import { scoreFrontAndClear, type Scores } from "./definition/score";

const WAKING_AFTER_MS = 3000; // free HF Spaces sleep; first request can take 30-60 s

type Step = "consent" | "scope" | "capture" | "upload" | "review" | "goal";

export default function App() {
  const [step, setStep] = useState<Step>("consent");
  const [choice, setChoice] = useState<ScopeChoice>();
  const [views, setViews] = useState<Views>({});
  const [turnMs, setTurnMs] = useState(0);
  const [retake, setRetake] = useState<ViewName>();
  const [torsoBare, setTorsoBare] = useState(false); // the user says the torso is bare although the check doubts it
  const [scores, setScores] = useState<Scores | null>(null); // memory only; held until restart or a new scan option
  const [diag, setDiag] = useState<Diag | null>(null); // dev only
  const [health, setHealth] = useState<Health>();
  const [error, setError] = useState(false);
  const [waking, setWaking] = useState(false);

  useEffect(() => {
    const ctrl = new AbortController();
    const timer = setTimeout(() => setWaking(true), WAKING_AFTER_MS);
    getHealth(ctrl.signal)
      .then(setHealth)
      .catch(() => !ctrl.signal.aborted && setError(true))
      .finally(() => clearTimeout(timer));
    return () => {
      ctrl.abort();
      clearTimeout(timer);
    };
  }, []);

  // The outlines are made, so the frames are released (ADR-051). With the opt-in on, the front frame is scored first
  // (both graphs load on demand, so the app does not carry the runtime otherwise); then every frame is released.
  const release = () => rgbStore.clear();
  const finish = async () => {
    const hadFront = rgbStore.get("front") !== undefined; // a retake of another view must not wipe the scores
    const load = () => import("./definition/models").then((m) => m.loadRunners());
    const s = await scoreFrontAndClear(import.meta.env.DEV ? withDiag(load, setDiag) : load);
    if (hadFront) setScores(s);
  };

  const restart = () => {
    release();
    setScores(null);
    setDiag(null);
    setViews({});
    setTurnMs(0);
    setRetake(undefined);
    setTorsoBare(false);
    setStep("capture");
  };

  const scopeNote = choice
    ? `Scan option: ${SCOPE_TEXT[choice.scope].title}${chestApproximate(choice.scope, choice.torso) ? " (chest approximate)" : ""}.`
    : undefined;

  return (
    <main className="font-sans">
      {step === "consent" && <Consent onAccept={() => setStep("scope")} />}
      {step === "scope" && (
        <ScopeStep
          onStart={(c) => {
            setChoice(c);
            setScores(null);
            setTorsoBare(false);
            rgbStore.enable(!!c.scoreOnDevice); // off keeps no frame at all (ADR-051)
            if (c.scoreOnDevice) void import("./definition/models").then((m) => m.loadRunners()).catch(() => {}); // start loading
            setStep("capture");
          }}
          onBack={() => setStep("consent")}
        />
      )}
      {step === "capture" && (
        <CaptureScreen
          key={retake ?? "full"}
          views={views}
          retake={retake}
          onDone={async (v, ms) => {
            await finish();
            setViews(v);
            if (!retake) setTurnMs(ms);
            setRetake(undefined);
            setStep("review");
          }}
          onUpload={retake ? undefined : () => setStep("upload")}
        />
      )}
      {step === "upload" && (
        <UploadCapture
          views={views}
          onDone={async (v) => {
            await finish();
            setViews(v);
            setStep("review");
          }}
          onBack={() => setStep("capture")}
        />
      )}
      {step === "review" && (
        <ReviewScreen
          views={views}
          turnMs={turnMs}
          scopeNote={scopeNote}
          scope={choice?.scope}
          scoreWanted={choice?.scope === "full_body" && !!choice.scoreOnDevice}
          scores={scores}
          torsoConfirmedBare={torsoBare}
          onConfirmBare={() => setTorsoBare(true)}
          onChangeScope={() => setStep("scope")}
          onRetake={(v) => {
            if (v === "front") setTorsoBare(false);
            setRetake(v);
            setStep("capture");
          }}
          onRestart={restart}
          onContinue={() => setStep("goal")}
        />
      )}
      {import.meta.env.DEV && step === "review" && diag && <DevPreview diag={diag} />}
      {step === "goal" && <GoalInputs sex={choice?.sex ?? "not_specified"} onBack={() => setStep("review")} />}

      <footer className="mx-auto max-w-xl p-4 text-sm text-gray-500">
        API:{" "}
        {error
          ? "unreachable"
          : health
            ? `${health.status} · ${Object.entries(health.providers)
                .map(([name, p]) => `${name}: ${p.status}`)
                .join(", ")}`
            : waking
              ? "waking server… this can take up to a minute"
              : "checking…"}
      </footer>
    </main>
  );
}
