import { useEffect, useState } from "react";
import CaptureScreen from "./capture/CaptureScreen";
import Consent from "./capture/Consent";
import ReviewScreen from "./capture/ReviewScreen";
import { rgbStore } from "./capture/rgbStore";
import UploadCapture from "./capture/UploadCapture";
import type { ViewName } from "./capture/types";
import type { Views } from "./capture/useTurnCapture";
import { getHealth, type Health } from "./lib/api";

const WAKING_AFTER_MS = 3000; // free HF Spaces sleep; first request can take 30-60 s

type Step = "consent" | "capture" | "upload" | "review";

export default function App() {
  const [step, setStep] = useState<Step>("consent");
  const [views, setViews] = useState<Views>({});
  const [turnMs, setTurnMs] = useState(0);
  const [retake, setRetake] = useState<ViewName>();
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

  const restart = () => {
    rgbStore.clear();
    setViews({});
    setTurnMs(0);
    setRetake(undefined);
    setStep("capture");
  };

  return (
    <main className="font-sans">
      {step === "consent" && <Consent onAccept={() => setStep("capture")} />}
      {step === "capture" && (
        <CaptureScreen
          key={retake ?? "full"}
          views={views}
          retake={retake}
          onDone={(v, ms) => {
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
          onDone={(v) => {
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
          onRetake={(v) => {
            setRetake(v);
            setStep("capture");
          }}
          onRestart={restart}
        />
      )}

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
