import { useEffect, useState } from "react";
import CaptureScreen from "./capture/CaptureScreen";
import Consent from "./capture/Consent";
import ReviewScreen from "./capture/ReviewScreen";
import { rgbStore } from "./capture/rgbStore";
import Stepper from "./Stepper";
import { chestApproximate, SCOPE_TEXT, type ScopeChoice, type Sex } from "./capture/scope";
import ScopeStep from "./capture/ScopeStep";
import UploadCapture from "./capture/UploadCapture";
import GoalPhoto, { type GoalViews } from "./capture/GoalPhoto";
import { torsoLooksCovered } from "./capture/torso";
import { VIEWS, type ViewName } from "./capture/types";
import type { Views } from "./capture/useTurnCapture";
import { getHealth, type Health } from "./lib/api";
import GoalInputs, { type GoalChoices } from "./lib/GoalInputsPanel";
import { emptyForm, type ProfileForm } from "./lib/profile";
import { buildScan } from "./lib/scan";
import type { Scan, UserProfile, ViewData } from "./lib/types.gen";
import ProfileScreen from "./profile/ProfileScreen";
import ResultsScreen from "./results/ResultsScreen";
import DevPreview, { withDiag, type Diag } from "./definition/DevPreview";
import { scoreFrontAndClear, type Scores } from "./definition/score";

const WAKING_AFTER_MS = 3000; // free HF Spaces sleep; first request can take 30-60 s

type Step = "consent" | "scope" | "capture" | "upload" | "review" | "goal" | "profile" | "results";
const STEP_INDEX: Record<Step, number> = { consent: 0, scope: 1, capture: 2, upload: 2, review: 3, goal: 4, profile: 5, results: 6 };

/** The goal scan: a front (required) and optionally one side; a front-only goal is valid (ADR-025). */
const goalScan = (g: GoalViews): Scan => ({
  views: Object.fromEntries(
    [g.front, g.side].flatMap((v) => (v ? [[v.name, { mask_png_b64: v.maskPngB64, keypoints: v.keypoints as ViewData["keypoints"] }]] : [])),
  ),
});

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
  // Everything below lives in memory only and is cleared on restart (privacy-and-safety.md: no storage).
  const [goalViews, setGoalViews] = useState<GoalViews>({});
  const [goalSex, setGoalSex] = useState<Sex>(); // asked, never defaulted (ADR-037 Amendment 3)
  const [goalScores, setGoalScores] = useState<Scores | null>(null); // front photo, browser memory only
  const [goalBack, setGoalBack] = useState<number | null>(null); // experimental back class from the goal's back photo (ADR-061)
  const [goalChoices, setGoalChoices] = useState<GoalChoices>({ highPriority: [] });
  const [form, setForm] = useState<ProfileForm>(emptyForm());
  const [profile, setProfile] = useState<UserProfile | null>(null);

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
    setGoalViews({});
    setGoalSex(undefined);
    setGoalScores(null);
    setGoalBack(null);
    setGoalChoices({ highPriority: [] });
    setForm(emptyForm());
    setProfile(null);
    setStep("capture");
  };

  const scopeNote = choice
    ? `Scan option: ${SCOPE_TEXT[choice.scope].title}${chestApproximate(choice.scope, choice.torso) ? " (chest approximate)" : ""}.`
    : undefined;

  return (
    <main className="font-sans pb-16">
      <Stepper current={STEP_INDEX[step]} />
      {step === "consent" &&<Consent onAccept={() => setStep("scope")} />}
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
      {step === "goal" && (
        <GoalInputs
          sex={choice?.sex ?? "not_specified"}
          onChange={setGoalChoices}
          onBack={() => setStep("review")}
          top={
            <GoalPhoto
              views={goalViews}
              onViews={setGoalViews}
              goalSex={goalSex}
              onGoalSex={setGoalSex}
              scoreWanted={choice?.scope === "full_body" && !!choice.scoreOnDevice}
              scope={choice?.scope}
              onGoalScores={setGoalScores}
              onGoalBack={setGoalBack}
            />
          }
          canContinue={!!goalViews.front?.maskPngB64 && goalSex !== undefined}
          onContinue={() => {
            setForm((f) => (f.age ? f : { ...f, age: choice?.age === undefined ? "" : String(choice.age) }));
            setStep("profile");
          }}
        />
      )}
      {step === "profile" && (
        <ProfileScreen
          form={form}
          onForm={setForm}
          sex={choice?.sex ?? "not_specified"}
          onBack={() => setStep("goal")}
          onSubmit={(p) => {
            setProfile(p);
            setStep("results");
          }}
        />
      )}
      {step === "results" && profile && goalViews.front && goalSex && (
        <ResultsScreen
          current={buildScan(VIEWS.flatMap((v) => (views[v] && !views[v].issues.length ? [views[v]] : [])))}
          goal={goalScan(goalViews)}
          profile={profile}
          goalSex={goalSex}
          scope={choice?.scope ?? "none"}
          goalBand={goalChoices.band}
          focus={goalChoices.highPriority}
          scores={torsoLooksCovered(choice?.scope, views.front?.torsoSkinShare, torsoBare) ? null : scores}
          goalScores={goalScores}
          goalBack={goalBack}
          onBack={() => setStep("profile")}
          onRestart={restart}
        />
      )}

      <footer className="relative z-10 mx-auto max-w-xl p-4 text-sm text-ink-3">
        {import.meta.env.DEV && health ? ( // provider status is a developer detail
          `API: ${health.status} · ${Object.entries(health.providers)
            .map(([name, p]) => `${name}: ${p.status}`)
            .join(", ")}`
        ) : error ? (
          "The server is unreachable right now."
        ) : !health && waking ? (
          "Waking the server… this can take up to a minute."
        ) : null}
      </footer>
    </main>
  );
}
