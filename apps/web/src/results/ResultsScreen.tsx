import { useEffect, useMemo, useState } from "react";
import type { Scope, Sex } from "../capture/scope";
import { ApiError, postAnalyze, postPlan } from "../lib/api";
import { bandText, bandVsCurrent, type Band, type GoalRegion } from "../lib/goalInputs";
import type { AnalyzeResponse, PlanRequest, PlanResponse, Scan, UserProfile } from "../lib/types.gen";
import { Heading, Item, Notice, Panel, Stagger } from "../ui/Layout";
import SideFigure from "../ui/SideFigure";
import type { Scores } from "../definition/score";
import { regionLabel } from "./answers";
import LeannessPanel from "./LeannessPanel";
import Assistant from "./Assistant";
import GapChart from "./GapChart";
import PlanCards from "./PlanCards";

interface Props {
  current: Scan;
  goal: Scan;
  profile: UserProfile;
  goalSex: Sex;
  scope: Scope;
  goalBand?: Band;
  focus?: GoalRegion[]; // regions the user asked to prioritise; empty = the server uses the largest gaps
  scores?: Scores | null; // your on-device leanness estimate (opt-in), browser memory only
  goalScores?: Scores | null;
  goalBack?: number | null; // the goal back photo's experimental class (ADR-061)
  onBack: () => void;
  onRestart: () => void;
}

type State =
  | { phase: "loading" }
  | { phase: "error"; message: string; retryAfter?: number }
  | { phase: "ready"; analysis: AnalyzeResponse; plan: PlanResponse | null; planError?: string };

const DEFAULT_DISCLAIMER = "Not medical advice; see a professional for injuries or health conditions.";

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-2xl border border-line bg-card/60 p-5">
      <p className="eyebrow">{label}</p>
      <p className="display mt-3 text-4xl text-accent-2">{value}</p>
      {sub && <p className="mt-2 text-sm leading-relaxed text-ink-2">{sub}</p>}
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-t border-line-soft py-3 first:border-0">
      <dt className="text-sm text-ink-3">{k}</dt>
      <dd className="text-right text-ink">{v}</dd>
    </div>
  );
}

export default function ResultsScreen({ current, goal, profile, goalSex, scope, goalBand, focus = [], scores, goalScores, goalBack, onBack, onRestart }: Props) {
  const [state, setState] = useState<State>({ phase: "loading" });
  const [attempt, setAttempt] = useState(0);
  const [waking, setWaking] = useState(false);

  useEffect(() => {
    const ctrl = new AbortController();
    setState({ phase: "loading" });
    setWaking(false);
    const timer = setTimeout(() => setWaking(true), 3000);
    (async () => {
      try {
        const analysis = await postAnalyze(
          { goal, current, profile, goal_profile: { sex: goalSex, not_shown: [] }, definition_scope: scope },
          ctrl.signal,
        );
        let plan: PlanResponse | null = null;
        let planError: string | undefined;
        try {
          plan = await postPlan({ profile, gap: analysis.gap, waist_g: analysis.waist_g ?? null, goal_sex: goalSex, focus: focus as PlanRequest["focus"] }, ctrl.signal); // the generator types a capped list as a tuple
        } catch (e) {
          if (ctrl.signal.aborted) return;
          planError = e instanceof ApiError ? e.message : "The plan could not be built just now.";
        }
        setState({ phase: "ready", analysis, plan, planError });
      } catch (e) {
        if (ctrl.signal.aborted) return;
        setState(
          e instanceof ApiError
            ? { phase: "error", message: e.status === 429 ? "Too many requests right now." : e.message, retryAfter: e.retryAfter }
            : { phase: "error", message: "Something went wrong." },
        );
      } finally {
        clearTimeout(timer);
      }
    })();
    return () => {
      ctrl.abort();
      clearTimeout(timer);
    };
    // The inputs are fixed for the life of this screen; a retry bumps `attempt`.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [attempt]);

  const facts = useMemo(
    () => (state.phase === "ready" ? { analysis: state.analysis, plan: state.plan, profile, focus } : null),
    [state, profile, focus],
  );

  return (
    <div className="relative z-10 mx-auto max-w-6xl px-4 pb-16 pt-10 sm:px-6 lg:grid lg:grid-cols-[minmax(0,1fr)_25rem] lg:gap-10">
      <Stagger className="min-w-0">
        <Heading eyebrow="Step 7 · Results" title="Your gap, region by region" id="res-h">
          Where your outline sits against your goal, what your body-fat range looks like, and a week of training built around it.
        </Heading>

        {state.phase === "loading" && (
          <Panel>
            <div className="grid items-center gap-6 sm:grid-cols-[1fr_14rem]">
              <div>
                <h2>Measuring your outlines…</h2>
                <p className="mt-3 text-ink-2">Comparing your four views with your goal photo, then building your plan.</p>
                {waking && <p className="mt-3 text-sm text-ink-3">Waking the server, this can take up to a minute the first time.</p>}
              </div>
              <div className="hidden sm:block">
                <SideFigure caption="" />
              </div>
            </div>
          </Panel>
        )}

        {state.phase === "error" && (
          <Panel>
            <h2>We could not finish</h2>
            <p className="mt-3 text-ink-2">{state.message}</p>
            {state.retryAfter && <p className="mt-2 text-sm text-ink-3">Try again in about {state.retryAfter} seconds.</p>}
            <div className="mt-6 flex flex-wrap gap-3">
              <button className="btn btn-primary" onClick={() => setAttempt((a) => a + 1)}>
                Try again
              </button>
              <button className="btn btn-ghost" onClick={onBack}>
                Back
              </button>
            </div>
          </Panel>
        )}

        {state.phase === "ready" && (
          <Report
            state={state}
            profile={profile}
            goalBand={goalBand}
            scores={scores}
            goalScores={goalScores}
            goalBack={goalBack}
            onBack={onBack}
            onRestart={onRestart}
          />
        )}
      </Stagger>

      <aside className="mt-10 print:hidden lg:mt-0">
        <div className="lg:sticky lg:top-6 lg:h-[calc(100vh-3rem)]">{facts ? <Assistant facts={facts} /> : null}</div>
      </aside>
    </div>
  );
}

function Report({
  state,
  profile,
  goalBand,
  scores,
  goalScores,
  goalBack,
  onBack,
  onRestart,
}: {
  state: Extract<State, { phase: "ready" }>;
  profile: UserProfile;
  goalBand?: Band;
  scores?: Scores | null;
  goalScores?: Scores | null;
  goalBack?: number | null;
  onBack: () => void;
  onRestart: () => void;
}) {
  const { analysis, plan, planError } = state;
  const gap = analysis.gap;
  const bc = plan?.body_comp ?? analysis.body_comp;
  const bf = bc?.body_fat ?? null;
  const priorities: string[] = plan?.priorities ?? [];

  const measured = Object.entries(gap.regions).filter(([, r]) => r.measured !== false);
  const top = measured.sort(([, a], [, b]) => Math.abs(b.delta) - Math.abs(a.delta))[0];
  const topCm = top ? top[1].delta * profile.height_cm : null;
  const vt = gap.v_taper;

  const notes = [
    ...(analysis.warnings ?? []),
    ...(plan?.warnings ?? []),
    ...(plan?.report?.realism_note ? [plan.report.realism_note] : []),
    ...(plan?.body_comp_note ? [plan.body_comp_note] : []),
    ...(bc?.note ? [bc.note] : []),
  ].filter((n, i, a) => n && a.indexOf(n) === i);

  const energy = (bc?.energy ?? []).filter((e) => e.goal === profile.goal || e.goal === "maintain");

  return (
    <>
      <Item className="mb-5 grid gap-4 sm:grid-cols-3">
        <Stat
          label="Biggest gap"
          value={top && topCm !== null ? `${topCm > 0 ? "+" : "−"}${Math.abs(topCm).toFixed(1)} cm` : "—"}
          sub={top ? `${regionLabel(top[0])}, ${topCm! > 0 ? "goal is larger" : "goal is smaller"}` : "No region could be measured."}
        />
        <Stat
          label="Body fat"
          value={bf ? `${bf.low}–${bf.high}%` : "—"}
          sub={
            bf
              ? goalBand
                ? `${bandText(goalBand)}. ${bandVsCurrent(goalBand, { low: bf.low, high: bf.high })}`
                : `${bf.method}, ${bf.confidence} confidence.`
              : "Set your sex to see a body-fat range."
          }
        />
        <Stat
          label="Frame V-taper"
          value={vt && vt.measured !== false && vt.current_index != null && vt.goal_index != null ? `${vt.current_index.toFixed(2)} → ${vt.goal_index.toFixed(2)}` : "—"}
          sub={vt?.ceiling?.zone_text ?? vt?.reason ?? "Estimate, shown for orientation only."}
        />
      </Item>

      <Panel className="mb-5">
        <h2>You, against your goal</h2>
        <p className="mt-2 text-sm text-ink-3">Gaps are shown in centimetres at your height. They are estimates from outlines, not measurements.</p>
        <div className="mt-4">
          <GapChart gap={gap} heightCm={profile.height_cm} priorities={priorities} />
        </div>
      </Panel>

      <LeannessPanel you={scores} goal={goalScores} goalBack={goalBack} />

      {bc && (bf || energy.length > 0 || bc.protein) && (
        <Panel className="mb-5">
          <h2>Body composition</h2>
          <dl className="mt-4">
            {bf && <Row k="Body-fat range" v={`${bf.low}–${bf.high}%  ·  ${bf.method}`} />}
            {bc.rmr_kcal != null && <Row k="Resting energy" v={`${Math.round(bc.rmr_kcal)} kcal`} />}
            {bc.tdee_kcal != null && <Row k="Maintenance" v={`${Math.round(bc.tdee_kcal)} kcal`} />}
            {energy.map((e) => (
              <Row key={e.goal} k={e.goal === "maintain" ? "Maintain" : `Your goal: ${e.goal}`} v={`${e.kcal_low}–${e.kcal_high} kcal  ·  ${e.protein_g_low}–${e.protein_g_high} g protein`} />
            ))}
            {bc.protein && <Row k="Protein, typical range" v={`${bc.protein.g_low}–${bc.protein.g_high} g a day`} />}
          </dl>
          {bc.protein?.text && <p className="mt-3 text-xs leading-relaxed text-ink-3">{bc.protein.text}</p>}
        </Panel>
      )}

      <Panel className="mb-5">
        <h2>Your week</h2>
        <div className="mt-4">
          {plan ? <PlanCards plan={plan} /> : <Notice tone="caution">{planError ?? "The plan could not be built just now."}</Notice>}
        </div>
        {plan?.served_by && plan.served_by !== "deterministic" && <p className="mt-4 text-xs text-ink-3">Wording assisted by an AI writer; every number comes from the planner.</p>}
      </Panel>

      {notes.length > 0 && (
        <Item className="mb-5 space-y-3">
          {notes.map((n) => (
            <Notice key={n}>{n}</Notice>
          ))}
        </Item>
      )}

      <Item>
        <p className="text-xs leading-relaxed text-ink-3">{plan?.report?.disclaimer ?? DEFAULT_DISCLAIMER}</p>
      </Item>
      {import.meta.env.DEV && (
        <Item className="mt-3">
          <p className="font-mono text-xs text-ink-4">
            dev: analyzer={gap.analyzer} · front_only={String(gap.goal_front_only)} · retrieval={plan?.retrieval ?? "-"} · served_by=
            {plan?.served_by ?? "-"} · fallback={plan?.fallback_reason ?? "-"}
          </p>
        </Item>
      )}
      <Item className="mt-6 flex flex-wrap gap-3 print:hidden">
        <button className="btn btn-primary" onClick={() => window.print()}>
          Save as PDF
        </button>
        <button className="btn btn-ghost" onClick={onBack}>
          Back
        </button>
        <button className="btn btn-ghost" onClick={onRestart}>
          Start over
        </button>
      </Item>
    </>
  );
}
