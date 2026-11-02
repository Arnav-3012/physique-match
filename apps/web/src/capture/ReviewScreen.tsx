import DefinitionBands from "../definition/DefinitionBands";
import type { Scores } from "../definition/score";
import { buildScan } from "../lib/scan";
import type { Scope } from "./scope";
import { torsoLooksCovered } from "./torso";
import { VIEWS, type ViewName } from "./types";
import type { Views } from "./useTurnCapture";

interface Props {
  views: Views;
  turnMs: number;
  scopeNote?: string; // what the user chose to compare (ADR-051)
  scope?: Scope;
  scoreWanted?: boolean; // the user opted in to the on-device estimate
  scores?: Scores | null; // held in memory since capture; null = not scored
  torsoConfirmedBare?: boolean;
  onConfirmBare?: () => void;
  onChangeScope?: () => void;
  onRetake: (view: ViewName) => void;
  onRestart: () => void;
  onContinue?: () => void;
}

export default function ReviewScreen({
  views, turnMs, scopeNote, scope, scoreWanted, scores, torsoConfirmedBare, onConfirmBare, onChangeScope, onRetake,
  onRestart, onContinue,
}: Props) {
  const covered = torsoLooksCovered(scope, views.front?.torsoSkinShare, torsoConfirmedBare);
  const usable = VIEWS.flatMap((v) => (views[v] && !views[v].issues.length ? [views[v]] : []));
  let scanError = "";
  try {
    buildScan(usable);
  } catch (e) {
    scanError = e instanceof Error ? e.message : String(e);
  }

  return (
    <section className="mx-auto max-w-2xl p-4" aria-labelledby="rev-h">
      <h1 id="rev-h" className="text-xl font-bold">What we would send</h1>
      <p className="text-gray-600">
        Exactly these outlines (head removed) plus 33 joint positions per view. No photo, no face, no camera feed.
        {turnMs > 0 && ` Turn took ${(turnMs / 1000).toFixed(1)} s.`}
      </p>

      {scopeNote && <p className="mt-1 text-gray-600">{scopeNote}</p>}

      {covered && (
        <div role="alert" className="mt-3 rounded border border-amber-600 bg-amber-50 p-3 text-amber-900">
          <p className="font-semibold">Your torso looks covered</p>
          <p>
            You chose Full body, which needs a bare torso (or a sports bra). With a top on, the leanness &amp; muscularity
            estimate would be meaningless, so it will not be scored. Size gaps still work. This check is a rough guess
            from colour and can be wrong.
          </p>
          <div className="mt-2 flex flex-wrap gap-3">
            <button className="rounded bg-black px-3 py-1 text-white" onClick={() => onRetake("front")}>Retake the front view</button>
            {onChangeScope && <button className="rounded border px-3 py-1" onClick={onChangeScope}>Choose a different option</button>}
            {onConfirmBare && <button className="rounded border px-3 py-1" onClick={onConfirmBare}>My torso is bare, score it</button>}
          </div>
        </div>
      )}

      <ul className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
        {VIEWS.map((name) => {
          const v = views[name];
          const ok = !!v && !v.issues.length;
          return (
            <li key={name} className="rounded border p-2">
              <p className="font-semibold capitalize">{name} {v && <span aria-label={ok ? "valid" : "invalid"}>{ok ? "✓" : "✗"}</span>}</p>
              {v?.maskPngB64 ? (
                <img src={`data:image/png;base64,${v.maskPngB64}`} alt={`${name} silhouette`} className="mt-1 aspect-[3/4] w-full bg-black object-contain" />
              ) : (
                <div className="mt-1 grid aspect-[3/4] place-items-center bg-neutral-200 text-sm text-gray-600">missing</div>
              )}
              {v && <p className="mt-1 text-xs text-gray-600">{Math.round((v.maskPngB64.length * 0.75) / 1024)} KB · sharp {Math.round(v.sharpness)}</p>}
              {import.meta.env.DEV && v?.torsoSkinShare != null && (
                <p className="text-xs text-gray-500">dev: torso skin share {v.torsoSkinShare.toFixed(2)}</p>
              )}
              {import.meta.env.DEV && v?.maskPngB64 && (
                <a className="text-xs underline" download={`${name}.png`} href={`data:image/png;base64,${v.maskPngB64}`}>
                  dev: save mask
                </a>
              )}
              {[...(v?.issues ?? []), ...(v?.warnings ?? [])].map((m) => (
                <p key={m} className="text-xs text-amber-700">{m}</p>
              ))}
              <button className="mt-1 text-sm underline" onClick={() => onRetake(name)}>Retake</button>
            </li>
          );
        })}
      </ul>

      {scoreWanted && !covered && (scores ? <DefinitionBands scores={scores} /> : <p className="mt-4 text-sm text-gray-600">Not scored on this device.</p>)}

      <p className={`mt-4 ${scanError ? "text-red-700" : "text-green-700"}`}>
        {scanError ? `Scan not ready: ${scanError}` : `Scan ready: ${usable.length}/4 valid views`}
      </p>
      <div className="mt-3 flex gap-3">
        {onContinue && (
          <button className="rounded bg-black px-4 py-2 text-white disabled:opacity-40" disabled={!!scanError} onClick={onContinue}>
            Continue to your goal
          </button>
        )}
        <button className="rounded border px-4 py-2" onClick={onRestart}>Start over</button>
      </div>
    </section>
  );
}
