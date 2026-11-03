import { motion } from "motion/react";
import DefinitionBands from "../definition/DefinitionBands";
import type { Scores } from "../definition/score";
import { buildScan } from "../lib/scan";
import { Heading, Item, Notice, Screen, useItem } from "../ui/Layout";
import SideFigure from "../ui/SideFigure";
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

function Tile({ name, v, onRetake }: { name: ViewName; v: Views[ViewName]; onRetake: (v: ViewName) => void }) {
  const ok = !!v && !v.issues.length;
  return (
    <motion.li variants={useItem()} className="rounded-2xl border border-line bg-card/60 p-3">
      <div className="flex items-center justify-between px-1">
        <p className="display text-2xl capitalize">{name}</p>
        {v && (
          <span
            className={`grid size-5 place-items-center rounded-full text-[0.7rem] ${ok ? "bg-accent text-on-accent" : "bg-caution text-on-accent"}`}
            aria-label={ok ? "valid" : "invalid"}
          >
            {ok ? "✓" : "!"}
          </span>
        )}
      </div>
      <div className="relative mt-2 aspect-[3/4] overflow-hidden rounded-xl border border-line bg-canvas">
        {v?.maskPngB64 ? (
          <>
            <img src={`data:image/png;base64,${v.maskPngB64}`} alt={`${name} silhouette`} className="h-full w-full object-contain" />
            <span className="scanline absolute inset-x-0 h-px bg-accent shadow-[0_0_12px_2px_rgb(200_168_106_/_0.6)]" aria-hidden />
          </>
        ) : (
          <div className="grid h-full place-items-center text-sm text-ink-3">missing</div>
        )}
      </div>
      {v && (
        <p className="mt-2 px-1 text-xs text-ink-3">
          {Math.round((v.maskPngB64.length * 0.75) / 1024)} KB · sharp {Math.round(v.sharpness)}
        </p>
      )}
      {import.meta.env.DEV && v?.torsoSkinShare != null && <p className="px-1 text-xs text-ink-3">dev: torso skin share {v.torsoSkinShare.toFixed(2)}</p>}
      {import.meta.env.DEV && v?.maskPngB64 && (
        <a className="px-1 text-xs text-accent-2 underline" download={`${name}.png`} href={`data:image/png;base64,${v.maskPngB64}`}>
          dev: save mask
        </a>
      )}
      {[...(v?.issues ?? []), ...(v?.warnings ?? [])].map((m) => (
        <p key={m} className="mt-1 px-1 text-xs text-caution">
          {m}
        </p>
      ))}
      <button className="mt-2 px-1 text-sm text-accent-2 underline-offset-4 hover:underline" onClick={() => onRetake(name)}>
        Retake
      </button>
    </motion.li>
  );
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
    <Screen labelledBy="rev-h" aside={<SideFigure caption="What the server would see is only an outline like this" />}>
      <Heading eyebrow="Step 4 · Review" title="What we would send" id="rev-h">
        Exactly these outlines (head removed) plus 33 joint positions per view. No photo, no face, no camera feed.
        {turnMs > 0 && ` Turn took ${(turnMs / 1000).toFixed(1)} s.`}
        {scopeNote && <span className="mt-2 block text-ink-3">{scopeNote}</span>}
      </Heading>

      {covered && (
        <Item className="mb-5">
          <Notice tone="caution" role="alert" title="Your torso looks covered">
            You chose Full body, which needs a bare torso (or a sports bra). With a top on, the leanness &amp; muscularity
            estimate would be meaningless, so it will not be scored. Size gaps still work. This check is a rough guess
            from colour and can be wrong.
            <div className="mt-4 flex flex-wrap gap-3">
              <button className="btn btn-primary !py-2.5" onClick={() => onRetake("front")}>
                Retake the front view
              </button>
              {onChangeScope && (
                <button className="btn btn-ghost !py-2.5" onClick={onChangeScope}>
                  Choose a different option
                </button>
              )}
              {onConfirmBare && (
                <button className="btn btn-ghost !py-2.5" onClick={onConfirmBare}>
                  My torso is bare, score it
                </button>
              )}
            </div>
          </Notice>
        </Item>
      )}

      <ul className="grid grid-cols-2 gap-4">
        {VIEWS.map((name) => (
          <Tile key={name} name={name} v={views[name]} onRetake={onRetake} />
        ))}
      </ul>

      {scoreWanted && !covered && (
        <Item className="mt-5">{scores ? <DefinitionBands scores={scores} /> : <p className="text-sm text-ink-3">Not scored on this device.</p>}</Item>
      )}

      <Item className="mt-6">
        <p className={`flex items-center gap-3 ${scanError ? "text-[#e58b7b]" : "text-ink"}`}>
          <span className={`size-2 rounded-full ${scanError ? "bg-[#e58b7b]" : "soft-pulse bg-ok"}`} aria-hidden />
          {scanError ? `Scan not ready: ${scanError}` : `Scan ready: ${usable.length}/4 valid views`}
        </p>
      </Item>
      <Item className="mt-5 flex flex-wrap gap-3">
        {onContinue && (
          <button className="btn btn-primary disabled:opacity-40" disabled={!!scanError} onClick={onContinue}>
            Continue to your goal
          </button>
        )}
        <button className="btn btn-ghost" onClick={onRestart}>
          Start over
        </button>
      </Item>
    </Screen>
  );
}
