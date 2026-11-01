import { buildScan } from "../lib/scan";
import { VIEWS, type ViewName } from "./types";
import type { Views } from "./useTurnCapture";

interface Props {
  views: Views;
  turnMs: number;
  onRetake: (view: ViewName) => void;
  onRestart: () => void;
}

export default function ReviewScreen({ views, turnMs, onRetake, onRestart }: Props) {
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

      <p className={`mt-4 ${scanError ? "text-red-700" : "text-green-700"}`}>
        {scanError ? `Scan not ready: ${scanError}` : `Scan ready: ${usable.length}/4 valid views`}
      </p>
      <button className="mt-3 rounded border px-4 py-2" onClick={onRestart}>Start over</button>
    </section>
  );
}
