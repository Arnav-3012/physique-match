import * as ort from "onnxruntime-web/wasm";
import { INPUT_H, INPUT_W } from "./preprocess";
import type { Runner, Runners } from "./score";

/**
 * The two fp32 ONNX graphs (ADR-053 Amendment 6), WASM provider only, loaded in parallel and kept for the session.
 * The runtime files are self-hosted (`npm run ort` copies them to public/ort): nothing is fetched from a CDN.
 */
const BASE = import.meta.env.BASE_URL;
// Absolute on purpose: Vite's dev server appends `?import` to root-relative dynamic imports and then refuses files in public/.
ort.env.wasm.wasmPaths = new URL(`${BASE}ort/`, self.location.href).href;
const URL_OF = { definition: `${BASE}models/definition.fp32.onnx`, adiposity: `${BASE}models/adiposity.fp32.onnx` };

const runner = (s: ort.InferenceSession): Runner => async (x) =>
  (await s.run({ x: new ort.Tensor("float32", x, [1, 3, INPUT_H, INPUT_W]) })).logits.data as Float32Array;

let sessions: Promise<[ort.InferenceSession, ort.InferenceSession]> | undefined;

/** Idempotent: call it when the opt-in is switched on to start loading, and again to score. */
export function loadRunners(): Promise<Runners> {
  sessions ??= Promise.all(
    [URL_OF.definition, URL_OF.adiposity].map((url) => ort.InferenceSession.create(url, { executionProviders: ["wasm"] })),
  ) as Promise<[ort.InferenceSession, ort.InferenceSession]>;
  sessions.catch(() => (sessions = undefined)); // a failed load can be retried
  return sessions.then(([d, a]) => ({ definition: runner(d), adiposity: runner(a) }));
}

export async function releaseRunners(): Promise<void> {
  const loaded = sessions;
  sessions = undefined;
  if (loaded) await Promise.all((await loaded).map((s) => s.release()));
}
