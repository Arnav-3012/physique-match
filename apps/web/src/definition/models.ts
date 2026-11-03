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
const URL_OF = {
  definition: `${BASE}models/definition.fp32.onnx`,
  adiposity: `${BASE}models/adiposity.fp32.onnx`,
  back: `${BASE}models/back.fp32.onnx`, // the 8-head model's back head, experimental (ADR-061); optional
};

const runner = (s: ort.InferenceSession): Runner => async (x) =>
  (await s.run({ x: new ort.Tensor("float32", x, [1, 3, INPUT_H, INPUT_W]) })).logits.data as Float32Array;

const create = (url: string) => ort.InferenceSession.create(url, { executionProviders: ["wasm"] });

type Loaded = [ort.InferenceSession, ort.InferenceSession, ort.InferenceSession | null];
let sessions: Promise<Loaded> | undefined;

/** Idempotent: call it when the opt-in is switched on to start loading, and again to score. The back graph is
 * optional: if it is missing or fails to load, the two required scores still work and there is simply no back score. */
export function loadRunners(): Promise<Runners> {
  sessions ??= Promise.all([create(URL_OF.definition), create(URL_OF.adiposity), create(URL_OF.back).catch(() => null)]);
  sessions.catch(() => (sessions = undefined)); // a failed load of a required graph can be retried
  return sessions.then(([d, a, b]) => ({ definition: runner(d), adiposity: runner(a), ...(b ? { back: runner(b) } : {}) }));
}

export async function releaseRunners(): Promise<void> {
  const loaded = sessions;
  sessions = undefined;
  if (loaded) await Promise.all((await loaded).flatMap((s) => (s ? [s.release()] : [])));
}
