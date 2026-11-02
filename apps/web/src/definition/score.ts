import { rgbStore } from "../capture/rgbStore";
import { frontTensor } from "./preprocess";

/** One graph's runner: the preprocessed front tensor in, its 4 CORAL logits out (ADR-053 Amendment 6). */
export type Runner = (x: Float32Array) => Promise<ArrayLike<number>>;
/** definition: the 8-head model's overall head. adiposity: the 2-head model's adiposity head. Nothing else is exported. */
export interface Runners {
  definition: Runner;
  adiposity: Runner;
}
export interface Scores {
  definition: number; // class 1-5
  adiposity: number; // class 1-5
}

/** CORAL: the class is 1 + the number of logits above 0. */
export const classOf = (logits: ArrayLike<number>): number => 1 + Array.from(logits).filter((v) => v > 0).length;

/** Both graphs read the same tensor: definition from its model, adiposity from its own. */
export async function scoreTensor(runners: Runners, x: Float32Array): Promise<Scores> {
  const definition = classOf(await runners.definition(x));
  const adiposity = classOf(await runners.adiposity(x));
  return { definition, adiposity };
}

/**
 * Score the stored front frame, then release every frame whatever happens (ADR-051: score first, then clear).
 * null when the model is off (nothing stored), the frame has no person, or loading / scoring failed.
 */
export async function scoreFrontAndClear(load: () => Promise<Runners>): Promise<Scores | null> {
  try {
    const img = rgbStore.get("front");
    const prob = rgbStore.getMask("front");
    const x = img && prob ? frontTensor(img, prob) : null;
    return x ? await scoreTensor(await load(), x) : null;
  } catch {
    return null; // the size gaps do not depend on this score; the app shows no definition and says so
  } finally {
    rgbStore.clear();
  }
}
