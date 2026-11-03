import { rgbStore } from "../capture/rgbStore";
import type { ViewName } from "../capture/types";
import { frontTensor } from "./preprocess";

/** One graph's runner: the preprocessed frame tensor in, its 4 CORAL logits out (ADR-053 Amendment 6). */
export type Runner = (x: Float32Array) => Promise<ArrayLike<number>>;
/** definition: the 8-head model's overall head. adiposity: the 2-head model's adiposity head. back: the 8-head model's
 * back head (ADR-061), present only when its graph loaded. Nothing else is exported. */
export interface Runners {
  definition: Runner;
  adiposity: Runner;
  back?: Runner;
}
export interface Scores {
  definition: number; // class 1-5, from the front photo
  adiposity: number; // class 1-5, from the front photo
  back?: number; // class 1-5, from the back photo; experimental, it did not beat the overall score (ADR-061)
}

/** CORAL: the class is 1 + the number of logits above 0. */
export const classOf = (logits: ArrayLike<number>): number => 1 + Array.from(logits).filter((v) => v > 0).length;

/** Both front graphs read the same tensor: definition from its model, adiposity from its own. */
export async function scoreTensor(runners: Runners, x: Float32Array): Promise<Scores> {
  const definition = classOf(await runners.definition(x));
  const adiposity = classOf(await runners.adiposity(x));
  return { definition, adiposity };
}

/** The stored frame of a view as a model tensor; null when nothing is stored (the opt-in is off) or no person is found. */
const tensorOf = (view: ViewName): Float32Array | null => {
  const img = rgbStore.get(view);
  const prob = rgbStore.getMask(view);
  return img && prob ? frontTensor(img, prob) : null; // the crop works from the person mask, so it suits any view
};

/**
 * Score the stored front frame (and the back frame, if one is stored and the back graph loaded), then release every
 * frame whatever happens (ADR-051: score first, then clear). null when the model is off (nothing stored), the front
 * frame has no person, or loading / scoring failed. A failed back score never costs the front scores.
 */
export async function scoreFrontAndClear(load: () => Promise<Runners>): Promise<Scores | null> {
  try {
    const front = tensorOf("front");
    if (!front) return null;
    const runners = await load();
    const scores = await scoreTensor(runners, front);
    const back = tensorOf("back");
    if (runners.back && back) {
      try {
        scores.back = classOf(await runners.back(back));
      } catch {
        /* the back score is optional */
      }
    }
    return scores;
  } catch {
    return null; // the size gaps do not depend on this score; the app shows no definition and says so
  } finally {
    rgbStore.clear();
  }
}

/** Score one stored back frame (the goal's back photo), then release every frame. null when it cannot be scored. */
export async function scoreBackAndClear(load: () => Promise<Runners>): Promise<number | null> {
  try {
    const back = tensorOf("back");
    if (!back) return null;
    const runners = await load();
    return runners.back ? classOf(await runners.back(back)) : null;
  } catch {
    return null;
  } finally {
    rgbStore.clear();
  }
}
