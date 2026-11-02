/**
 * What the two demo scores are called and how they are shown (ADR-053 Amendment 4, ADR-044 Amendment 1). The model
 * has two heads with classes 1-5; the UI shows three bands, never the numbers, and no per-region claim.
 */
export const DEFINITION_LABEL = "leanness & muscularity (estimate)";
export const ADIPOSITY_LABEL = "adiposity (waist-girth proxy)";
export const BADGE = "experimental, trained on synthetic data";

export type Band = "low" | "mid" | "high";

/** Classes 1-2 low, 3 mid, 4-5 high (my mapping, ADR-053 Amendment 4; the user gave none). */
export function bandOf(cls: number): Band {
  if (!Number.isInteger(cls) || cls < 1 || cls > 5) throw new RangeError(`class must be 1-5, got ${cls}`);
  return cls <= 2 ? "low" : cls === 3 ? "mid" : "high";
}

/** The limits text shown next to the scores. Mirrored word for word in docs/decisions.md (ADR-044 Amendment 1). */
export const LIMITS_COPY =
  "These muscle-definition and adiposity scores are experimental estimates from a model trained only on computer-generated bodies. The training bodies cannot show a very muscular body that also carries a lot of body fat, so if that describes you, expect a high definition score and a low adiposity score. Adiposity is a waist-girth proxy and is least reliable for very muscular bodies. These scores do not choose your goal: cut, recomp or bulk is always your choice.";
