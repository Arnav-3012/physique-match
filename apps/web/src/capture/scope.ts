/**
 * Capture scope (ADR-051): what the user wants to compare, asked once before capture. Definition is scored only
 * where the scan shows the body, because the model is trained on bare-body renders. The server derives the
 * per-region status from the same scope (services/api/app/definition_scope.py); keep the two in step.
 */
export const SCOPES = ["arms_shoulders_legs", "full_body", "none"] as const;
export type Scope = (typeof SCOPES)[number];

export const DEFINITION_REGIONS = ["abs", "chest", "shoulders", "arms", "back", "legs"] as const;
export type DefinitionRegion = (typeof DEFINITION_REGIONS)[number];

export type Sex = "male" | "female" | "not_specified";
export type Torso = "bare" | "sports_bra";

export const MIN_ADULT_AGE = 18; // mirrors packages/schemas MIN_ADULT_AGE
export const ADULT_ONLY_LINE = "This option is for adults.";

/**
 * Regions with a score in the demo (ADR-053 Amendment 4): the model has two heads, an overall definition
 * ("leanness & muscularity (estimate)", abdomen-dominated, carried by `abs`) and adiposity. Only the full-body
 * scan shows the abdomen; regional definition is TBD after demo.
 */
export const SCOPE_REGIONS: Record<Scope, readonly DefinitionRegion[]> = {
  none: [],
  arms_shoulders_legs: [],
  full_body: ["abs"],
};

export const SCOPE_TEXT: Record<Scope, { title: string; blurb: string }> = {
  arms_shoulders_legs: {
    title: "Arms, shoulders and legs",
    blurb:
      "Tank top and shorts. Size gaps work as usual. Definition is not scored, because this scan does not show the abdomen. No shirtless scan.",
  },
  full_body: {
    title: "Full body",
    blurb:
      "Adds the overall leanness & muscularity estimate. Men: shirtless and shorts. Women: sports bra and shorts. You can choose a bare torso or a sports bra / crop top.",
  },
  none: {
    title: "Clothed scan",
    blurb: "Fitted clothes. Size gaps and body-fat range only; muscle-definition scores are not given.",
  },
};

/** Whole years, 1-120; anything else is "unset" (never guessed). */
export function parseAge(text: string): number | undefined {
  const n = Number(text.trim());
  return text.trim() !== "" && Number.isInteger(n) && n >= 1 && n <= 120 ? n : undefined;
}

/** The full-body option is for adults only; under 18 or unset it is not offered. The other two always are. */
export function availableScopes(age: number | undefined): Scope[] {
  const adult = age !== undefined && age >= MIN_ADULT_AGE;
  return SCOPES.filter((s) => s !== "full_body" || adult);
}

/** Default torso clothing from the sex field; unset sex has no default, the user picks (never preselected). */
export function defaultTorso(sex: Sex): Torso | undefined {
  return sex === "male" ? "bare" : sex === "female" ? "sports_bra" : undefined;
}

/** A bra covers part of the chest, so the chest estimate is approximate. */
export const chestApproximate = (scope: Scope, torso: Torso | undefined) =>
  scope === "full_body" && torso === "sports_bra";

export function clothingGuidance(scope: Scope, torso: Torso | undefined): string {
  if (scope === "none") return "Fitted clothes: gym wear, a fitted tee, shorts or leggings. You do not need to remove clothes.";
  if (scope === "arms_shoulders_legs") return "Wear a sleeveless or tank top and shorts. You do not need to go shirtless.";
  if (torso === "bare") return "Bare torso and shorts, for the whole 4-view scan.";
  if (torso === "sports_bra") return "Sports bra or crop top and shorts, for the whole 4-view scan.";
  return "Choose a bare torso or a sports bra / crop top below, plus shorts, for the whole 4-view scan.";
}

/** Mirrors packages/schemas DefinitionStatus. `measured` means eligible for a score (rename TBD, ADR-051 Amendment 1). */
export type DefinitionStatus =
  | "measured"
  | "not_measured_covered"
  | "not_measured_age_gate"
  | "not_measured_goal_covers"
  | "not_measured_not_in_demo";
export type RegionDisplay = { kind: "score"; value: number } | { kind: "na"; reason: string };

const NA_FALLBACK = "Not scored."; // the server always sends a reason; this only guards a malformed response

/**
 * The one gate between a status and a number on screen (ADR-051): a score is shown only for an eligible region;
 * every other status shows its reason, whatever score came with it. null = eligible but no score yet.
 */
export function displayRegion(
  status: DefinitionStatus,
  reason: string | undefined,
  score: number | undefined,
): RegionDisplay | null {
  if (status !== "measured") return { kind: "na", reason: reason || NA_FALLBACK };
  return score === undefined ? null : { kind: "score", value: score };
}

/** What stays hidden while sex is "Prefer not to say" (ADR-037 Amendment 4, ADR-052). */
export const SEX_UNSET_HIDES = [
  "body-fat % and the goal body-fat bands",
  "calorie targets",
  "back width (lats)",
  "the V-taper ceiling",
] as const;

export const sexUnsetNote = (sex: Sex | undefined): string | null =>
  sex === "not_specified"
    ? `Without a sex choice these are not shown: ${SEX_UNSET_HIDES.join(", ")}. Size gaps and the training plan still work.`
    : null;

export interface ScopeChoice {
  scope: Scope;
  age: number | undefined;
  sex: Sex;
  torso: Torso | undefined;
  scoreOnDevice?: boolean; // the opt-in for the on-device definition model: full body only, off by default
}
