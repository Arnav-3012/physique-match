/** Demo comparison inputs (ADR-052): goal body-fat band, region priorities, optional self-rating. Client state only. */
import type { Sex } from "../capture/scope";

/** Mirrors services/api/app/bodycomp.py BF_FLOOR: no band below it is ever offered. */
export const BF_FLOOR = { male: 8, female: 18 } as const;

export interface Band {
  label: string;
  low: number;
  high: number;
  lowEndNotSustainable: boolean;
}

/**
 * ACE categories (a fitness-industry chart, read secondhand from a manual page that attributes it to ACE; the official
 * page was not read), clipped at BF_FLOOR. Obese and essential-fat categories are not goals and are not offered.
 */
export const BANDS: Record<"male" | "female", Band[]> = {
  male: [
    { label: "Athletic", low: 8, high: 13, lowEndNotSustainable: true },
    { label: "Fitness", low: 14, high: 17, lowEndNotSustainable: false },
    { label: "Average", low: 18, high: 24, lowEndNotSustainable: false },
  ],
  female: [
    { label: "Athletic", low: 18, high: 20, lowEndNotSustainable: true },
    { label: "Fitness", low: 21, high: 24, lowEndNotSustainable: false },
    { label: "Average", low: 25, high: 31, lowEndNotSustainable: false },
  ],
};

/** How the bands are described on screen. Primary-source verification of the ACE figures is TBD after demo. */
export const BANDS_LABEL = "approximate, fitness-industry categories";

/** Bands for the user's sex; null when sex is unset: body fat needs it (ADR-037 Amendment 4). */
export const bandsFor = (sex: Sex): Band[] | null => (sex === "not_specified" ? null : BANDS[sex]);

export const bandText = (b: Band) =>
  `${b.label} ${b.low}-${b.high}%${b.lowEndNotSustainable ? " (the low end is not sustainable)" : ""}`;

export interface Range {
  low: number;
  high: number;
}

/** The band against the user's current RFM range, by midpoints. */
export function bandVsCurrent(band: Band, current: Range | undefined): string {
  if (!current) return "Compared with your current estimate after the analysis.";
  const diff = Math.round((band.low + band.high) / 2 - (current.low + current.high) / 2);
  if (diff === 0) return `About the same as your current estimate of ${current.low}-${current.high}%.`;
  return `About ${Math.abs(diff)} points ${diff < 0 ? "lower" : "higher"} than your current estimate of ${current.low}-${current.high}%.`;
}

export const GOAL_REGIONS = [
  "shoulders", "chest", "upper_back", "lats", "waist", "hips", "upper_arm", "forearm", "thigh", "calf",
] as const; // prettier-ignore
export type GoalRegion = (typeof GOAL_REGIONS)[number];

/** Labels follow ADR-037 and ADR-039: lats is an estimate, upper back is not measured. */
export const REGION_LABEL: Record<GoalRegion, string> = {
  shoulders: "Shoulders",
  chest: "Chest",
  upper_back: "Upper back (not measured)",
  lats: "Back width (estimate)",
  waist: "Waist",
  hips: "Hips",
  upper_arm: "Upper arm",
  forearm: "Forearm",
  thigh: "Thigh",
  calf: "Calf",
};

/** The reference chart does not exist yet, so the scale shows text anchors (ADR-052). */
export const RATING_ANCHORS = ["1 soft", "2 average", "3 fit", "4 lean", "5 very lean and defined"] as const;
/** An untuned placeholder, fixed before any result and never tuned after one (ADR-052 Amendment 1). */
export const DISAGREE_BY = 2; // points

/** A note when the user's rating and the model's estimate differ by DISAGREE_BY or more; null otherwise. */
export function ratingNote(self: number | undefined, model: number | undefined): string | null {
  if (self === undefined || model === undefined || Math.abs(self - model) < DISAGREE_BY) return null;
  return `Your rating (${self}) and the model's estimate (${model}) differ. The estimate is experimental and goal photos are often posed, lit or edited.`;
}
