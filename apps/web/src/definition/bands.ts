/** Class 1-5 -> the band the app shows (ADR-053 Amendment 4 item 3): low = 1-2, mid = 3, high = 4-5. */
export type Band = "low" | "mid" | "high";

export const bandOf = (cls: number): Band => (cls <= 2 ? "low" : cls === 3 ? "mid" : "high");

/** The two things the app shows, nothing else (ADR-053 Amendment 6). */
export const BAND_LABELS = {
  definition: "leanness & muscularity (estimate)",
  adiposity: "adiposity (waist-girth proxy)",
} as const;
export const EXPERIMENTAL_BADGE = "experimental - trained on synthetic data";
