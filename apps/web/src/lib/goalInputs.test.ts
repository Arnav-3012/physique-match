import { describe, expect, it } from "vitest";
import { BANDS, BANDS_LABEL, BF_FLOOR, bandsFor, bandText, bandVsCurrent, DISAGREE_BY, ratingNote } from "./goalInputs";

describe("goal body-fat bands (ADR-052)", () => {
  it("are unavailable when sex is unset", () => {
    expect(bandsFor("not_specified")).toBeNull();
    expect(bandsFor("male")).toHaveLength(3);
    expect(bandsFor("female")).toHaveLength(3);
  });

  it("never offer a band below the body-fat floor and are ordered", () => {
    for (const sex of ["male", "female"] as const) {
      const bands = BANDS[sex];
      expect(Math.min(...bands.map((b) => b.low))).toBe(BF_FLOOR[sex]);
      for (const b of bands) expect(b.low).toBeLessThanOrEqual(b.high);
      for (let i = 1; i < bands.length; i++) expect(bands[i].low).toBeGreaterThan(bands[i - 1].high);
    }
  });

  it("marks the low end of the lowest band as not sustainable, and no other", () => {
    for (const sex of ["male", "female"] as const) {
      const flags = BANDS[sex].map((b) => b.lowEndNotSustainable);
      expect(flags).toEqual([true, false, false]);
    }
    expect(bandText(BANDS.male[0])).toContain("not sustainable");
  });

  it("are labelled approximate fitness-industry categories", () => {
    expect(BANDS_LABEL).toBe("approximate, fitness-industry categories");
  });

  it("compares with the current range only when it exists", () => {
    expect(bandVsCurrent(BANDS.male[0], undefined)).toContain("after the analysis");
    expect(bandVsCurrent(BANDS.male[0], { low: 20, high: 24 })).toContain("lower");
    expect(bandVsCurrent(BANDS.male[2], { low: 10, high: 14 })).toContain("higher");
  });
});

describe("self-rating note", () => {
  it("appears only when both exist and differ by the threshold or more", () => {
    expect(ratingNote(undefined, 3)).toBeNull();
    expect(ratingNote(3, undefined)).toBeNull();
    expect(ratingNote(3, 3 + DISAGREE_BY - 1)).toBeNull();
    expect(ratingNote(1, 1 + DISAGREE_BY)).not.toBeNull();
    expect(ratingNote(5, 5 - DISAGREE_BY)).not.toBeNull();
  });
});
