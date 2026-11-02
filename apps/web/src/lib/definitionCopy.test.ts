import { describe, expect, it } from "vitest";
import { ADIPOSITY_LABEL, bandOf, DEFINITION_LABEL, LIMITS_COPY } from "./definitionCopy";

describe("bandOf", () => {
  it("maps the five classes to low, mid, high", () => {
    expect([1, 2, 3, 4, 5].map(bandOf)).toEqual(["low", "low", "mid", "high", "high"]);
  });

  it("rejects anything that is not a class 1-5", () => {
    for (const bad of [0, 6, 2.5, Number.NaN]) expect(() => bandOf(bad)).toThrow(RangeError);
  });
});

describe("copy", () => {
  it("names the definition score as an estimate and adiposity as a waist-girth proxy", () => {
    expect(DEFINITION_LABEL).toBe("leanness & muscularity (estimate)");
    expect(ADIPOSITY_LABEL).toContain("waist-girth proxy");
  });

  it("never uses the retired word, and keeps the limits and the user's-choice sentences", () => {
    for (const text of [DEFINITION_LABEL, ADIPOSITY_LABEL, LIMITS_COPY]) expect(text.toLowerCase()).not.toContain("bulky");
    expect(LIMITS_COPY).toContain("least reliable for very muscular bodies");
    expect(LIMITS_COPY).toContain("cut, recomp or bulk is always your choice");
  });
});
