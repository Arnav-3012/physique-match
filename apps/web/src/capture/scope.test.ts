import { describe, expect, it } from "vitest";
import {
  availableScopes,
  chestApproximate,
  clothingGuidance,
  DEFINITION_REGIONS,
  defaultTorso,
  displayRegion,
  parseAge,
  SCOPE_REGIONS,
  SCOPE_TEXT,
  sexUnsetNote,
} from "./scope";

describe("availableScopes (age gate, ADR-051)", () => {
  it("offers the full-body option only to adults", () => {
    for (const age of [undefined, 1, 17]) expect(availableScopes(age)).not.toContain("full_body");
    for (const age of [18, 30, 99]) expect(availableScopes(age)).toContain("full_body");
  });

  it("always offers the clothed scan and the arms/shoulders/legs option", () => {
    for (const age of [undefined, 12, 18]) {
      expect(availableScopes(age)).toContain("none");
      expect(availableScopes(age)).toContain("arms_shoulders_legs");
    }
  });
});

describe("parseAge", () => {
  it("accepts whole years and treats anything else as unset", () => {
    expect(parseAge("30")).toBe(30);
    expect(parseAge(" 18 ")).toBe(18);
    for (const bad of ["", " ", "abc", "17.5", "0", "-3", "121"]) expect(parseAge(bad)).toBeUndefined();
  });
});

describe("scope regions", () => {
  it("only the full-body scan shows the abdomen, so only it scores the overall definition (via abs)", () => {
    expect(SCOPE_REGIONS.none).toEqual([]);
    expect(SCOPE_REGIONS.arms_shoulders_legs).toEqual([]);
    expect(SCOPE_REGIONS.full_body).toEqual(["abs"]);
    expect(DEFINITION_REGIONS).toContain("abs");
  });

  it("the scope A text says why definition is not scored", () => {
    expect(SCOPE_TEXT.arms_shoulders_legs.blurb).toContain("does not show the abdomen");
  });
});

describe("displayRegion: a number only for an eligible region", () => {
  const NOT_ELIGIBLE = [
    "not_measured_covered",
    "not_measured_age_gate",
    "not_measured_goal_covers",
    "not_measured_not_in_demo",
  ] as const;

  it("never shows a score for a non-eligible status, even if one is passed in", () => {
    for (const status of NOT_ELIGIBLE) {
      const shown = displayRegion(status, "why", 4);
      expect(shown).toEqual({ kind: "na", reason: "why" });
      expect(JSON.stringify(shown)).not.toContain("4");
    }
  });

  it("shows the reason, with a fallback text when none came", () => {
    expect(displayRegion("not_measured_covered", undefined, undefined)).toEqual({ kind: "na", reason: "Not scored." });
  });

  it("shows a score only for an eligible region that has one", () => {
    expect(displayRegion("measured", undefined, 3)).toEqual({ kind: "score", value: 3 });
    expect(displayRegion("measured", undefined, undefined)).toBeNull();
  });
});

describe("sex unset note", () => {
  it("lists body-fat %, calories, back width and the V-taper ceiling as hidden, only for 'Prefer not to say'", () => {
    const note = sexUnsetNote("not_specified") ?? "";
    for (const word of ["body-fat %", "calorie", "back width", "V-taper"]) expect(note).toContain(word);
    expect(sexUnsetNote("male")).toBeNull();
    expect(sexUnsetNote("female")).toBeNull();
    expect(sexUnsetNote(undefined)).toBeNull();
  });
});

describe("torso clothing", () => {
  it("defaults by sex and never preselects when sex is unset", () => {
    expect(defaultTorso("male")).toBe("bare");
    expect(defaultTorso("female")).toBe("sports_bra");
    expect(defaultTorso("not_specified")).toBeUndefined();
  });

  it("labels the chest approximate only for a bra in the full-body scope", () => {
    expect(chestApproximate("full_body", "sports_bra")).toBe(true);
    expect(chestApproximate("full_body", "bare")).toBe(false);
    expect(chestApproximate("arms_shoulders_legs", "sports_bra")).toBe(false);
  });

  it("never asks for a shirtless scan outside the full-body scope", () => {
    for (const scope of ["none", "arms_shoulders_legs"] as const) {
      expect(clothingGuidance(scope, "bare").toLowerCase()).not.toContain("bare torso");
      expect(clothingGuidance(scope, undefined).toLowerCase()).not.toContain("shirtless and");
    }
    expect(clothingGuidance("full_body", "bare")).toContain("Bare torso");
    expect(clothingGuidance("full_body", undefined)).toContain("Choose");
  });
});
