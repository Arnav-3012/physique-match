import { describe, expect, it } from "vitest";
import { pose, sidePose } from "./test-utils";
import { LM } from "./types";
import { absYawDeg, facingCamera, matchesView, shoulderRatio, visibleSide, zYawDeg } from "./yaw";

const ref = shoulderRatio(pose(), 1);

describe("yaw", () => {
  it("is 0 deg square on and ~90 deg side-on", () => {
    expect(absYawDeg(pose(), 1, ref)).toBeCloseTo(0, 5);
    expect(absYawDeg(sidePose("left"), 1, ref)).toBeGreaterThan(80);
  });

  it("clamps a wider-than-reference frame to 0 deg instead of NaN", () => {
    expect(absYawDeg(pose(), 1, ref * 0.5)).toBe(0);
  });

  it("tells front from back by nose depth", () => {
    expect(facingCamera(pose())).toBe(true);
    expect(facingCamera(pose({ [LM.nose]: { z: 0.2 } }))).toBe(false);
    expect(facingCamera(pose({ [LM.nose]: { visibility: 0.1 } }))).toBe(false);
  });

  it("tells left from right by the occluded shoulder", () => {
    expect(visibleSide(sidePose("left"))).toBe("left");
    expect(visibleSide(sidePose("right"))).toBe("right");
    expect(visibleSide(pose())).toBeNull();
  });

  it("reads depth gap as side-on: ~0 square, ~84 in profile", () => {
    expect(zYawDeg(pose())).toBe(0);
    expect(zYawDeg(sidePose("left"))).toBeGreaterThan(80);
  });

  it("matches front/back by angle only (turn order separates them) and sides by angle", () => {
    expect(matchesView("front", pose(), 1, ref)).toBe(true);
    expect(matchesView("back", pose(), 1, ref)).toBe(true);
    expect(matchesView("left", sidePose("left"), 1, ref)).toBe(true);
    expect(matchesView("right", sidePose("right"), 1, ref)).toBe(true);
    expect(matchesView("left", pose(), 1, ref)).toBe(false);
    expect(matchesView("front", sidePose("left"), 1, ref)).toBe(false);
  });

  it("a wrong, too-large reference cannot pass a frontal frame as a side view", () => {
    expect(absYawDeg(pose(), 1, ref * 4)).toBeGreaterThan(72); // ratio alone says "turned"
    expect(matchesView("left", pose(), 1, ref * 4)).toBe(false); // depth says "frontal"
    expect(matchesView("front", pose(), 1, ref * 4)).toBe(false);
  });

  it("does not capture a side view at 60 deg of turn", () => {
    const c = Math.cos(Math.PI / 3) * 0.1;
    const turned = pose({ [LM.lShoulder]: { x: 0.5 + c }, [LM.rShoulder]: { x: 0.5 - c } });
    expect(absYawDeg(turned, 1, ref)).toBeCloseTo(60, 0);
    expect(matchesView("left", turned, 1, ref)).toBe(false);
  });

  it("does not capture a side view mid-turn at 45 deg, even if depth is noisy", () => {
    const c = Math.cos(Math.PI / 4) * 0.1; // half the 0.2 shoulder span, foreshortened
    const mid = pose({ [LM.lShoulder]: { x: 0.5 + c, z: 0.3 }, [LM.rShoulder]: { x: 0.5 - c, z: -0.3 } });
    expect(zYawDeg(mid)).toBeGreaterThan(65); // noisy depth alone would say "side-on"
    expect(matchesView("left", mid, 1, ref)).toBe(false);
  });
});
