import { describe, expect, it } from "vitest";
import { boundingBox, checkFullBody, checkLighting, checkLooseClothing, coverage, touchedEdges, viewIssues } from "./quality";
import { pose } from "./test-utils";
import { LM } from "./types";

const W = 100, H = 200;
/** Rectangle person at x 30..70, y 20..180 as a probability mask. */
const person = (x0 = 30, x1 = 70, y0 = 20, y1 = 180) => {
  const m = new Float32Array(W * H);
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) m[y * W + x] = 1;
  return m;
};

describe("quality", () => {
  it("finds the bounding box and coverage", () => {
    expect(boundingBox(person(), W, H, 0.5)).toEqual({ x0: 30, y0: 20, x1: 70, y1: 180 });
    expect(boundingBox(new Float32Array(W * H), W, H, 0.5)).toBeNull();
    expect(coverage(person(), 0.5)).toBeCloseTo((40 * 160) / (W * H));
  });

  it("accepts profile/back poses where the far side or face is hidden", () => {
    expect(checkFullBody(pose())).toEqual([]);
    const hidden = { visibility: 0.05 };
    const far = pose({ [LM.rShoulder]: hidden, [LM.rHip]: hidden, [LM.rKnee]: hidden, [LM.rAnkle]: hidden, [LM.nose]: hidden });
    expect(checkFullBody(far)).toEqual([]);
  });

  it("names what is missing when a whole pair is unseen or off-frame", () => {
    const both = { visibility: 0.1 };
    expect(checkFullBody(pose({ [LM.lAnkle]: both, [LM.rAnkle]: both }))[0]).toContain("ankles");
    expect(checkFullBody(pose({ [LM.lAnkle]: { y: 1.0 }, [LM.rAnkle]: { y: 1.0 } }))[0]).toContain("ankles");
  });

  it("flags dark and bright frames", () => {
    expect(checkLighting(120)).toEqual([]);
    expect(checkLighting(20)).toHaveLength(1);
    expect(checkLighting(240)).toHaveLength(1);
  });

  it("accepts a good view and rejects empty / edge-touching ones", () => {
    expect(viewIssues(person(), W, H, pose())).toEqual([]);
    expect(viewIssues(new Float32Array(W * H), W, H, pose()).length).toBeGreaterThan(0);
    expect(viewIssues(person(30, 70, 0, 180), W, H, pose()).join()).toContain("top");
    expect(viewIssues(person(30, 70, 20, 200), W, H, pose()).join()).toContain("bottom");
  });

  it("ignores a stray mask pixel on the border", () => {
    const m = person();
    m[0] = 1;
    expect(touchedEdges(m, W, H)).toEqual([]);
  });

  it("warns on a torso much wider than the shoulders, not on a snug one", () => {
    // shoulder keypoints span 0.4..0.6 of W = 20 px
    expect(checkLooseClothing(person(40, 60), W, H, pose())).toEqual([]);
    expect(checkLooseClothing(person(20, 80), W, H, pose())).toHaveLength(1);
  });
});
