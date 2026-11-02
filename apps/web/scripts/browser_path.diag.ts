// Is the wrong real-photo result a bug in the browser path? The val renders go through the app's own frontTensor (the
// alpha channel stands in for the person mask) and the served ONNX files, WASM, and are scored against their labels.
// Fixed before the first run, against the Python val numbers of the same graphs: rho >= reference - 0.03 and
// MAE <= reference + 0.10 on both outputs. A pass rules out crop, mask, resize, scaling and output-mapping bugs on renders;
// it does not test the segmenter or a real photo. Val only, never test.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { decode } from "fast-png";
import * as ort from "onnxruntime-web";
import { expect, it } from "vitest";
import { frontTensor, INPUT_H, INPUT_W } from "../src/definition/preprocess";
import { classOf } from "../src/definition/score";

const ROOT = resolve("../..");
const REFERENCE = { definition: { rho: 0.928, mae: 0.265 }, adiposity: { rho: 0.895, mae: 0.346 } };
const RHO_MARGIN = 0.03, MAE_MARGIN = 0.1;
const OUTPUTS = ["definition", "adiposity"] as const;

const rows = (file: string) => readFileSync(resolve(ROOT, file), "utf8").trim().split("\n").map((l) => l.split(","));

function ranks(v: number[]): number[] {
  const order = v.map((_, i) => i).sort((a, b) => v[a] - v[b]);
  const out = new Array<number>(v.length);
  for (let i = 0; i < order.length; ) {
    let j = i;
    while (j + 1 < order.length && v[order[j + 1]] === v[order[i]]) j++;
    for (let k = i; k <= j; k++) out[order[k]] = (i + j) / 2;
    i = j + 1;
  }
  return out;
}

function spearman(a: number[], b: number[]): number {
  const [ra, rb] = [ranks(a), ranks(b)];
  const mean = (v: number[]) => v.reduce((s, x) => s + x, 0) / v.length;
  const [ma, mb] = [mean(ra), mean(rb)];
  const cov = ra.reduce((s, x, i) => s + (x - ma) * (rb[i] - mb), 0);
  return cov / Math.sqrt(ra.reduce((s, x) => s + (x - ma) ** 2, 0) * rb.reduce((s, x) => s + (x - mb) ** 2, 0));
}

it("val renders through the browser path score like the Python path", async () => {
  const [labelHead, ...labelRows] = rows("synth/labels.csv");
  const col = (name: string) => labelHead.indexOf(name);
  const rig = new Map(rows("synth/out/render/params.csv").slice(1).map((r) => [r[1], r[9]]));
  const val = labelRows.filter((r) => r[col("split")] === "val");

  ort.env.wasm.numThreads = 1;
  const sessions = Object.fromEntries(
    await Promise.all(OUTPUTS.map(async (o) => [o, await ort.InferenceSession.create(resolve(`public/models/${o}.fp32.onnx`), { executionProviders: ["wasm"] })] as const)),
  );
  const pred: Record<string, number[]> = { definition: [], adiposity: [] };
  const truth: Record<string, number[]> = { definition: [], adiposity: [] };
  for (const r of val) {
    const id = r[col("id")];
    const png = decode(readFileSync(resolve(ROOT, `synth/out/render/images/${id}_bump_${rig.get(id)}_front.png`)));
    if (png.channels !== 4 || png.depth !== 8) throw new Error(`${id}: expected 8-bit RGBA, got ${png.channels} channels, ${png.depth} bits`);
    const data = new Uint8ClampedArray(png.data.buffer, png.data.byteOffset, png.data.length);
    const prob = Float32Array.from({ length: png.width * png.height }, (_, i) => data[i * 4 + 3] / 255);
    const x = frontTensor({ data, width: png.width, height: png.height }, prob);
    if (!x) throw new Error(`${id}: no person found in the alpha channel`);
    for (const o of OUTPUTS) {
      const out = await sessions[o].run({ x: new ort.Tensor("float32", x, [1, 3, INPUT_H, INPUT_W]) });
      pred[o].push(classOf(out.logits.data as Float32Array));
      truth[o].push(Number(r[col(o)]));
    }
  }

  let pass = true;
  for (const o of OUTPUTS) {
    const rho = spearman(pred[o], truth[o]);
    const mae = pred[o].reduce((s, p, i) => s + Math.abs(p - truth[o][i]), 0) / pred[o].length;
    const ok = rho >= REFERENCE[o].rho - RHO_MARGIN && mae <= REFERENCE[o].mae + MAE_MARGIN;
    pass &&= ok;
    const counts = [1, 2, 3, 4, 5].map((c) => pred[o].filter((p) => p === c).length);
    console.log(`${o}: n ${pred[o].length}  browser path rho ${rho.toFixed(3)} MAE ${mae.toFixed(3)}  Python rho ${REFERENCE[o].rho} MAE ${REFERENCE[o].mae}  predicted class counts ${counts.join("/")}  ${ok ? "PASS" : "FAIL"}`);
  }
  console.log(pass ? "VERDICT: no browser-path bug on renders" : "VERDICT: browser-path bug likely; stop and report");
  expect(pass).toBe(true);
});
