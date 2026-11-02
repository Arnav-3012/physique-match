// ONNX Runtime Web (WASM only) against Python ONNX Runtime on the same fixture tensors, both models (ADR-053 Amendment 6).
// Run from apps/web after `python -m ml.src.definition.export_fp32` wrote the models and the fixture:
//   node scripts/ort_web_parity.mjs
// Tolerance fixed before the first run: identical predicted classes on every fixture image and
// max |logit difference| <= 0.05, per model. A failure means the pre-registered fallback applies.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-web";

const MAX_LOGIT_DIFF = 0.05;
const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const fixtureDir = here("../../../ml/data/export/");
const { shape, ...expected } = JSON.parse(readFileSync(fixtureDir + "fixture.json", "utf8"));
const bytes = readFileSync(fixtureDir + "fixture.bin");
const [n, c, h, w] = shape;
const per = c * h * w;
const classes = (logits) => logits.filter((v) => v > 0).length;

ort.env.wasm.numThreads = 1; // what the app can rely on without cross-origin isolation
let ok = true;
for (const name of ["definition", "adiposity"]) {
  const session = await ort.InferenceSession.create(here(`../public/models/${name}.fp32.onnx`), {
    executionProviders: ["wasm"],
  });
  let worst = 0;
  let mismatched = 0;
  for (let i = 0; i < n; i++) {
    const x = Float32Array.from(bytes.subarray(i * per, (i + 1) * per), (v) => v / 255);
    const got = Array.from((await session.run({ x: new ort.Tensor("float32", x, [1, c, h, w]) })).logits.data);
    const want = expected[`${name}_logits`][i];
    worst = Math.max(worst, ...want.map((v, k) => Math.abs(v - got[k])));
    if (classes(got) !== classes(want)) mismatched++;
  }
  const pass = worst <= MAX_LOGIT_DIFF && mismatched === 0;
  ok &&= pass;
  console.log(`${name}: ${n} images, max |wasm - python ort| ${worst.toExponential(2)} (<= ${MAX_LOGIT_DIFF}), ` +
    `class mismatches ${mismatched}: ${pass ? "PASS" : "FAIL"}`);
}
process.exit(ok ? 0 : 1);
