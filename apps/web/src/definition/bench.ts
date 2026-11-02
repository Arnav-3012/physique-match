// Dev-only latency check (ADR-053 Amendment 6): the real preprocessing and both real graphs in a real browser tab.
// Open http://localhost:5173/bench.html after `npm run dev`. The frame is synthetic (a figure on a plain background)
// at the size the capture pipeline produces (long side 720): preprocessing cost depends on the frame size, not on
// what is in it. Pre-registered: combined p50 and p95 <= 300 ms.
import { loadRunners } from "./models";
import { frontTensor, type Frame } from "./preprocess";
import { scoreTensor } from "./score";

const W = 540, H = 720, WARMUP = 10, RUNS = 100, LIMIT_MS = 300;

function syntheticFrame(): { img: Frame; prob: Float32Array } {
  const data = new Uint8ClampedArray(W * H * 4);
  const prob = new Float32Array(W * H);
  const inside = (x: number, y: number) =>
    ((x - 270) / 36) ** 2 + ((y - 110) / 46) ** 2 < 1 || // head
    (x > 215 && x < 325 && y > 160 && y < 400) || // torso
    (x > 150 && x < 215 && y > 170 && y < 330) || (x > 325 && x < 390 && y > 170 && y < 330) || // arms
    (x > 225 && x < 265 && y > 400 && y < 690) || (x > 275 && x < 315 && y > 400 && y < 690); // legs
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const k = y * W + x, on = inside(x, y);
      prob[k] = on ? 1 : 0;
      data.set(on ? [205, 160, 135, 255] : [60 + (x % 7), 62, 70 + (y % 5), 255], k * 4);
    }
  }
  return { img: { data, width: W, height: H }, prob };
}

const pct = (xs: number[], q: number) => [...xs].sort((a, b) => a - b)[Math.min(xs.length - 1, Math.floor(q * xs.length))];
const ms = (t0: number) => performance.now() - t0;

async function run(out: HTMLElement) {
  const log = (s: string) => void (out.textContent += s + "\n");
  out.textContent = "";
  const t0 = performance.now();
  const runners = await loadRunners();
  log(`load, both sessions in parallel: ${ms(t0).toFixed(0)} ms`);
  const { img, prob } = syntheticFrame();

  const combined: number[] = [], pre: number[] = [], def: number[] = [], adip: number[] = [];
  for (let i = 0; i < WARMUP + RUNS; i++) {
    const a = performance.now();
    const x = frontTensor(img, prob)!;
    const b = performance.now();
    await runners.definition(x);
    const c = performance.now();
    await runners.adiposity(x);
    const d = performance.now();
    if (i === 0) log(`first run (cold): ${(d - a).toFixed(0)} ms`);
    if (i >= WARMUP) {
      pre.push(b - a); def.push(c - b); adip.push(d - c); combined.push(d - a);
    }
  }
  const scores = await scoreTensor(runners, frontTensor(img, prob)!);
  const [p50, p95] = [pct(combined, 0.5), pct(combined, 0.95)];
  log(`preprocess p50 ${pct(pre, 0.5).toFixed(1)} ms, definition p50 ${pct(def, 0.5).toFixed(1)} ms, adiposity p50 ${pct(adip, 0.5).toFixed(1)} ms`);
  log(`COMBINED over ${RUNS} runs: p50 ${p50.toFixed(1)} ms, p95 ${p95.toFixed(1)} ms (limit ${LIMIT_MS}): ${p50 <= LIMIT_MS && p95 <= LIMIT_MS ? "PASS" : "FAIL"}`);
  log(`scores on the synthetic figure (a plumbing check, not a result): ${JSON.stringify(scores)}`);
  log(`env: crossOriginIsolated ${self.crossOriginIsolated}, hardwareConcurrency ${navigator.hardwareConcurrency}, ${navigator.userAgent}`);
}

const out = document.getElementById("out")!;
document.getElementById("run")!.addEventListener("click", () => run(out).catch((e) => void (out.textContent += `ERROR: ${e}\n`)));
