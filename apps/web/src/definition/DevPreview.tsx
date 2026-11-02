import { useEffect, useRef } from "react";
import { INPUT_H, INPUT_W } from "./preprocess";
import type { Runners } from "./score";

/** Dev only: what the models received for the last front frame, and their raw outputs. Memory only, never saved. */
export interface Diag {
  tensor: Float32Array;
  definition: number[];
  adiposity: number[];
}

export const withDiag = (load: () => Promise<Runners>, report: (d: Diag) => void) => async (): Promise<Runners> => {
  const runners = await load();
  let definition: number[] = [];
  return {
    definition: async (x) => {
      const out = await runners.definition(x);
      definition = Array.from(out);
      return out;
    },
    adiposity: async (x) => {
      const out = await runners.adiposity(x);
      report({ tensor: x, definition, adiposity: Array.from(out) });
      return out;
    },
  };
};

const shown = (logits: number[]) => logits.map((v) => v.toFixed(2)).join(", ");

export default function DevPreview({ diag }: { diag: Diag }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const img = new ImageData(INPUT_W, INPUT_H);
    const plane = INPUT_W * INPUT_H;
    for (let k = 0; k < plane; k++) {
      for (let c = 0; c < 3; c++) img.data[k * 4 + c] = diag.tensor[c * plane + k] * 255;
      img.data[k * 4 + 3] = 255;
    }
    canvas.current?.getContext("2d")?.putImageData(img, 0, 0);
  }, [diag]);

  return (
    <section className="mx-auto max-w-2xl p-4 text-sm" aria-label="Developer preview of the model input">
      <h2 className="font-semibold">dev: what the models received</h2>
      <canvas ref={canvas} width={INPUT_W} height={INPUT_H} className="mt-1 border" />
      <p className="mt-1">definition logits: {shown(diag.definition)}</p>
      <p>adiposity logits: {shown(diag.adiposity)}</p>
    </section>
  );
}
