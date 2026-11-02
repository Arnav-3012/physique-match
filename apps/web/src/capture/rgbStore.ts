import type { ViewName } from "./types";

/**
 * The chosen RGB frame per view, for the on-device definition model (P4). Memory only; never persisted or sent.
 * Frames are kept only while the opt-in model is on (`enable`, off by default): with it off `set` stores nothing,
 * so nothing outlives the outline step (ADR-051). `clear` drops every frame and blanks the shared scratch canvas,
 * which otherwise keeps the last drawn frame. P4 model step: score first, then `clear`.
 */
const frames = new Map<ViewName, ImageData>();
const masks = new Map<ViewName, Float32Array>(); // person probability of the same frame (same size), also memory only
let enabled = false;
let scratch: { width: number; height: number } | undefined;

/** Resizing a canvas clears its pixels, so a zero size releases the last frame. */
const blank = () => {
  if (scratch) scratch.width = scratch.height = 0;
};

export const rgbStore = {
  enable: (on: boolean) => {
    enabled = on;
    if (!on) rgbStore.clear();
  },
  set: (view: ViewName, img: ImageData, prob?: Float32Array) => {
    if (!enabled) return;
    frames.set(view, img);
    if (prob) masks.set(view, prob);
  },
  get: (view: ViewName) => frames.get(view),
  getMask: (view: ViewName) => masks.get(view),
  /** The canvas `pipeline.drawScaled` draws every frame into. */
  attachScratch: (canvas: { width: number; height: number }) => void (scratch = canvas),
  releaseScratch: blank,
  clear: () => {
    frames.clear();
    masks.clear();
    blank();
  },
};

if (typeof window !== "undefined") window.addEventListener("pagehide", rgbStore.clear);
