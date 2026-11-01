import type { ViewName } from "./types";

/** The chosen RGB frame per view, for the on-device definition model (P4). Memory only; never persisted or sent. */
const frames = new Map<ViewName, ImageData>();

export const rgbStore = {
  set: (view: ViewName, img: ImageData) => void frames.set(view, img),
  get: (view: ViewName) => frames.get(view),
  clear: () => frames.clear(),
};

if (typeof window !== "undefined") window.addEventListener("pagehide", rgbStore.clear);
