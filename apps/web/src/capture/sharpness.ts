/** RGBA -> luma (0-255). */
export function toGray(rgba: Uint8ClampedArray): Float32Array {
  const g = new Float32Array(rgba.length / 4);
  for (let i = 0; i < g.length; i++) {
    g[i] = 0.299 * rgba[4 * i] + 0.587 * rgba[4 * i + 1] + 0.114 * rgba[4 * i + 2];
  }
  return g;
}

/** Variance of the 4-neighbour Laplacian, over `mask` pixels when given. Higher = sharper. */
export function laplacianVariance(gray: Float32Array, w: number, h: number, mask?: ArrayLike<number>): number {
  let n = 0;
  let sum = 0;
  let sumSq = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      if (mask && !mask[i]) continue;
      const lap = gray[i - 1] + gray[i + 1] + gray[i - w] + gray[i + w] - 4 * gray[i];
      sum += lap;
      sumSq += lap * lap;
      n++;
    }
  }
  if (n === 0) return 0;
  const mean = sum / n;
  return sumSq / n - mean * mean;
}
