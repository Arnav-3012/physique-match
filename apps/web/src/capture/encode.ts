import { encode } from "fast-png";

/** 1-channel 8-bit PNG holding only {0,255} (canvas.toBlob can only emit RGBA), base64. */
export function maskToPngB64(mask: Uint8Array, width: number, height: number): string {
  const png = encode({ width, height, data: mask, depth: 8, channels: 1 });
  let bin = "";
  for (let i = 0; i < png.length; i += 0x8000) bin += String.fromCharCode(...png.subarray(i, i + 0x8000));
  return btoa(bin);
}
