/**
 * The privacy statement shown before the camera starts (consent and the scope step). Every sentence was checked
 * against the code (ADR-051). Mirrored word for word in docs/privacy-and-safety.md; tests/test_privacy_copy.py
 * checks they match. Keep it one double-quoted line with no escapes. Never "nothing leaves your device".
 */
export const PRIVACY_COPY =
  "Photos are processed on this device. No photo is uploaded or saved. By design, this app writes no photo, thumbnail or video frame to your browser's storage or to a file, and frames are released from memory once the outlines are made. The server receives only a black-and-white outline (silhouette), body joint positions and the numbers you enter. It uses the outline to calculate your result and does not store it. The hosting service keeps its own technical request logs (time, address, status), not your outline. The muscle-definition model runs in your browser and is off unless you turn it on.";
