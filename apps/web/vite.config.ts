import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  optimizeDeps: { exclude: ["onnxruntime-web"] }, // it loads its own .wasm / .mjs at runtime (public/ort)
  test: { include: ["src/**/*.test.ts"] },
});
