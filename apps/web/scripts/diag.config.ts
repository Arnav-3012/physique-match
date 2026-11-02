import { defineConfig } from "vitest/config";

// Only for the browser-path diagnostic: `npx vitest run --config scripts/diag.config.ts`, from apps/web.
export default defineConfig({ test: { include: ["scripts/browser_path.diag.ts"], testTimeout: 600_000 } });
