import { useEffect, useState } from "react";
import { getHealth, type Health } from "./lib/api";

const WAKING_AFTER_MS = 3000; // free HF Spaces sleep; first request can take 30-60 s

export default function App() {
  const [health, setHealth] = useState<Health>();
  const [error, setError] = useState(false);
  const [waking, setWaking] = useState(false);

  useEffect(() => {
    const ctrl = new AbortController();
    const timer = setTimeout(() => setWaking(true), WAKING_AFTER_MS);
    getHealth(ctrl.signal)
      .then(setHealth)
      .catch(() => !ctrl.signal.aborted && setError(true))
      .finally(() => clearTimeout(timer));
    return () => {
      ctrl.abort();
      clearTimeout(timer);
    };
  }, []);

  return (
    <main className="mx-auto max-w-xl p-8 font-sans">
      <h1 className="text-3xl font-bold">PhysiqueGap</h1>
      <p className="mt-2 text-gray-600">Your photos never leave your device.</p>
      <p className="mt-6">
        API:{" "}
        {error
          ? "unreachable"
          : health
            ? `${health.status} · ${Object.entries(health.providers)
                .map(([name, p]) => `${name}: ${p.status}`)
                .join(", ")}`
            : waking
              ? "waking server… this can take up to a minute"
              : "checking…"}
      </p>
    </main>
  );
}
