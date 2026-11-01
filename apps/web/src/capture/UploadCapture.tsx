import { useState } from "react";
import { processUpload } from "./pipeline";
import { VIEWS, type ViewName } from "./types";
import type { Views } from "./useTurnCapture";

const REQUIRED: ViewName[] = ["front", "left"];

interface Props {
  views: Views;
  onDone: (views: Views) => void;
  onBack: () => void;
}

export default function UploadCapture({ views: initial, onDone, onBack }: Props) {
  const [views, setViews] = useState<Views>(initial);
  const [busy, setBusy] = useState<ViewName | null>(null);
  const [errors, setErrors] = useState<Partial<Record<ViewName, string>>>({});

  const setError = (name: ViewName, msg: string) => setErrors((e) => ({ ...e, [name]: msg }));

  async function pick(name: ViewName, file?: File) {
    if (!file) return;
    setBusy(name);
    setError(name, "");
    try {
      const view = await processUpload(file, name);
      if (view) setViews((v) => ({ ...v, [name]: view }));
      else setError(name, "No person found in this photo");
    } catch (e) {
      setError(name, e instanceof Error ? e.message : "Could not read this image");
    } finally {
      setBusy(null);
    }
  }

  return (
    <section className="mx-auto max-w-xl p-4" aria-labelledby="up-h">
      <h1 id="up-h" className="text-xl font-bold">Upload photos</h1>
      <p className="mt-2 text-gray-600">
        One full-body photo per view. They are read in your browser and never uploaded; only the outline is kept.
        Use photos you have the right to use. Front and left are required.
      </p>
      <ul className="mt-4 space-y-3">
        {VIEWS.map((name) => (
          <li key={name} className="flex flex-wrap items-center gap-3">
            <label className="w-40 font-semibold capitalize" htmlFor={`up-${name}`}>
              {name}
              {REQUIRED.includes(name) ? " *" : ""}
            </label>
            <input id={`up-${name}`} type="file" accept="image/*" disabled={busy !== null} onChange={(e) => pick(name, e.target.files?.[0])} />
            {busy === name && <span>Reading…</span>}
            {views[name] && busy !== name && <span className="text-green-700">✓ outline ready</span>}
            {errors[name] && <span className="text-red-700">{errors[name]}</span>}
          </li>
        ))}
      </ul>
      <div className="mt-6 flex gap-3">
        <button
          className="rounded bg-black px-4 py-2 text-white disabled:opacity-40"
          disabled={busy !== null || !REQUIRED.every((v) => views[v])}
          onClick={() => onDone(views)}
        >
          Review
        </button>
        <button className="rounded border px-4 py-2" onClick={onBack}>Use the camera instead</button>
      </div>
    </section>
  );
}
