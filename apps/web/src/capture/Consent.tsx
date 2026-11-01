import { useState } from "react";

const ROWS = [
  ["Captured on your device", "Camera frames, analysed in your browser. You see a silhouette, not yourself."],
  ["Sent to the server", "Four black-and-white outlines (head removed) and 33 joint positions per view. You can preview exactly what, before anything is sent."],
  ["Never sent or stored", "Your photos, your face, your camera feed. Nothing is saved on a server; frames are dropped from memory when you leave."],
] as const;

export default function Consent({ onAccept }: { onAccept: () => void }) {
  const [adult, setAdult] = useState(false);

  return (
    <section className="mx-auto max-w-xl p-8" aria-labelledby="consent-h">
      <h1 id="consent-h" className="text-2xl font-bold">Before we start</h1>
      <dl className="mt-4 space-y-3">
        {ROWS.map(([title, text]) => (
          <div key={title}>
            <dt className="font-semibold">{title}</dt>
            <dd className="text-gray-600">{text}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-4 text-gray-600">
        Wear fitted clothing: gym wear, a fitted tee, shorts or leggings. You do not need to remove clothes.
      </p>
      <label className="mt-6 flex items-center gap-2">
        <input type="checkbox" checked={adult} onChange={(e) => setAdult(e.target.checked)} />
        <span>I am 18 or older and I agree to the above.</span>
      </label>
      <button
        className="mt-4 rounded bg-black px-4 py-2 text-white disabled:opacity-40"
        disabled={!adult}
        onClick={onAccept}
      >
        Continue
      </button>
    </section>
  );
}
