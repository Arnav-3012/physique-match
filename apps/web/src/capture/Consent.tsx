import { useState } from "react";
import { PRIVACY_COPY } from "./privacyCopy";

export default function Consent({ onAccept }: { onAccept: () => void }) {
  const [adult, setAdult] = useState(false);

  return (
    <section className="mx-auto max-w-xl p-8" aria-labelledby="consent-h">
      <h1 id="consent-h" className="text-2xl font-bold">Before we start</h1>
      <p className="mt-4 text-gray-600">{PRIVACY_COPY}</p>
      <p className="mt-4 text-gray-600">
        Before anything is sent you can preview exactly what: the review step shows the outlines. Next you choose what
        to compare, which decides what to wear; you never have to remove clothes.
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
