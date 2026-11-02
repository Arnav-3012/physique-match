import { BAND_LABELS, bandOf, EXPERIMENTAL_BADGE } from "./bands";
import type { Scores } from "./score";

/**
 * The only definition UI (ADR-053 Amendment 6): ONE overall band and ONE adiposity band. No per-region rows, bars or
 * scores exist, and the overall band is never copied into region rows.
 */
export default function DefinitionBands({ scores }: { scores: Scores }) {
  return (
    <section className="mt-4 rounded border p-3" aria-labelledby="def-h">
      <h2 id="def-h" className="font-semibold">
        On-device estimate{" "}
        <span className="ml-1 rounded bg-amber-100 px-2 py-0.5 text-xs font-normal text-amber-900">{EXPERIMENTAL_BADGE}</span>
      </h2>
      <dl className="mt-2 grid grid-cols-[1fr_auto] gap-x-4 gap-y-1">
        <dt>{BAND_LABELS.definition}</dt>
        <dd className="font-semibold capitalize">{bandOf(scores.definition)}</dd>
        <dt>{BAND_LABELS.adiposity}</dt>
        <dd className="font-semibold capitalize">{bandOf(scores.adiposity)}</dd>
      </dl>
      <p className="mt-2 text-xs text-gray-600">
        Adiposity is a waist-girth proxy, not a body-fat %, and is least reliable for very muscular bodies. Trained on
        synthetic renders only. Not medical advice.
      </p>
    </section>
  );
}
