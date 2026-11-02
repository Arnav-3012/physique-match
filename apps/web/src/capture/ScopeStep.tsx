import { useState } from "react";
import { PRIVACY_COPY } from "./privacyCopy";
import {
  ADULT_ONLY_LINE,
  availableScopes,
  chestApproximate,
  clothingGuidance,
  defaultTorso,
  parseAge,
  SCOPE_REGIONS,
  SCOPE_TEXT,
  sexUnsetNote,
  type Scope,
  type ScopeChoice,
  type Sex,
  type Torso,
} from "./scope";

const SEXES: { value: Sex; label: string }[] = [
  { value: "male", label: "Male" },
  { value: "female", label: "Female" },
  { value: "not_specified", label: "Prefer not to say" },
];
const TORSOS: { value: Torso; label: string }[] = [
  { value: "bare", label: "Bare torso" },
  { value: "sports_bra", label: "Sports bra / crop top" },
];

interface Props {
  onStart: (choice: ScopeChoice) => void;
  onBack: () => void;
}

/** "What do you want to compare?", asked once, with the requirement card before the camera starts (ADR-051). */
export default function ScopeStep({ onStart, onBack }: Props) {
  const [ageText, setAgeText] = useState("");
  const [sex, setSex] = useState<Sex>();
  const [picked, setPicked] = useState<Scope>();
  const [torsoPick, setTorsoPick] = useState<Torso>();
  const [onDevice, setOnDevice] = useState(false); // opt-in, off by default (ADR-013)

  const age = parseAge(ageText);
  const offered = availableScopes(age);
  const scope = picked && offered.includes(picked) ? picked : undefined; // age edited after choosing full body
  const torso = torsoPick ?? (sex ? defaultTorso(sex) : undefined);
  const ready = scope !== undefined && sex !== undefined && (scope !== "full_body" || torso !== undefined);

  const start = (s: Scope) =>
    onStart({
      scope: s,
      age,
      sex: sex ?? "not_specified",
      torso: s === "full_body" ? torso : undefined,
      scoreOnDevice: s === "full_body" && onDevice,
    });

  return (
    <section className="mx-auto max-w-xl p-4" aria-labelledby="scope-h">
      <h1 id="scope-h" className="text-xl font-bold">What do you want to compare?</h1>
      <p className="mt-1 text-gray-600">
        Muscle definition can only be scored where the scan shows your body, so this decides what to wear. Size gaps
        work in any fitted clothing.
      </p>

      <fieldset className="mt-4">
        <legend className="font-semibold">About you</legend>
        <label className="mt-1 flex items-center gap-2">
          Age
          <input
            className="w-20 rounded border px-2 py-1"
            inputMode="numeric"
            value={ageText}
            onChange={(e) => setAgeText(e.target.value)}
            aria-describedby="age-note"
          />
        </label>
        <p id="age-note" className="text-sm text-gray-500">Used only to decide which options are offered.</p>
        <div className="mt-2 flex flex-wrap gap-4" role="radiogroup" aria-label="Sex">
          {SEXES.map((s) => (
            <label key={s.value} className="flex items-center gap-1">
              <input type="radio" name="sex" checked={sex === s.value} onChange={() => setSex(s.value)} />
              {s.label}
            </label>
          ))}
        </div>
        <p className="text-sm text-gray-500">
          Needed for body-fat %, calories, back width and the V-taper ceiling. Never guessed or filled in for you.
        </p>
        {sexUnsetNote(sex) && <p className="text-sm text-amber-700">{sexUnsetNote(sex)}</p>}
      </fieldset>

      <fieldset className="mt-4">
        <legend className="font-semibold">Scan option</legend>
        {offered.map((s) => (
          <label key={s} className="mt-2 flex items-start gap-2">
            <input type="radio" name="scope" className="mt-1" checked={scope === s} onChange={() => setPicked(s)} />
            <span>
              <span className="font-semibold">{SCOPE_TEXT[s].title}</span>
              <span className="block text-gray-600">{SCOPE_TEXT[s].blurb}</span>
            </span>
          </label>
        ))}
        {!offered.includes("full_body") && <p className="mt-2 text-sm text-gray-500">{ADULT_ONLY_LINE}</p>}
      </fieldset>

      {scope && (
        <div className="mt-4 rounded border p-3" aria-live="polite">
          <h2 className="font-semibold">Before the camera starts</h2>
          <p className="mt-1">{clothingGuidance(scope, torso)}</p>
          {scope === "full_body" && (
            <div className="mt-2 flex flex-wrap gap-4" role="radiogroup" aria-label="Torso clothing">
              {TORSOS.map((t) => (
                <label key={t.value} className="flex items-center gap-1">
                  <input type="radio" name="torso" checked={torso === t.value} onChange={() => setTorsoPick(t.value)} />
                  {t.label}
                </label>
              ))}
            </div>
          )}
          {/* Dev only until the scores are shown to work on real photos (dev-log, first real-photo result). */}
          {import.meta.env.DEV && scope === "full_body" && (
            <label className="mt-3 flex items-start gap-2">
              <input type="checkbox" className="mt-1" checked={onDevice} onChange={(e) => setOnDevice(e.target.checked)} />
              <span>
                Score my leanness &amp; muscularity on this device (experimental, off by default)
                <span className="block text-sm text-gray-600">
                  A small model runs in this browser on your front photo, then the photo is released. Only two bands
                  can leave the device. Without this, size gaps still work and nothing is scored.
                </span>
              </span>
            </label>
          )}
          {chestApproximate(scope, torso) && (
            <p className="mt-1 text-sm text-amber-700">With a sports bra the chest estimate is labelled approximate.</p>
          )}
          <p className="mt-2 text-sm text-gray-600">
            {SCOPE_REGIONS[scope].length
              ? "One 4-view turn scan in this clothing feeds the size gaps. A leanness & muscularity comparison with your goal (experimental) is planned for a later step."
              : scope === "none"
                ? "Definition will be marked not scored, with the reason: clothed scan."
                : "Definition will be marked not scored, with the reason: this scan does not show the abdomen."}
          </p>
          <p className="mt-2 text-sm text-gray-600">{PRIVACY_COPY}</p>
          {sex === undefined && <p className="mt-1 text-sm text-amber-700">Choose an option under Sex to continue.</p>}
        </div>
      )}

      <div className="mt-4 flex flex-wrap gap-3">
        <button className="rounded bg-black px-4 py-2 text-white disabled:opacity-40" disabled={!ready} onClick={() => scope && start(scope)}>
          Start scan
        </button>
        {scope && scope !== "none" && (
          <button className="rounded border px-4 py-2 disabled:opacity-40" disabled={sex === undefined} onClick={() => start("none")}>
            Use clothed scan instead
          </button>
        )}
        <button className="rounded border px-4 py-2" onClick={onBack}>Back</button>
      </div>
    </section>
  );
}
