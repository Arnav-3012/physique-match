import { useState } from "react";
import Choice from "../ui/Choice";
import { Heading, Item, Notice, Panel, Screen } from "../ui/Layout";
import SideFigure from "../ui/SideFigure";
import TextField from "../ui/TextField";
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
    <Screen labelledBy="scope-h" aside={<SideFigure />}>
      <Heading eyebrow="Step 2 · Your scan" title="What do you want to compare?" id="scope-h">
        Muscle definition can only be scored where the scan shows your body, so this decides what to wear. Size gaps
        work in any fitted clothing.
      </Heading>

      <div className="space-y-5">
        <Panel>
          <fieldset>
            <legend>About you</legend>
            <TextField
              className="mt-5"
              fieldClassName="max-w-[11rem]"
              label="Age"
              inputMode="numeric"
              placeholder="e.g. 29"
              value={ageText}
              onChange={(e) => setAgeText(e.target.value)}
              hint="Used only to decide which options are offered."
            />
            <p className="mt-7 text-[0.7rem] font-medium uppercase tracking-[0.18em] text-ink-3" id="sex-label">
              Sex
            </p>
            <div className="mt-2 grid gap-3 sm:grid-cols-3" role="radiogroup" aria-labelledby="sex-label">
              {SEXES.map((s) => (
                <Choice key={s.value} compact type="radio" name="sex" checked={sex === s.value} onChange={() => setSex(s.value)} title={s.label} />
              ))}
            </div>
            <p className="mt-3 text-sm text-ink-3">
              Needed for body-fat %, calories, back width and the V-taper ceiling. Never guessed or filled in for you.
            </p>
            {sexUnsetNote(sex) && <p className="mt-2 text-sm text-caution">{sexUnsetNote(sex)}</p>}
          </fieldset>
        </Panel>

        <Panel>
          <fieldset>
            <legend>Scan option</legend>
            <div className="mt-5 space-y-3">
              {offered.map((s) => (
                <Choice key={s} type="radio" name="scope" checked={scope === s} onChange={() => setPicked(s)} title={SCOPE_TEXT[s].title} description={SCOPE_TEXT[s].blurb} />
              ))}
            </div>
            {!offered.includes("full_body") && <p className="mt-4 text-sm text-ink-3">{ADULT_ONLY_LINE}</p>}
          </fieldset>
        </Panel>

        {scope && (
          <Panel className="border-accent/40" >
            <div aria-live="polite">
              <h2>Before the camera starts</h2>
              <p className="mt-3 leading-relaxed text-ink-2">{clothingGuidance(scope, torso)}</p>
              {scope === "full_body" && (
                <div className="mt-5 grid gap-3 sm:grid-cols-2" role="radiogroup" aria-label="Torso clothing">
                  {TORSOS.map((t) => (
                    <Choice key={t.value} compact type="radio" name="torso" checked={torso === t.value} onChange={() => setTorsoPick(t.value)} title={t.label} />
                  ))}
                </div>
              )}
              {/* Opt-in, off by default. Shown to everyone since the owner's 2026-11-03 decision; still experimental (ADR-059). */}
              {scope === "full_body" && (
                <div className="mt-5">
                  <Choice
                    type="checkbox"
                    checked={onDevice}
                    onChange={() => setOnDevice(!onDevice)}
                    title={<>Score my leanness &amp; muscularity on this device (experimental, off by default)</>}
                    description="A small model runs in this browser on the front photo of you and of your goal, then the photos are released. The two bands stay on this device. Experimental: not yet validated on real photos. Without this, size gaps still work and nothing is scored."
                  />
                </div>
              )}
              {chestApproximate(scope, torso) && (
                <p className="mt-4 text-sm text-caution">With a sports bra the chest estimate is labelled approximate.</p>
              )}
              <p className="mt-5 text-sm leading-relaxed text-ink-2">
                {SCOPE_REGIONS[scope].length
                  ? "One 4-view turn scan in this clothing feeds the size gaps. A leanness & muscularity comparison with your goal (experimental) is planned for a later step."
                  : scope === "none"
                    ? "Definition will be marked not scored, with the reason: clothed scan."
                    : "Definition will be marked not scored, with the reason: this scan does not show the abdomen."}
              </p>
              <div className="mt-5">
                <Notice title="Your privacy">{PRIVACY_COPY}</Notice>
              </div>
              {sex === undefined && <p className="mt-4 text-sm text-caution">Choose an option under Sex to continue.</p>}
            </div>
          </Panel>
        )}
      </div>

      <Item className="mt-8 flex flex-wrap items-center gap-3">
        <button className="btn btn-primary disabled:opacity-40" disabled={!ready} onClick={() => scope && start(scope)}>
          Start scan
        </button>
        {scope && scope !== "none" && (
          <button className="btn btn-ghost disabled:opacity-40" disabled={sex === undefined} onClick={() => start("none")}>
            Use clothed scan instead
          </button>
        )}
        <button className="btn btn-ghost" onClick={onBack}>
          Back
        </button>
      </Item>
    </Screen>
  );
}
