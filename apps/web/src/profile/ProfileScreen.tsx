import { useState } from "react";
import type { Sex } from "../capture/scope";
import { sexUnsetNote } from "../capture/scope";
import { ACTIVITY, EQUIPMENT, GOALS, LEVELS, toProfile, type ProfileForm } from "../lib/profile";
import type { UserProfile } from "../lib/types.gen";
import Choice from "../ui/Choice";
import { Heading, Item, Notice, Panel, Screen } from "../ui/Layout";
import SideFigure from "../ui/SideFigure";
import TextField from "../ui/TextField";

interface Props {
  form: ProfileForm;
  onForm: (f: ProfileForm) => void;
  sex: Sex;
  onBack: () => void;
  onSubmit: (p: UserProfile) => void;
}

const Err = ({ msg }: { msg?: string }) => (msg ? <p className="mt-2 text-sm text-caution">{msg}</p> : null);

/** The numbers behind the body-fat range, calories and plan. Typed here, sent as numbers, never stored. */
export default function ProfileScreen({ form, onForm, sex, onBack, onSubmit }: Props) {
  const [errors, setErrors] = useState<Record<string, string>>({});
  const set = (patch: Partial<ProfileForm>) => onForm({ ...form, ...patch });

  const submit = () => {
    const r = toProfile(form, sex);
    setErrors(r.errors);
    if (r.profile) onSubmit(r.profile);
  };

  const toggleEquipment = (v: ProfileForm["equipment"][number]) =>
    set({ equipment: form.equipment.includes(v) ? form.equipment.filter((x) => x !== v) : [...form.equipment, v] });

  return (
    <Screen labelledBy="profile-h" aside={<SideFigure caption="Your numbers turn outlines into centimetres" />}>
      <Heading eyebrow="Step 6 · About you" title="The numbers behind your plan" id="profile-h">
        Height and weight give your gaps a scale and your body-fat range and calories their starting point. They are
        sent as numbers only and are not stored.
      </Heading>

      <div className="space-y-5">
        <Panel>
          <fieldset>
            <legend>Your body</legend>
            <div className="mt-5 grid gap-5 sm:grid-cols-3">
              <div>
                <TextField label="Height" suffix="cm" inputMode="decimal" placeholder="175" value={form.height} onChange={(e) => set({ height: e.target.value })} />
                <Err msg={errors.height} />
              </div>
              <div>
                <TextField label="Weight" suffix="kg" inputMode="decimal" placeholder="72" value={form.weight} onChange={(e) => set({ weight: e.target.value })} />
                <Err msg={errors.weight} />
              </div>
              <div>
                <TextField label="Age" suffix="yrs" inputMode="numeric" placeholder="29" value={form.age} onChange={(e) => set({ age: e.target.value })} />
                <Err msg={errors.age} />
              </div>
            </div>
            <div className="mt-6">
              <TextField
                label="Waist (optional)"
                suffix="cm"
                inputMode="decimal"
                placeholder="e.g. 82"
                fieldClassName="max-w-[14rem]"
                hint="If you add a tape measurement it replaces the waist we estimate from your scan."
                value={form.tape}
                onChange={(e) => set({ tape: e.target.value })}
              />
              <Err msg={errors.tape} />
            </div>
            {sexUnsetNote(sex) && (
              <div className="mt-6">
                <Notice tone="caution">{sexUnsetNote(sex)}</Notice>
              </div>
            )}
          </fieldset>
        </Panel>

        <Panel>
          <fieldset>
            <legend>How active are you?</legend>
            <div className="mt-5 space-y-3" role="radiogroup" aria-label="Activity level">
              {ACTIVITY.map((a) => (
                <Choice key={a.value} compact type="radio" name="activity" checked={form.activity === a.value} onChange={() => set({ activity: a.value })} title={a.title} description={a.text} />
              ))}
            </div>
            <Err msg={errors.activity} />
          </fieldset>
        </Panel>

        <Panel>
          <fieldset>
            <legend>Your training</legend>
            <p className="mt-5 text-[0.7rem] font-medium uppercase tracking-[0.18em] text-ink-3" id="level-label">
              Experience
            </p>
            <div className="mt-2 grid gap-3 sm:grid-cols-3" role="radiogroup" aria-labelledby="level-label">
              {LEVELS.map((l) => (
                <Choice key={l.value} type="radio" name="level" checked={form.level === l.value} onChange={() => set({ level: l.value })} title={l.title} description={l.text} />
              ))}
            </div>
            <Err msg={errors.level} />

            <p className="mt-7 text-[0.7rem] font-medium uppercase tracking-[0.18em] text-ink-3" id="goal-label">
              What do you want to do?
            </p>
            <div className="mt-2 grid gap-3 sm:grid-cols-3" role="radiogroup" aria-labelledby="goal-label">
              {GOALS.map((g) => (
                <Choice key={g.value} type="radio" name="goal" checked={form.goal === g.value} onChange={() => set({ goal: g.value })} title={g.title} description={g.text} />
              ))}
            </div>
            <p className="mt-3 text-sm text-ink-3">Your choice. It is never worked out from your body-fat estimate.</p>
            <Err msg={errors.goal} />

            <div className="mt-7 flex items-center gap-4">
              <p className="text-[0.7rem] font-medium uppercase tracking-[0.18em] text-ink-3" id="days-label">
                Days a week
              </p>
              <div className="flex items-center gap-3" role="group" aria-labelledby="days-label">
                <button type="button" className="grid size-9 place-items-center rounded-full border border-line text-lg text-ink hover:border-accent disabled:opacity-40" disabled={form.days <= 1} onClick={() => set({ days: form.days - 1 })} aria-label="One day fewer">
                  −
                </button>
                <span className="display w-8 text-center text-3xl text-accent-2" aria-live="polite">
                  {form.days}
                </span>
                <button type="button" className="grid size-9 place-items-center rounded-full border border-line text-lg text-ink hover:border-accent disabled:opacity-40" disabled={form.days >= 7} onClick={() => set({ days: form.days + 1 })} aria-label="One day more">
                  +
                </button>
              </div>
            </div>
          </fieldset>
        </Panel>

        <Panel>
          <fieldset>
            <legend>Your equipment</legend>
            <div className="mt-2 flex flex-wrap items-center justify-between gap-3">
              <p className="text-sm text-ink-3">Tick everything you can use. The plan only picks from these.</p>
              <button
                type="button"
                className="rounded-full border border-accent/50 px-4 py-1.5 text-xs uppercase tracking-[0.14em] text-accent-2 transition hover:bg-accent/10"
                onClick={() => set({ equipment: form.equipment.length === EQUIPMENT.length ? [] : EQUIPMENT.map((e) => e.value) })}
              >
                {form.equipment.length === EQUIPMENT.length ? "Clear all" : "Select all"}
              </button>
            </div>
            <div className="mt-5 grid gap-3 sm:grid-cols-2">
              {EQUIPMENT.map((e) => (
                <Choice key={e.value} compact type="checkbox" checked={form.equipment.includes(e.value)} onChange={() => toggleEquipment(e.value)} title={e.label} />
              ))}
            </div>
            <Err msg={errors.equipment} />
          </fieldset>
        </Panel>

        <Panel>
          <fieldset>
            <legend>One optional question</legend>
            <p className="mt-2 text-sm leading-relaxed text-ink-2">Do you have a history of an eating disorder?</p>
            <div className="mt-4 grid gap-3 sm:grid-cols-3" role="radiogroup" aria-label="Eating-disorder history">
              {(
                [
                  ["no", "No"],
                  ["yes", "Yes"],
                  ["skip", "Prefer not to say"],
                ] as const
              ).map(([v, label]) => (
                <Choice key={v} compact type="radio" name="ed" checked={form.ed === v} onChange={() => set({ ed: v })} title={label} />
              ))}
            </div>
            <p className="mt-3 text-sm text-ink-3">A yes turns off calorie numbers and shows a supportive note instead. The training plan stays.</p>
          </fieldset>
        </Panel>
      </div>

      <Item className="mt-8 flex flex-wrap gap-3">
        <button className="btn btn-primary" onClick={submit}>
          See my results
        </button>
        <button className="btn btn-ghost" onClick={onBack}>
          Back
        </button>
      </Item>
      {Object.keys(errors).length > 0 && <p className="mt-3 text-sm text-caution">A few answers need your attention before we can continue.</p>}
    </Screen>
  );
}
