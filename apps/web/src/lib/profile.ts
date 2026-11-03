import type { Activity, Equipment, Goal, Level, Sex, UserProfile } from "./types.gen";

/** The closed equipment checklist (ADR-054): the 12 values the planner pool uses. Labels are display only. */
export const EQUIPMENT: { value: Equipment[number]; label: string }[] = [
  { value: "body only", label: "Body weight only" },
  { value: "dumbbell", label: "Dumbbells" },
  { value: "barbell", label: "Barbell" },
  { value: "machine", label: "Machines" },
  { value: "cable", label: "Cables" },
  { value: "kettlebells", label: "Kettlebells" },
  { value: "bands", label: "Resistance bands" },
  { value: "e-z curl bar", label: "E-Z curl bar" },
  { value: "exercise ball", label: "Exercise ball" },
  { value: "medicine ball", label: "Medicine ball" },
  { value: "other", label: "Other equipment" },
  { value: "none", label: "No equipment at all" },
];

export const ACTIVITY: { value: Activity; title: string; text: string }[] = [
  { value: "sedentary", title: "Sedentary", text: "Little or no exercise, desk-based days." },
  { value: "light", title: "Lightly active", text: "Light exercise or sport 1 to 3 days a week." },
  { value: "moderate", title: "Moderately active", text: "Moderate exercise or sport 3 to 5 days a week." },
  { value: "active", title: "Very active", text: "Hard exercise or sport 6 to 7 days a week." },
  { value: "very_active", title: "Extremely active", text: "Hard daily training or a physical job." },
];

export const LEVELS: { value: Level; title: string; text: string }[] = [
  { value: "beginner", title: "Beginner", text: "New to lifting, or back after a long break." },
  { value: "intermediate", title: "Intermediate", text: "Training steadily for a year or more." },
  { value: "advanced", title: "Advanced", text: "Several years of structured training." },
];

export const GOALS: { value: Goal; title: string; text: string }[] = [
  { value: "cut", title: "Cut", text: "Lose fat while keeping muscle." },
  { value: "recomp", title: "Recomp", text: "Stay near maintenance while the shape changes." },
  { value: "bulk", title: "Bulk", text: "Build size with a small surplus." },
];

/** Form state as typed. Numbers stay strings until `toProfile` so a half-typed value is never a number. */
export interface ProfileForm {
  height: string;
  weight: string;
  age: string;
  activity?: Activity;
  level?: Level;
  goal?: Goal;
  days: number;
  equipment: Equipment[number][];
  ed?: "yes" | "no" | "skip";
  tape: string;
}

export const emptyForm = (age?: number): ProfileForm => ({
  height: "",
  weight: "",
  age: age === undefined ? "" : String(age),
  days: 3,
  equipment: [],
  tape: "",
});

const num = (s: string) => (s.trim() === "" ? NaN : Number(s));

/** Client-side bounds are sanity checks only (ours, not clinical); the API stays the authority. */
export function toProfile(f: ProfileForm, sex: Sex): { profile?: UserProfile; errors: Record<string, string> } {
  const errors: Record<string, string> = {};
  const height = num(f.height);
  const weight = num(f.weight);
  const age = num(f.age);
  const tape = num(f.tape);
  if (!(height >= 120 && height <= 230)) errors.height = "Enter your height in cm (120 to 230).";
  if (!(weight >= 30 && weight <= 250)) errors.weight = "Enter your weight in kg (30 to 250).";
  if (!(Number.isInteger(age) && age >= 10 && age <= 100)) errors.age = "Enter your age in whole years.";
  if (f.tape.trim() !== "" && !(tape >= 40 && tape <= 200)) errors.tape = "Enter your waist in cm (40 to 200), or leave it empty.";
  if (!f.activity) errors.activity = "Choose how active you are.";
  if (!f.level) errors.level = "Choose your training level.";
  if (!f.goal) errors.goal = "Choose what you want to do.";
  if (f.equipment.length === 0) errors.equipment = "Tick at least one kind of equipment.";
  if (Object.keys(errors).length) return { errors };
  return {
    errors,
    profile: {
      height_cm: height,
      weight_kg: weight,
      age,
      sex,
      activity: f.activity!,
      level: f.level!,
      goal: f.goal!,
      days_per_week: f.days,
      equipment: f.equipment,
      ed_history: f.ed === "yes" ? true : f.ed === "no" ? false : null,
      tape_waist_cm: f.tape.trim() === "" ? null : tape,
    },
  };
}
