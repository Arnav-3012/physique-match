import { REGION_LABEL, type GoalRegion } from "../lib/goalInputs";
import type { AnalyzeResponse, PlanResponse, UserProfile } from "../lib/types.gen";

/** What a chat answer may use: the user's own results. No free text ever reaches a model from here (ADR-059). */
export interface Facts {
  analysis: AnalyzeResponse;
  plan: PlanResponse | null;
  profile: UserProfile;
  focus?: string[]; // regions the user asked to prioritise (ADR-060)
}

export const regionLabel = (key: string) => REGION_LABEL[key as GoalRegion] ?? key;

export const QUESTIONS = [
  { id: "why_plan", text: "Why this plan?" },
  { id: "why_exercises", text: "Why these exercises?" },
  { id: "minimum", text: "What is a typical minimum?" },
  { id: "calories", text: "Why are there no calorie numbers?" },
  { id: "accuracy", text: "How accurate are the gaps?" },
  { id: "back_width", text: "Why is back width an estimate?" },
  { id: "realistic", text: "Is my goal realistic?" },
] as const;

export type QuestionId = (typeof QUESTIONS)[number]["id"];

const list = (xs: string[]) => (xs.length <= 1 ? xs.join("") : `${xs.slice(0, -1).join(", ")} and ${xs.at(-1)}`);

export function answer(id: QuestionId, f: Facts): string[] {
  const { analysis, plan, profile } = f;
  switch (id) {
    case "why_plan": {
      if (!plan) return ["The plan could not be built just now, so there is nothing to explain yet. Try again in a moment."];
      const top = plan.priorities.map(regionLabel);
      return [
        `You asked for ${profile.days_per_week} training day${profile.days_per_week === 1 ? "" : "s"} a week, so the plan is split to fit that.`,
        top.length
          ? `The extra weekly sets go to ${list(top)}, ${
              f.focus?.length ? "the areas you asked to prioritise (at most three get extra volume)" : "the regions where your scan showed the largest measured gaps"
            }. Every other muscle group still gets its weekly minimum.`
          : "No region stood out as a clear priority, so volume is spread evenly at each group's minimum.",
        ...(plan.note ? [plan.note] : []),
      ];
    }
    case "why_exercises":
      return [
        "Exercises are picked from a library of strength exercises, filtered to the equipment you ticked and your experience level.",
        "For each muscle group the options are ranked by how well their descriptions match what that group needs, and the plan lists the sources it used.",
      ];
    case "minimum": {
      const short = plan?.shortfalls ?? [];
      return [
        "Each muscle group has a typical minimum number of hard weekly sets, taken from published research. The plan never goes below it for any group, even one where your scan found no gap.",
        short.length
          ? `With your days and equipment, ${list(short.map((s) => s.group))} could not reach their minimum. They are named in your plan instead of being hidden, and the minimum is not lowered.`
          : "Every group reaches its minimum in your plan.",
      ];
    }
    case "calories": {
      const bc = plan?.body_comp ?? analysis.body_comp;
      if (bc?.energy?.length)
        return [
          "Your calorie targets are shown as ranges. A deficit never exceeds 20% of your maintenance calories, a surplus never exceeds 10%, and a target never goes below your estimated resting rate.",
        ];
      const why = [plan?.body_comp_note, bc?.note].find((x) => x && x.trim());
      if (profile.ed_history === true)
        return ["Because you told us about an eating-disorder history, we show training only and no calorie numbers. Support from a professional matters more than any number here."];
      if (profile.sex === "not_specified")
        return ["Calorie targets need your sex, because the formulas differ. You left it unset, so they are hidden. Your size gaps and training plan still work."];
      return [why ?? "Calorie targets need a waist measurement. Add a tape measurement and we can show them."];
    }
    case "accuracy":
      return [
        "Gaps are measured from outlines, not from tape, so they carry several centimetres of noise from posture, clothing and camera position.",
        "That is why every region shows a confidence meter. Treat a small gap as roughly on target, and a region with low confidence as a hint to re-scan or measure by tape.",
        ...(analysis.gap.goal_front_only ? ["Your goal photo is front-only, so girth regions are less certain than widths."] : []),
      ];
    case "back_width":
      return [
        "Back width is inferred from your chest, shoulder and waist ratios, not measured. It is not a measure of lat muscle, so it is never ranked as a priority.",
        "Upper back size cannot be measured from a front photo, and a back silhouette adds no size information.",
      ];
    case "realistic": {
      const note = plan?.report?.realism_note;
      return [
        note ?? "Nothing in your numbers suggested that a gap is beyond what your frame allows.",
        "Goal photos are often posed, pumped, lit or edited, so use them as a direction, not a promise.",
      ];
    }
  }
}
