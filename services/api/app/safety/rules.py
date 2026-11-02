"""Independent safety re-check of a /plan result (docs/privacy-and-safety.md, "Body-fat % and calorie
rules"). It imports nothing from the body-composition code: the limits are restated on purpose, and
tests/test_safety_rules.py pins each one to that code so drift fails loudly. A breach raises; the route
turns it into a generic 500 and emits nothing."""

from physique_schemas.models import KCAL_TOLERANCE, BodyComp, UserProfile

MAX_DEFICIT = 0.20  # of TDEE; never below RMR
MAX_SURPLUS = 0.10  # of TDEE
MAX_LOSS_PER_WEEK = 0.01  # of body weight
KCAL_PER_KG = 7700  # the conversion bodycomp.py uses for the weekly-loss cap
MIN_BMI = 18.5
BF_FLOOR = {"male": 8.0, "female": 18.0}  # the 18 is our rule (evals/p6-README.md)
ADULT_AGE = 18
CUT_LIKE = {"cut", "recomp"}  # goals that need a deficit


class SafetyBreach(Exception):
    pass


def check_plan_safety(profile: UserProfile, comp: BodyComp | None) -> None:
    """Raise SafetyBreach listing every rule the result breaks. `None` carries no numbers at all."""
    if comp is None:
        return
    problems = []
    no_numbers = profile.ed_history or profile.age < ADULT_AGE or profile.sex == "not_specified"
    if no_numbers and (comp.energy or comp.rmr_kcal is not None or comp.tdee_kcal is not None):
        problems.append("calorie numbers given where the profile allows none")
    if profile.ed_history and comp.protein is not None:
        problems.append("protein numbers given with an eating-disorder history")
    if profile.sex == "not_specified" and comp.body_fat is not None:
        problems.append("body fat given without a sex")

    tol = KCAL_TOLERANCE
    if comp.energy and (comp.rmr_kcal is None or comp.tdee_kcal is None):
        problems.append("energy targets without rmr and tdee")
    for t in comp.energy if comp.rmr_kcal is not None and comp.tdee_kcal is not None else []:
        tdee, rmr = comp.tdee_kcal, comp.rmr_kcal
        if t.kcal_low < max(rmr, (1 - MAX_DEFICIT) * tdee) - tol:
            problems.append(f"{t.goal}: {t.kcal_low} kcal under the deficit cap or RMR")
        if t.kcal_high > (1 + MAX_SURPLUS) * tdee + tol:
            problems.append(f"{t.goal}: {t.kcal_high} kcal over the surplus cap")
        loss_kg_week = max(0.0, tdee - t.kcal_low) * 7 / KCAL_PER_KG
        if loss_kg_week > MAX_LOSS_PER_WEEK * profile.weight_kg + tol * 7 / KCAL_PER_KG:
            problems.append(f"{t.goal}: loss of {loss_kg_week:.2f} kg/week over the cap")
        if t.goal not in {"maintain", profile.goal}:
            problems.append(f"{t.goal}: not the user's own choice ({profile.goal})")

    if comp.body_fat is not None and profile.sex in BF_FLOOR:
        bmi = profile.weight_kg / (profile.height_cm / 100) ** 2
        mid = (comp.body_fat.low + comp.body_fat.high) / 2
        if (mid < BF_FLOOR[profile.sex] or bmi < MIN_BMI) and any(t.goal in CUT_LIKE for t in comp.energy):
            problems.append("a deficit target below the BMI or body-fat floor")
    if problems:
        raise SafetyBreach("; ".join(problems))
