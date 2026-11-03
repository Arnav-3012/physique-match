# P6 "models": the planner, the checker and the language model

P6 trained no new neural network. It used one language model (LLM) and built two non-learned algorithms that matter just as much: the planner and the checker. This file opens each one. Numbers come from `docs/logs/dev-log.md`, the ADRs and the repo; unmeasured items are marked.

---

## 1. The planner (an algorithm, not a model)

**Plain picture.** A packing problem: 15 muscle buckets must each reach 4, using exercises that pour into several buckets at once, spread over a few days, without overfilling any bucket or any day.

**Inputs.** The user's profile (days a week, equipment, level), the chosen priority regions, the candidate lists (top 10 exercises per muscle).
**Output.** For each day, a list of exercises with 3 sets each, the weekly total per muscle, and a list of muscles that fell short.

**The parameters (all ours, fixed before any plan existed, ADR-055):**

| Name | Value | Meaning |
|---|---|---|
| `SETS_PER_EXERCISE` | 3 | sets every exercise gets |
| `MAX_EXERCISES_PER_DAY` | 9 | the largest day of the old 3-day template |
| `EXTRA_SETS_PER_PRIORITY_GROUP` | 2 | extra weekly sets for priority regions' muscles |
| `MAX_WEEKLY_SETS_PER_GROUP` | 10 | credited sets per muscle per week |
| `MAX_SETS_PER_GROUP_PER_SESSION` | 8 | per muscle per day (lower of this and the cited 11) |
| `MAX_SESSIONS_PER_GROUP_PER_WEEK` | 4 | days a muscle may be trained |
| `TOP_K` | 10 | candidates per muscle |

Cited numbers (read from `kb/baselines.json`, never in code): the floor of **4** fractional sets per week for each of **15** muscle groups; the counting rule **primary 1, secondary 0.5**; the per-session soft cap **11**.

**Pseudo-code (`app/planner/plan.py`).**
```
week = {muscle: 0 for each of 15}; days = empty lists
loop (pass 1, floors):
    unmet = muscles with week < 4
    live  = for each unmet muscle, its candidates that fit somewhere right now
    stop if no muscle has any fitting candidate
    muscle = the one with the fewest fitting candidates
             (tie: larger remaining deficit, then name)
    exercise = argmax over its fitting candidates of
               (useful credit, better search rank, smaller id)
    put it on the eligible day with the fewest sets for that muscle
for muscle in file order (pass 2, extras): same fill, target = floor + 2 for priority muscles
for each still-empty day: add one fitting exercise from the day's focus muscles
shortfalls = muscles still below 4  → named, never hidden
check(...)  # independent verification, see section 2
```

"Fits" means: the day has fewer than 9 exercises, does not already hold this exercise, the muscle (and every muscle it splashes into) stays within the per-session and weekly caps, and no muscle exceeds 4 training days.

**Day layouts.** 1 day: full body. 2 or 3 days: full body each day. 4: upper, lower, upper, lower. 5: push, pull, lower, upper, lower. 6: push, pull, lower, twice. 7: those six plus a light day (abdominals, calves, forearms, traps). An exercise chosen for a muscle may go only on a day whose focus includes that muscle.

**Why this design, and what was rejected.**
- *Greedy fill* is simple and explainable; an exact solver could find better plans but is harder to audit and to explain. A solver was used once, offline, to prove `p14` is feasible (minimum total shortfall 0.00 over 123 and over 305 candidate exercises).
- *The model chooses the whole plan*: rejected, floors and caps need a guarantee.
- *A fixed order of muscles*: tried first, failed (the `adductors`/`glutes` cap collision, 15 failing tests); then a static scarcest-first order failed `p14` on `calves`; then a dynamic order with name tie-break failed `p14` on `triceps` and `quadriceps`; the order with "larger remaining deficit first" passed all 13 must-be-met cases.

**What breaks it.** Few exercises for a muscle (for a body-weight-only user the candidate lists in the fixtures are empty for abductors, adductors, biceps, calves, forearms, hamstrings, middle back, shoulders and traps, so `p10` ends with named shortfalls); a very short week (1 day: `p12` ends with named shortfalls); a greedy miss (the checker catches it as a named shortfall). In my scratch measurement each of `p10` and `p12` ended with 7 named groups using a slightly different tie-break from the final code; the final code's counts were not printed, so treat 7 as indicative only.

**Cost.** Not measured: the planner's runtime and the end-to-end latency of `/plan`.

---

## 2. The checker (`app/planner/check.py`)

**Plain picture.** A second scale. It does not read the planner's totals; it weighs the plan again.

**Algorithm.**
```
read floors and the counting rule from kb/baselines.json itself
for each day: add primary (1.0 x sets) and secondary (0.5 x sets) credit per muscle
raise if: a day has more than 9 exercises or a repeat
          a muscle's day load exceeds the session limit
          a muscle exceeds 10 weekly sets or 4 training days
          a muscle is below its floor and not in the shortfall list
          a named shortfall's numbers disagree with the recomputed ones
```
It is used twice: on the planner's result, and on the LLM's plan (with the extra rule that the LLM's short muscles must be a subset of the plain plan's).

---

## 3. The language model (the writer)

**What it is.** `openai/gpt-oss-120b`, called through Groq's OpenAI-compatible chat API over `httpx` (our client in `app/agent/llm.py`; no vendor SDK). The backup provider is OpenRouter with a no-training preference. **Not checked here:** the model's licence terms and its parameter details beyond the name.

**Request shape.**
```
POST {base}/chat/completions
{ model, messages [system, user], temperature 0,
  response_format: { type "json_schema", strict true, schema <WriterOutput> } }
```
`WriterOutput` (in `app/agent/schema.py`): `summary` (≤ 500 characters) and 1-7 `days`, each with a `label` (≤ 40) and 1-9 `items` of `{exercise_id, reps matching "10" or "8-12", cue ≤ 160 characters}`. Extra fields are rejected, and **there is no sets field**.

**The facts block (user message).** One JSON object: `days` count, the priority regions with their gap as a percent of height and reliability as a percent, the muscles short of their minimum, optional body-fat range / protein / calorie bands, `options_by_group` (3 ids per muscle), `plan_days` (each day's focus and baseline ids) and `exercises` (each id once: name, main muscle, the first 100 characters of its instructions). No user free text; no client `reason` text.

**Measured sizes.** First design: 94,225 characters, Groq reported about 24,400 tokens requested, limit 8,000 per minute → HTTP 413. Now: 13,150 characters for the p01 probe request (HTTP 200). A test keeps the prompt under 20,000 characters (our number, about 5,000 tokens). Tokens actually consumed per accepted report: not measured (Groq's rate message mentioned 4,000 requested for one call).

**The acceptance pipeline.**
```
raw text ─ parse as WriterOutput ─ day count ─ ids allowed for the day's focus, no repeats
   ─ weekly credit recomputed ─ shortfalls ⊆ plain plan's ─ check() ─ cue is not a metadata header
   ─ banned words / regional claims / drugs / URL / markdown / HTML ─ numbers are in the facts
   → accepted
any failure at any point → the plain plan, with one of: invalid_json, check_failed, unsafe_text
a provider failure     → timeout, rate_limited (HTTP 429 and Groq's 413), disabled (no key / flag off)
```
Call budget: at most 3 calls and 1 retry (retry only for `timeout` or `rate_limited`).

**What breaks it.**
- A prompt over the provider's token-per-minute limit (seen).
- Giving the model the wrong text to copy from (seen: the metadata header became the cue).
- Rate limits: the free tier gives roughly one or two reports a minute across all users (derived from 8,000 tokens a minute and about 4,000 per report; not load-tested).
- The writer plan breaking a floor or cap (rejected by `check()`); `p03` falls back with `check_failed` in two live runs, cause undiagnosed.

---

## 4. The retriever (reused from P5)

The candidate lists come from the P5 hybrid retriever: BM25 top-20 and bge-small (int8 ONNX, 384 numbers per chunk) top-20 combined by Reciprocal Rank Fusion with k = 60; see `learn/phase-05/`. P6 changed nothing in it except loading it once at startup with a BM25-only fallback. Planner tests read committed lists (`tests/fixtures/planner_ranks_*.json`, 4 profiles each) so CI needs no model.

---

## Glossary (this file)

- **Credit:** weekly sets counted toward a muscle (1 per set if primary, 0.5 if secondary).
- **Most-constrained-first:** fill the muscle with the fewest remaining options first.
- **Structured output / JSON schema:** forcing the model's answer into a fixed shape.
- **Facts block:** the JSON we give the model; the only information it has.
- **Temperature 0:** the model picks its most likely words (as repeatable as these services allow).
- **TPM:** tokens per minute, the provider's rate limit.

## Viva questions for this file

1. Write the planner's pass 1 in your own words. Why most-constrained-first?
2. What does the checker read, and why is it not the planner's own numbers?
3. What is in the facts block, and what is deliberately absent?
4. Why does the schema have no sets field?
5. Trace a model answer that cites an exercise from another day's focus: which check rejects it, and what does the user receive?
