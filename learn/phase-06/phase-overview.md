# P6 overview, from zero: the planner, the `/plan` endpoint and the writer

**Status:** closed on 2026-11-03 at demo scope. Some gate items were **not met as specified**; they are listed in section 6, not hidden.

**Who this is for.** You do not need the other files first. This one tells the story: what we built, why, what we measured, what went wrong. `concepts.md` explains each idea step by step; `model-architecture.md` opens the algorithm and the language model.

---

## 1. The problem P6 solves

By the end of P5 the app could *measure* you (size gaps, body-fat range) and *search* a library of exercises. P6 turns that into **a weekly workout plan a person can follow**, and puts a safety net around it.

A plan has to be three things at once:

1. **Correct.** Every muscle must get at least the weekly volume that research supports (the "floor", 4 sets a week per muscle group). No muscle may be silently skipped.
2. **Safe.** No calorie target below safe limits, nothing for someone with an eating-disorder history, no shaming words, no drug advice.
3. **Readable.** Sentences a person understands, not a spreadsheet.

The first two are *rules*, so a program checks them. The third is *wording*, which a language model (LLM) is good at, but a language model can make mistakes. So the design is: **a plain program decides the plan, the LLM only helps with wording and exercise variety, and the program checks everything the LLM did before it is shown.**

---

## 2. What we built, in order

| Step | What | Where | Result |
|---|---|---|---|
| 1 | The exam written and frozen before any planner existed | `evals/plan.jsonl`, `evals/safety.jsonl`, `evals/p6-README.md`, `evals/patterns.py`, `evals/pattern_checks.jsonl` | 15 plan cases, 40 safety cases, 5 files fingerprinted |
| 2 | Equipment became a closed checklist; errors never echo the input | `physique_schemas/models.py` (`Equipment`), `app/errors.py` | 12 allowed values |
| 3 | Picking the priority regions | `app/planner/priorities.py` | up to 3 regions, gap at least 0.010 |
| 4 | Candidate exercises per muscle group | `app/planner/candidates.py` | top 10 per group |
| 5 | The deterministic weekly planner | `app/planner/plan.py`, `params.py` | 13 of 13 "must be met" cases meet every floor |
| 6 | An independent checker for the plan | `app/planner/check.py` | raises on a silent shortfall or a broken cap |
| 7 | Body composition recomputed server-side and a safety re-check | `app/bodycomp.py`, `app/safety/rules.py` | 20% / 10% / 1% / BMI / body-fat limits |
| 8 | The `/plan` endpoint | `app/routes/plan.py` | one request in, one plan out |
| 9 | The LLM writer and its acceptance checks | `app/agent/*` | off by default, closed list of fallback reasons |
| 10 | Realism note and fixed warnings | `app/planner/realism.py` | threshold 0.05 |
| 11 | Gate suites and live runs | `evals/run.py`, `plan_suite.py`, `safety_suite.py`, `probe_llm.py` | see section 5 |

---

## 3. The data flow, plainly

```
PlanRequest  (profile with a CLOSED equipment checklist, the size gaps from /analyze,
              optional waist number, optional goal sex)
   │  wrong shape or unknown word → 422, and the error never repeats what you sent
   ▼
priorities           which up to 3 regions have the biggest positive gap?
   ▼
candidates           for each of 15 muscle groups: the top 10 matching exercises from the search
   ▼
planner (no AI)      fill every muscle to its floor, then add extras for priority regions,
   │                 under caps; result = days → exercises → 3 sets each
   ▼
checker (no AI)      recompute every muscle from the final plan; raise if anything is wrong
   ▼
body numbers         recomputed from your profile; safety re-check of calorie rules
   ▼
writer (AI, optional) picks exercises BY ID from the candidate lists and writes wording
   │                 accepted only if every check passes; otherwise the plain plan is served
   ▼
PlanResponse         days, weekly totals, short groups, notes, warnings, report, and which
                     path wrote it (primary / fallback / deterministic) with a reason
```

---

## 4. The decisions that mattered (and why)

| Decision | Why | Alternative rejected |
|---|---|---|
| Freeze the exam first (fingerprints) | So nobody can tune the exam to make the planner pass | Write cases after seeing plans |
| Planner is plain code, not an AI | Floors and caps are hard rules; an AI cannot guarantee them | Let the LLM write the whole plan |
| The LLM may only pick **ids from lists**, and has no "sets" field | It cannot raise or lower any volume | Free-form plan text |
| Whole LLM answer accepted or thrown away | Partial fixes could hide a wrong plan | Patch the answer |
| Equipment is a checklist | Free text is where hostile text enters | Typed equipment, filtered |
| Fixed code-written text for notes, warnings, disclaimer | The model cannot misstate safety text | Let the model write them |
| Never lower a floor to pass | A shortfall is reported, not hidden | Round up, or lower the number |
| `LLM_WRITER_ENABLED` defaults to false | The app works without any model | Writer always on |

---

## 5. Metrics achieved versus the gate (real numbers)

| Gate item | Result | Verdict |
|---|---|---|
| 100% schema-valid | 15 of 15 (the plain fallback plan is valid by construction) | met |
| Every exercise cites a retrieved chunk | 15 of 15 by code | met |
| Baseline-volume invariant | 13 of 13 "must be met" cases meet every floor; `p10` (body weight only) and `p12` (one day) end with **named** shortfalls | met |
| Safety suite | offline: 36 pass, 0 fail, **4 need the owner's scoring** (`supportive_note`) | not claimable as 40 of 40 |
| Plan rubric ≥ 4 of 5 | the four code items passed 15 of 15; "readable" was to be scored by the owner, blind | **not met as specified** |
| Live LLM writer | first full run: **0 of 15** served by the model; second: 13 of 15; subset after the cue fix: 2 of 4 | partial |
| "No number differs from its source" | enforced on the model's prose by a check; not measured separately | not measured |

**Readability:** only a rating by Claude exists, on 2 plans, both "readable". That is n = 2, it is not the owner's blind scoring, and it is not a pass rate.

Test counts from the owner's runs: the last full suite I saw was **776 passed, 1 skipped**. It has not been re-run since the prompt and cue fixes.

---

## 6. What went wrong (the useful part)

1. **The planner failed 15 tests the first time.** All for one reason: the muscle `adductors` had only one exercise available, and it would have pushed `glutes` from 9.0 to 10.5 sets, over the weekly cap of 10. The planner refused, correctly. Fix: change the *order* in which muscles are filled (most-constrained first). The fixed order I first measured (scarcest group first) still left one case (`p14`, a 2-day plan) short on `calves`. A dynamic order with ties broken by group name also left `p14` short (`triceps` and `quadriceps` at 3.0 of 4). Breaking ties by the larger remaining deficit first, then group name, met every floor. The order was chosen **after** seeing failures, so passing is evidence about these 15 profiles, not a guarantee.
2. **The first live run served 0 of 15 plans by the model.** The prompt was 94,225 characters; the provider's free tier allows 8,000 tokens a minute; the request was refused (HTTP 413), and my code filed that under "bad JSON". Fix: a smaller prompt (13,150 characters) and a separate reason.
3. **The model's "cues" were truncated database text.** I had given it the metadata header instead of the instructions. Every automated check passed anyway; only reading the output showed it.
4. **A formatter rewrote a frozen file.** The pre-commit hook changed the layout of `evals/patterns.py`; the regexes were verified identical, and an amendment was recorded.
5. **A tag landed on the wrong commit** because I did not chain `git tag` to `git commit`.
6. **I printed a blind-scoring key** into the transcript.

Details: `docs/mistakes.md`.

---

## 7. Skipped or not measured (kept as TBD)

Report screen and chat-style UI (moved to P7); the Dockerfile for `/plan` with the model fetched at build; the extra adductor exercise in the knowledge base; any change to a planner number (needs its own ADR); the owner's blind readability scoring; the 4 `supportive_note` scores; why `p03` falls back (`check_failed`, reproducible, cause not diagnosed); end-to-end latency and cost of `/plan` (**never measured**); whether the OpenRouter fallback works live (it served 0 plans).

---

## 8. Five interview questions with answers

**1. Why not let the AI write the plan?** A plan has hard rules: every muscle at least 4 sets a week, no muscle over 10, no calorie target outside safe limits. A language model cannot guarantee that; a program can, and a second independent program can verify it. The AI does what it is good at (wording, variety) inside rules it cannot change.

**2. How do you know the AI did not break a rule?** Its output is only accepted if it parses, every exercise id is from the allowed list for that day, an independent checker recomputes every muscle's weekly volume from its exercises with no more short groups than the plain plan, and the text passes banned-word, URL and number checks. Otherwise the plain plan is served and the reason is one of six fixed words.

**3. What was the hardest bug?** The prompt size. It looked like "bad JSON" in the results; a diagnostic that printed the HTTP status showed a 413 from the token limit. The lesson: do not fold different failures into one label, and measure before the first live run.

**4. How did you avoid tuning the exam to the code?** The exam files were fingerprinted before the planner existed; a change needs a written amendment with old and new fingerprints. When a case disagreed with the planner, the rule was to report it, not edit either side. I did amend one fingerprint, after a formatter changed layout only, and proved the regexes identical first.

**5. What does the planner not guarantee?** It is greedy and made no optimality claim; it was chosen after seeing failures; a solver showed `p14` is feasible but that does not prove the greedy always finds a plan. The independent checker is the backstop: a miss is always a *named* shortfall, never silent.

---

## Glossary (this file)

- **Floor:** the minimum weekly sets per muscle group (4), from a cited review.
- **Shortfall:** a muscle group that ends below its floor, listed by name.
- **Deterministic:** same input, same output, no AI.
- **Fallback:** serving the plain plan because the AI path failed a check or call.
- **Frozen:** fingerprinted so it cannot be changed without a written amendment.
