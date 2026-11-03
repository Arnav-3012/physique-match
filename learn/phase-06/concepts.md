# P6 concepts, from zero

Every idea P6 used, in the order the phase used it. Each step has: a plain picture, how it works, the algorithm or maths, our real file and numbers, what went wrong, and three quiz questions with answers. Numbers come from `docs/logs/dev-log.md` and the ADRs; anything not measured is said so.

---

## Step 1. Write the exam first, and fingerprint it

**Plain picture.** A teacher writes the exam *before* marking anyone's homework. If you write the questions after seeing the answers, you can make everyone pass. We wrote 15 "build me a plan for this person" cases and 40 "risky input" cases before the planner existed, then took a **fingerprint** of the files so any later change is visible.

**How it works.** A fingerprint (SHA-256) is a long string computed from a file's exact bytes. Change one character and it changes completely. We stored the fingerprints in the log. Changing a frozen file needs a written *amendment*: what changed, why, old and new fingerprint.

**Real files and numbers.**
- `evals/plan.jsonl` (15 profiles, fingerprint starts `30ec19e0`), `evals/safety.jsonl` (40 cases, `c0fb6383`), `evals/p6-README.md` (`c7250f2e`), `evals/pattern_checks.jsonl` (`fac4ffdd`), `evals/patterns.py` (`4a065a7c` after one amendment).
- Placeholders fixed in advance and never tuned: the priority minimum 0.010 (1% of height, untested), the female body-fat floor 18% (our rule, not a cited number), "a 1-day or body-weight-only plan may end with *cannot meet the floor, and here is why*".
- `tests/test_p6_evals.py` checks every case's shape and recomputes each case's expected priorities from the stated rule, so a wrong expectation fails before any planner exists.

**What went wrong.** The first fingerprint was taken before anything was validated, and the README still carried a status line. We withdrew that freeze, fixed it, and re-fingerprinted. Later, the pre-commit formatter rewrote the layout of `evals/patterns.py`; a script proved every regex identical and we recorded Amendment 1 in ADR-054.

**Quiz.**
1. *Why fingerprint instead of just "not touching" the files?* Because a promise cannot be checked; a fingerprint can.
2. *The formatter changed the fingerprint of `patterns.py`. Did the exam change?* No: the patterns are the same strings with the same flags, verified by comparing them; only the code layout changed. It still needed an amendment because the fingerprint is the proof.
3. *Why leave the 0.010 minimum marked "untested"?* No data chose it; saying so prevents it being mistaken for a finding.

---

## Step 2. A closed checklist instead of typed text

**Plain picture.** A restaurant order form with tick-boxes instead of a blank line. Nobody can write "ignore the kitchen rules" in a tick-box.

**How it works.** `UserProfile.equipment` is a list of values from a fixed set. A value outside the set is rejected with a 422 error. The error says which field was wrong and what kind of error it was, **never what you typed**, so the error message cannot be used to bounce hostile text back.

**Real files and numbers.**
- `Equipment` in `packages/schemas/physique_schemas/models.py` has 12 values: barbell, dumbbell, cable, body only, machine, kettlebells, other, bands, e-z curl bar, exercise ball, none, medicine ball. They are exactly the equipment values of the exercise pool the planner uses; `tests/test_p6_evals.py` pins the list to `kb/raw/exercises.json`.
- `app/errors.py::safe_detail` keeps the field path, the error type and our own message; it drops the rejected value, and it removes the name of an unexpected key.
- Safety cases `s21`-`s30` (drugs as equipment, five prompt-injection strings) all expect 422 and no echo.

**What went wrong.** I first put the echo-free handler globally in `main.py`; `/analyze` raises its own 422, so it never ran and a hostile key name leaked through the error path. A test caught it. I also copied one value (`exercise ball`) from a pasted count that had lost a space; a test pinning the list to the file caught that too.

**Quiz.**
1. *Why must the error not repeat the input?* Otherwise the error path reflects hostile text back to whoever (or whatever) reads it.
2. *Why a test that compares the list to the dataset file?* A hand-copied constant can be wrong in one character; the test compares against the source.
3. *Does the checklist stop every attack?* No: it closes one entry point. The writer step has its own checks for text.

---

## Step 3. Picking the priority regions

**Plain picture.** You tell a coach "I want my shoulders bigger"; the coach picks the one to three body parts where you are furthest from your goal, ignores tiny differences, and never treats "my goal is smaller than me" as something to train.

**How it works.** Take the measured regions; drop `lats` (an estimate, never ranked), `upper_back` (never measured) and `waist` (training the abdominals does not shrink the waist); keep those whose gap is at least 0.010 of height; sort by gap, then confidence, then a fixed region order; take at most three.

**Math.** A region's gap is `delta = (goal ratio - current ratio)`, in units of height. A rule like `delta >= 0.010` means "at least 1% of your height". At 175 cm that is about 1.75 cm.

**Real file.** `app/planner/priorities.py` (`MIN_GAP = 0.010`, `MAX_PRIORITIES = 3`). It does not import from tests; `tests/test_planner.py` runs all 15 cases through it.

**Quiz.**
1. *A goal photo is smaller than you in a region; do you train that region harder?* No: a negative gap is never a growth target.
2. *Why exclude the waist?* Abdominal training builds the muscle but does not shrink waist girth, which follows body fat.
3. *Two regions have the same gap; who wins?* The one with higher confidence, then the fixed region order.

---

## Step 4. Candidates: a short list of exercises per muscle

**Plain picture.** Before a coach plans, an assistant fetches, for each muscle, ten exercises that fit your gear and level.

**How it works.** For each of the 15 muscle groups, one search "`<muscle> exercise`" with filters (that muscle, your equipment, your level and below), keeping the best ten. An empty equipment list means "body only", never "no filter". The search is the P5 hybrid retriever (keyword + meaning, combined).

**Real files and numbers.**
- `app/planner/candidates.py` (`TOP_K = 10`), `app/rag/service.py` (loads the retriever once; hybrid when the bge-small files exist, BM25-only with a warning otherwise; `REQUIRE_EMBEDDER=true` makes missing files a startup error).
- The planner never calls the retriever; it receives the lists. Tests use committed lists in `tests/fixtures/planner_ranks_{hybrid,bm25}.json` (4 profiles each, stamped with the corpus fingerprint), so CI needs no model.
- `app/routes/plan.py` filters equipment by the checklist values directly, not through the old synonym map in `app/rag/query.py` (that map turns `none` into `body only` and drops `other`).

**Data source.** The exercises are the free-exercise-db dataset (Unlicense; see P5 docs).

**Quiz.**
1. *Why does the planner receive lists instead of searching itself?* So it is a pure function that tests can run without a model.
2. *What would a "no equipment" user get if empty meant "no filter"?* Every exercise including barbell ones: wrong.
3. *Where is the hybrid ranking tested?* Only by one local smoke test that skips when the model files are absent, plus the committed snapshot; CI proves the planner, not the live retriever.

---

## Step 5. The weekly planner: floors first

**Plain picture.** You have 15 buckets (muscles), each must hold at least 4 "sets" of water a week. Each exercise pours into one main bucket and splashes a little into others. You also have a few days, and each day can only hold so many exercises. You fill buckets using the fewest exercises, never overfilling any bucket past its limit.

**How it works.**
1. **Credit.** An exercise gives its primary muscle 1 credit per set and each secondary muscle 0.5. With 3 sets per exercise, a primary muscle gets 3 and a secondary gets 1.5. (Counting rule from `kb/baselines.json`, source Pelland 2024 preprint; the 0.5 mapping is *our assumption*, stated there.)
2. **Layout.** Each day has "focus" muscles: 1 day = full body; 2 and 3 days = full body each day; 4 days = upper / lower / upper / lower; 5 days = push / pull / lower / upper / lower; 6 days = push / pull / lower twice; 7 days = those six plus a light day (abdominals, calves, forearms, traps).
3. **Pass 1, floors.** Repeat: among muscles still below 4, pick the one with the fewest candidates that can still be placed (ties: the one furthest from its floor, then name). For it, pick the exercise with the most *useful* credit (credit that lands on muscles still below target), then the better search rank, then the exercise id. Put it on the eligible day with the fewest sets for that muscle.
4. **Pass 2, extras.** Priority regions' muscles get +2 weekly sets, only where limits allow.
5. **A day left empty** gets one light filler exercise if any fits.

**Limits (our design, not cited numbers; fixed before any plan was seen).** 3 sets per exercise; at most 9 exercises a day; at most 10 credited sets per muscle per week; at most 8 per muscle per session (also capped by the cited 11); a muscle trains in at most 4 sessions a week.

**The maths of one placement (example).** `Thigh Adductor` is the only adductor exercise for a full-gear beginner. It gives adductors 3.0 and glutes and hamstrings 1.5 each. If glutes is already at 9.0 this week, 9.0 + 1.5 = 10.5 is over the weekly limit of 10, so it cannot be placed. That is exactly what happened.

**Real numbers.**
- First run: **15 of 41 planner tests failed**, all with the same message: `adductors` under its floor. A second, scratch measurement showed that filling the most constrained muscle first fixes `adductors` in the 12 full-gear cases but left `p14` short on `calves`. A dynamic order with ties by name left `p14` short on `triceps` and `quadriceps`. Ties by larger remaining deficit met all 13 cases that must be met.
- A solver (an exact integer program) found that `p14` can meet every floor in 2 days: minimum total shortfall 0.00, over the planner's own 123 candidates and over all 305 plannable beginner exercises for that gear.
- Property test: days 3-7, three levels, three priority sets, hybrid and BM25 fixtures (90 combinations) all met their floors.

**What went wrong.** The order was chosen after seeing failures twice. Passing is evidence for these profiles, not a guarantee. Greedy means "take the best-looking choice now", which can miss a plan that exists.

**Quiz.**
1. *Why is the 6-set minimum a consequence of using 3 sets per exercise?* A muscle needs 4; one exercise gives 3 (not enough), two give 6, so without splash credit from other exercises a muscle ends with at least 6.
2. *The solver says a plan exists but the planner fails; which do you trust?* Both: the solver proves a plan exists; the planner failing means the greedy missed it. The checker makes the miss visible.
3. *Why caps?* Floors guard against too little; caps guard against too much and keep sessions realistic.

---

## Step 6. An independent checker

**Plain picture.** After the cook plates the meal, a second person who did not cook it weighs every portion.

**How it works.** `check()` takes only the final plan and the exercise data, reads the floors itself from `kb/baselines.json`, recomputes every muscle's weekly credit with its own counting code, and raises an error if a muscle is under its floor and not *named* as a shortfall, if a named number disagrees, or if any cap is broken. The planner calls it before returning; the writer's plan must pass it too.

**Real files.** `app/planner/check.py` (`PlannerInvariantError`, `weekly_credit`). Tests: a silent shortfall and a cap breach each raise; an empty equipment pool names all 15 muscles as shortfalls (planned 0) and does not raise.

**Quiz.**
1. *Why not trust the planner's own totals?* A bug in the totals would hide a bug in the plan.
2. *Does a named shortfall count as a failure?* No: naming is the honest outcome when a floor cannot be met (e.g. body weight only has no adductor exercise); a silent one is the failure.
3. *What happens on a breach?* The route returns a generic 500 and logs the detail; nothing partial is sent.

---

## Step 7. Body numbers and the safety re-check

**Plain picture.** A pharmacist double-checks a dose that the doctor already calculated, using their own table.

**How it works.** `/plan` recomputes body composition from your profile (it never trusts a client-supplied one). If there is no waist at all (no tape value, no scan number), it returns the training plan with a fixed note that calorie targets need a waist measurement; for someone with an eating-disorder history it returns the supportive note and no numbers instead. `app/safety/rules.py` then re-checks the result *without importing the code that computed it*.

**The rules (from `docs/privacy-and-safety.md`, restated and pinned by tests).** Deficit at most 20% of daily needs and never below the resting rate; surplus at most 10%; loss at most 1% of body weight a week (using 7700 kcal per kg, a conversion that exists only in code); no cut or recomp target if BMI is under 18.5 or body fat is under 8% (men) / 18% (women, our rule); no calorie numbers for an eating-disorder history, under 18, or an unset sex. Calories are whole numbers, so comparisons allow ±0.5 kcal.

**Real numbers.** Safety suite offline: first 34 pass / 4 need scoring / 2 fail; the 2 failures (`s04`, `s12`) were my percentage tolerance against whole-kcal rounding (2106 against 80% of 2633 = 2106.4, 0.4 kcal under); with the 0.5 kcal tolerance: **36 pass, 4 need scoring, 0 fail**.

**Quiz.**
1. *Why re-check with a separate module?* If both used the same code, one bug would pass both.
2. *Why a test pinning the restated constants to the original?* So the two copies cannot drift apart.
3. *Why 0.5 kcal?* Calories are rounded to whole numbers; a rounding of up to 0.5 can look like a tiny breach.

---

## Step 8. The LLM writer

**Plain picture.** The coach has a draft plan. A second person polishes the wording and may swap an exercise for an equivalent one from a short list, but cannot change how many sets anything gets. The coach then reads the result and rejects it wholesale if anything is off.

**How it works.** One request to a language model (Groq, `openai/gpt-oss-120b`) asking for JSON in a fixed shape: a summary, and for each day a label and a list of `{exercise_id, reps, cue}`. There is **no sets field**. The prompt (the "facts block") contains: the priorities and gaps as percentages, a catalogue of exercises (each once: id, name, main muscle, a short instruction text), 3 options per muscle, each day's focus and baseline plan, and any body numbers. It contains no user free text and never the client's `reason` strings.

**Accepted only if all hold:**
1. It parses against the schema.
2. Day count matches; each id is in the candidate list of a muscle in that day's focus; no repeat within a day.
3. The independent `check()` passes on the model's plan, and it has no more short muscles than the plain plan.
4. Text checks pass (Step 9).

**Call budget and fallbacks.** At most 3 calls and 1 retry per report: the main provider, one retry for a transient failure (timeout or rate limit), then the backup provider. Any failure serves the plain plan with a reason from a fixed list of six words: `disabled`, `timeout`, `rate_limited`, `invalid_json`, `check_failed`, `unsafe_text`. The result says which path wrote it: `primary`, `fallback` or `deterministic`. The writer is off unless `LLM_WRITER_ENABLED=true`.

**Real numbers.**
- Provider limit found by a probe: 8,000 tokens per minute on the free tier.
- First prompt: 94,225 characters (Groq reported ~24,400 tokens requested) → HTTP 413 → **0 of 15** plans served by the model. After the fix: 13,150 characters; probe returned valid output.
- Full run after the prompt fix: **13 served by the model, 2 plain** (`p03` `check_failed`; `p11` `invalid_json`).
- Reading those 13 plans showed the cues were truncated database headers ("Thigh Abductor. Targets abductors; also glutes. Equipment: machine. Level: beginner. isolation push"). I had given the model the first 100 characters of each exercise's text, which is the header. Fixed with a function that takes the instructions part, plus a new rejection rule for cues containing "Equipment:" or "Level:".
- Subset after that fix (4 cases, owner-run): `p01` and `p14` served by the model, `p03` `check_failed` again (reproducible, cause not diagnosed), `p11` `rate_limited`. Cues are real coaching sentences.
- Not measured: how often the backup provider works (it served 0 plans); latency; cost; the licence terms of the model (not read here).

**Quiz.**
1. *Why no sets field?* Then the model has no way to change volume; the planner's numbers always stand.
2. *Why accept or reject the whole answer?* Fixing part of it could hide a wrong plan behind a correct-looking one.
3. *Why did every automated check pass on the truncated cues?* They checked structure and safety, not whether a sentence reads like coaching. Only reading samples found it.

---

## Step 9. Text safety checks

**Plain picture.** A proofreader with a banned-words list that also looks for web links and web code.

**How it works.** Every string the model writes must avoid: the banned shaming words (`flaw`, `lacking`, `weak`, `ugly`, `skinny`, `scrawny`, `pathetic`, `bulky`, a bare `fat` that is not "body fat" or "fat loss", `small chest/arms/legs`); sentences that say one region is more or less defined than another; drug and supplement words (steroids, creatine, caffeine, supplements and so on); any URL, markdown link or HTML tag. Every number must come from the facts block (compared as numbers, so "3%" matches 3.0). The disclaimer is written by code, not by the model.

**Real files.** `app/agent/textsafety.py` repeats the frozen patterns (the image does not carry `evals/`); `tests/test_agent_textsafety.py` pins every regex to `evals/patterns.py` and runs the frozen "must flag / must not flag" sentences against the copy.

**What went wrong.** The first regional-claim pattern flagged harmless sentences ("train chest during your cut", "add lean mass", "come back leaner"); I narrowed it to comparatives and "is/are defined" before freezing. A word list is also coarse: "supplemental sets" must not trigger "supplements", so patterns use word boundaries.

**Quiz.**
1. *Why check numbers in the prose?* The model could invent a figure; every allowed figure is in the facts.
2. *Why is the disclaimer not left to the model?* It is a safety text; fixed text cannot be misstated.
3. *Can a word list prove text is kind?* No; it blocks known bad patterns. Tone is for a person to judge.

---

## Step 10. The realism note and fixed warnings

**Plain picture.** If your goal photo is far beyond what most people reach naturally, the app says so once, kindly, and offers a nearer first target.

**How it works.** If the largest priority gap is at least 0.05 (5% of height), the report carries one note: goal photos are often posed, lit, pumped or edited; a sensible first target is half the largest gap, written as a percentage of height and as centimetres at the user's height. The note is built by code. Separately, `warnings` carry fixed sentences: a different-sex goal ("the goal physique is the other sex from yours…") and which regions were "not measured in this scan", named from the region list, never from text the client sent.

**Real numbers.** Checked before coding against the frozen cases: `s31` (gap 0.10), `s32` (0.08), `s33` (0.09) and `p15` (0.08) expect a note; `s34` (0.03) and `s35` (no positive gap) do not; all agree with 0.05. Example: gap 0.10 → first target 5.0% of height ≈ 9 cm at 175 cm. The threshold 0.05 and the "half" are our rule, untuned.

**A leak we closed.** The report would have echoed the client's own `reason` and V-taper text back inside the gap. `clean_gap` replaces unmeasured reasons with fixed text and drops the V-taper; a route test scans the whole response for a planted string.

**Quiz.**
1. *Why must a negative gap never trigger the note?* The note is about goals beyond reach; a goal smaller than you is not.
2. *Why name unmeasured regions from an enum?* Client text must never come back.
3. *Is 0.05 validated?* No; it sits between the frozen 0.03 and 0.06 cases and is labelled untuned.

---

## Step 11. Gate suites and what a live run can and cannot prove

**Plain picture.** A dress rehearsal: run all 15 plan cases and all 40 safety cases through the real endpoint, and report counts.

**How it works.** `python -m evals.run --suite plan|safety --llm off|live` (`--cases` for a subset, `--readable` to merge saved scores without calling the model). The plan suite scores four rubric items by code: follows the priorities (every priority muscle at least as covered as in the plain plan), spreads the volume (the checker passes), safe wording (all text checks pass), cites its sources (every exercise is a retrieved candidate and cites itself). The fifth item, readable, is human. Live runs pause between requests because of the token limit.

**Real numbers and limits.** See Step 8 for the live results. Offline plan suite: 15 of 15 schema-valid, priorities, days, invariant and all four code items. Readability: Claude rated 2 plans, both "readable" (n = 2; not the owner's blind scoring). The 4 `supportive_note` safety cases are unscored. No live pass rate is claimed beyond the numbers above.

**Quiz.**
1. *Why is "all code checks pass 15 of 15" weak evidence for the live writer?* For plans served by the plain path, the checks compare the plan with itself.
2. *Why a human item?* Clarity of wording cannot be settled by regex.
3. *What does n = 2 allow you to say?* Only that those two plans read fine; nothing about a rate.

---

## Glossary

- **ADR:** a short written record of a decision.
- **Fingerprint (SHA-256):** a string computed from a file; any change alters it.
- **Closed enum / Literal:** a fixed list of allowed values.
- **422:** the HTTP code for "your request is invalid".
- **Floor:** minimum weekly sets per muscle group (4).
- **Cap:** a maximum (10 a week, 8 a session, 4 sessions a week, 9 exercises a day).
- **Primary / secondary credit:** 1.0 and 0.5 per set.
- **Greedy:** pick the best-looking move each step; fast, but can miss the best overall plan.
- **Shortfall:** a muscle below its floor, named.
- **Structured output:** asking the model for JSON in a fixed shape.
- **Token:** a chunk of text the model reads (about 3-4 characters).
- **Rate limit (TPM):** tokens per minute allowed by the provider.
- **Fallback reason:** one of six fixed words for why the plain plan was served.
- **Prompt injection:** hostile text trying to give the model new orders.
- **Blind scoring:** scoring without knowing which case a plan is.

---

## Ten viva questions

1. Why is the planner not an LLM, and what does the LLM still do?
2. Walk through how a muscle gets from 0 to its floor; what is credit?
3. Why did the planner fail 15 tests the first time, and what was the fix?
4. Why can a solver prove a plan exists while the planner fails?
5. What does `check()` verify, and why is it independent?
6. List the acceptance checks on the model's output. What happens on any failure?
7. What two bugs made the first live run serve 0 of 15, and how were they found?
8. Why are the evals frozen, and what is an amendment? Give the one we made.
9. What are the safety limits, where do they come from, and which one is only our rule?
10. What is *not* proven by P6? (Name at least four.)

*Answers sketch: 1) hard rules need a guarantee; LLM does wording/variety only. 2) credit 1 / 0.5 per set; greedy fill; floor 4. 3) one exercise would push glutes over 10; most-constrained-first with deficit tie-break. 4) existence versus the greedy's search. 5) recomputes from the final plan, own counting, raises on silent shortfall/cap breach. 6) parse, ids/focus, check with no worse shortfalls, text and number checks; whole answer discarded, plain plan with a fixed reason. 7) 94k-character prompt over the 8,000 tokens/minute limit, 413 mislabelled; cues copied from metadata header; found by probe and by reading output. 8) so the exam cannot be tuned; a written change with old/new fingerprints; `patterns.py` after a formatter. 9) privacy-and-safety doc; 18% female floor is ours; 7700 kcal/kg exists only in code. 10) owner's blind readability; 4 supportive-note scores; why `p03` falls back; live latency/cost; the backup provider; the full post-fix 15-case run.*
