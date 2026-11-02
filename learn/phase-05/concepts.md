# P5 concepts, from zero

This file explains every idea P5 uses, in the order you meet them. You do not need prior knowledge. Each idea has the same structure: a plain-English picture, then how it works, then the real numbers from our project, the file where it lives, and what goes wrong.

Where the numbers come from: `kb/manifest.json` (the index totals), `kb/chunks.jsonl` (the chunks themselves), `kb/ingest_exercises.py` and `kb/build_index.py` (how chunks are made), `evals/results/rag_run2.json` (the test results), `docs/logs/dev-log.md` (the P5 log), and `docs/decisions.md` ADR-048 (the decisions).

---

## Part A: What P5 is for

Our app will eventually write a workout plan for a user. Before it writes exercises, it needs a library to pick from. P5 builds that library and a way to search it.

Think of a library with 893 index cards. Each card has one exercise, or one short section of a guide. When the planner needs "a cable exercise for the back, for a beginner", it searches the cards and gets back the best few. P5 builds the cards, the search, and a test that checks the search is good.

Two other things live in P5: a list of cited weekly volumes (how many sets per muscle per week the research supports) and nutrition ranges. The planner must never go below the volume floor, so these numbers are stored as data, not typed into code.

---

## Step 1: What a chunk is (read this first)

**The plain picture.** Imagine copying a textbook onto index cards. One card holds one idea. If a card holds a whole chapter, a search for "rest between sets" will find the card, but you will have to read the whole chapter to find the answer. If a card holds half a sentence, it means nothing alone. A good card holds one complete idea that can be understood without the cards around it.

That card is a **chunk**: a piece of text we store, search and return as one unit.

**Why it has to be a good size.** Our search tool, the embedding model (Step 9), turns each chunk into a list of 384 numbers. That list is one summary of the whole chunk. A summary of a long, mixed chunk is blurry. A summary of a short, focused chunk is sharp. The model also reads at most 512 pieces of text per chunk (called tokens; a token is roughly a word or part of a word). Anything beyond 512 is cut off without warning.

**How big our chunks are.** There are two kinds, and they are made differently:

| Kind | Made from | How many | Chunk id looks like |
|---|---|---|---|
| Exercise | one exercise in the free exercise library | 876 | `ex:Barbell_Curl` |
| Guideline | one heading section of a document we wrote | 17 | `doc:rest-progression#rest-between-sets` |
| **Total** | | **893** | `kb/manifest.json`: `"chunks": 893` |

**Exercise chunks.** One exercise = one chunk. The text is built by one function, `chunk()` in `kb/ingest_exercises.py` (lines 21-43). It joins the exercise's name, main muscles, extra muscles, equipment, level, type and its step-by-step instructions into one paragraph. Here is a real one, `ex:3_4_Sit-Up`, shortened:

> *3/4 Sit-Up. Targets abdominals. Equipment: body only. Level: beginner. compound pull strength. Lie down on the floor and secure your feet. ... Repeat for the recommended amount of repetitions.*

Why one exercise per chunk? An exercise is already a complete idea: what it is, what it works, what you need, how to do it. If we split it into one chunk per instruction step, "Reverse the motion, going only 3/4 of the way down" would be found on its own, but nobody would know which exercise it belongs to. The metadata (muscle, equipment, level) that filters use would also have to be copied into every piece.

**Guideline chunks.** Our own documents (in `kb/docs/`) are split on their headings. A heading starting with `#` or `##` begins a new chunk, and the body under it becomes that chunk's text. The split is done by `doc_chunks()` in `kb/build_index.py`. The heading is part of the text, so the chunk reads "Rest between sets. Rest ...". Smaller headings (`###`) do not start a new chunk; they stay inside their parent.

Example: `kb/docs/rep-ranges-failure.md` has the headings "Rep ranges" and "How close to failure" under the title. Each becomes one chunk, `doc:rep-ranges-failure#rep-ranges` and `doc:rep-ranges-failure#how-close-to-failure`. The `#...` part is the heading turned into a web-style slug (lowercase, spaces become `-`, punctuation removed). The rule is written in `slug()` in `kb/build_index.py`.

**Why headings and not a fixed length?** The author already marked where one topic ends and the next begins. Cutting every 100 words would sometimes split a sentence, and sometimes put two topics on one card. The eval questions (Step 8) ask about single topics, so heading chunks match them.

**Tokens: measured (user-run, 2026-11-02).** Our plan had a draft figure of about 150-250 tokens per chunk. Nobody had counted until now. The count:

- 893 chunks in total
- shortest: 25 tokens
- median: 153 tokens (the middle chunk; half are shorter, half longer)
- longest: 707 tokens
- **2 chunks are longer than 512 tokens.** The model cuts those, so the end of their text is never embedded:
  - `ex:Power_Clean`: 707 tokens, about 195 cut off
  - `ex:Clean_and_Jerk`: 565 tokens, about 53 cut off

  Their dataset categories (user-run check): `Clean_and_Jerk` is **olympic weightlifting**, `Power_Clean` is **strength**. Only strength and powerlifting count as `plannable` (`kb/ingest_exercises.py`), so the retriever's default pool leaves out `Clean_and_Jerk` but **can return `Power_Clean`**. For `Power_Clean` the exercise name, muscles, equipment and level are at the start of the text, inside the first 512 tokens, so the meaning vector still knows what it is; what is lost is the end of the instructions. That is reasoning from the chunk layout, not a measurement. The planner that would use the pool is P6 and does not exist yet, so we make no claim about it.

  **Decision (user, 2026-11-02): leave the chunking as it is.** The truncation is logged as a known limit (ADR-048 Amendment 4), `build_index.py` now prints the list, and a test pins it to exactly these two ids.

The median sits inside the plan's 150-250 range, so the draft guess was close for the typical chunk. The two long chunks are the exception.

To list the two long chunks (run from the repo root; it only reads files and prints one line per chunk):

```
uv run python -c "import json; from tokenizers import Tokenizer; t=Tokenizer.from_file('ml/data/models/bge-small/tokenizer.json'); [print(json.loads(l)['id'], len(t.encode(json.loads(l)['text']).ids)) for l in open('kb/chunks.jsonl') if len(t.encode(json.loads(l)['text']).ids)>512]"
```

Each line it prints is a chunk id and its token count. The token count includes two marker tokens the model adds, so it runs about two higher than the text alone.

**Common mistakes.**
- Thinking a bigger chunk is always better because it has more context. It also has more topics, so its one summary vector is blurrier.
- Thinking a smaller chunk is always better. A sentence alone ("Repeat for the recommended amount") loses its meaning.
- Changing chunk boundaries after the eval is scored. That changes the test. Our rule: chunking may be fixed only before the first scoring run, or logged as a change.

**Quiz.**
1. Why is a whole exercise one chunk, not one chunk per instruction? *The exercise is one complete idea, and its name, muscles and equipment would be lost in a single step.*
2. Which of our guideline headings starts a new chunk, `#`, `##` or `###`? *`#` and `##`. `###` stays inside its parent.*
3. What is the median chunk length in tokens, and what is the problem? *153 tokens, measured. Two chunks are over the 512 limit and lose their last part.*

---

## Step 2: Where the exercises come from (the exercise library)

**Plain picture.** We did not write 876 exercises ourselves. A community made a free library of exercises as one JSON file. We download it once, read it, and turn each entry into a chunk. Think of copying a recipe book that anyone may copy.

**How it works.** `kb/ingest_exercises.py` reads `kb/raw/exercises.json` and calls `chunk()` on each entry. Two small rules are applied on the way:
- The dataset says `expert`; our app says `advanced`. The mapping is one line, `LEVEL = {...}`.
- A `plannable` flag is set when the exercise's category is `strength` or `powerlifting`. Those two categories make 622 of the 876 exercises. The rest (stretching, cardio, plyometrics, olympic lifts, strongman) stay in the index but are only returned when someone asks for them by name. This is our rule for the default pool; the planner wants hypertrophy work.

**Data source and licence.** `yuhonas/free-exercise-db` on GitHub, file `dist/exercises.json`, 1,005,327 bytes, SHA-256 `5bb747e3...d40bf`. Licence: Unlicense, which means public domain. We read the licence file in full (`kb/raw/LICENSE.md`).

**What we did not check.** The library is written by volunteers and is not hand-filtered. Some exercises have wrong or vague muscle labels. A wrong label becomes a wrong search result.

**Numbers.** 876 exercise chunks; 622 `plannable`; 254 not plannable. Equipment: 77 exercises have no equipment (`none`). Level: 525 beginner, 294 intermediate, 57 expert.

**Common mistakes.** Forgetting `secondaryMuscles`. A squat works the quads mainly and the glutes too; the glutes are secondary, and our counting rule (Step 4) gives them half credit.

**Quiz.**
1. Why are 254 exercises indexed but not plannable? *They are useful for explicit searches, but the default planner pool is strength and powerlifting.*
2. What licence allows us to use the library? *Unlicense (public domain).*
3. What happens to the image paths in the dataset? *They are dropped; the index is text only.*

---

## Step 3: Our own guideline documents

**Plain picture.** The library has exercises but no advice. We wrote short documents with advice (how many sets, how close to failure, how long to rest, how much protein) in our own words, based on the research we read. These become the 17 guideline chunks.

**Files.** Seven documents in `kb/docs/`: `volume-frequency`, `rep-ranges-failure`, `rest-progression`, `muscle-group-guide`, `split-templates`, `protein-energy`, `see-a-professional`. Each starts with a small header (front matter) that lists its topic, level, goal and the sources it is based on. A test checks that every source id listed exists in `kb/sources.md`.

**How they become chunks.** `doc_chunks()` in `kb/build_index.py` reads the header, then splits the body on headings. Here is a real guideline chunk, `doc:protein-energy#protein`, shortened: "Protein. A typical protein range for people who lift weights is 1.6–2.2 grams per kilogram of total body weight per day. ..."

Notice the chunk repeats its heading at the start ("Protein."). That helps the embedding model know what the chunk is about.

**Why they are not in the exercise filters.** Guideline chunks have no equipment and no real muscle, so when a filter is set (for example "chest, dumbbell, beginner") they are removed. They appear only in unfiltered searches.

**Numbers.** 17 guideline chunks in total (`kb/manifest.json`). The per-document counts were not listed in the log, so we do not quote them.

**Common mistakes.** Writing advice that is not in the cited source. The docs say "our own words", and a number that cannot be checked is left out (ADR-048 §1).

**Quiz.**
1. Why does the heading repeat at the start of a chunk? *So the embedding knows the topic.*
2. Why do guideline chunks disappear under a filter? *They have no equipment or real muscle, and filtered searches look only at exercises.*
3. Where is the list of allowed sources kept? *`kb/sources.md`, one row per source.*

---

## Step 4: The cited volume floor (how many sets per muscle per week)

**Plain picture.** Imagine a checklist: every muscle the app tracks must get at least some training each week. The checklist number is 4 sets. Research (not us) says that is roughly the lowest weekly amount that showed measurable muscle growth on average.

**How sets are counted.** This is called fractional counting, and it is our rule, not a fact of nature:
- A set counts **1** for each main (primary) muscle of the exercise.
- A set counts **0.5** for each helper (secondary) muscle.

Example: 3 sets of an exercise whose primary muscle is chest and whose secondary muscle is triceps. Chest gets 3 sets of credit, triceps gets 1.5.

The 1 and 0.5 come from one research paper's method, but the library's primary and secondary lists were not written with that method in mind. The file says so: `counting.assumption` in `kb/baselines.json`.

**The numbers.** 15 muscle groups, each with `min_sets_per_week: 4`. The per-session guide is 11 sets for one muscle in one workout (from a 2025 preprint, and it is a soft guide, not a rule).

**Data source.** Pelland et al. 2024 (a preprint, not yet peer-reviewed; version 2, `kb/sources.md`). The figure is in its Table 2A, "Minimum Effective Dose", at page 24 of the PDF. We read pages 1-3 and 13-24.

**What the number does not say.** It is an average over 67 studies, mostly young adults, over about 10 weeks. It is not a promise for you, and the app says "typical minimum" rather than "your minimum".

**Common mistakes.** Using the 4 as a maximum. It is a floor; more is allowed.

**Quiz.**
1. What does a secondary muscle count for? *0.5 of a set.*
2. Why "typical minimum" and not "your minimum"? *It is a population average from research, not a personal prescription.*
3. Is 11 a safety limit? *No. It is a soft planning guide from a cautious preprint.*

---

## Step 5: Mapping our body regions to muscle groups

**Plain picture.** The body scan talks in regions ("upper arm", "thigh"). The exercise library talks in muscle groups ("biceps", "triceps", "quadriceps", "hamstrings", "adductors"). A translation table connects them. `kb/region_map.json` is that table.

**How it works.** A region can map to several groups. "Thigh" maps to quadriceps, hamstrings and adductors. The floor of 4 applies to each group separately. If it applied to the thigh as a whole, 12 sets of quad work could hide zero hamstring work.

**Checks.** `validate()` in `kb/baselines.py` refuses to continue if a region has no group, a group has no baseline, or a source id is missing. It fails loudly; it never fills in a default.

**Two notes that matter for safety.**
- Abs training does not shrink the waist. The map says so, and the report must not claim it does.
- Upper back and back width (lats) are never measured for size from a front photo, but they still get their weekly minimum. Not measuring size is not a reason to skip training.

**Numbers.** 10 regions; 15 muscle groups; 2 groups deliberately untracked (lower back, neck).

**Quiz.**
1. Why is the floor per group and not per region? *So one group cannot hide another.*
2. What happens if a group has no baseline? *Validation raises an error. There is no default.*
3. Does training the abs make the waist smaller? *No; the map and the report say so.*

---

## Step 6: The example week and the feasibility test

**Plain picture.** Before we promise that a plan reaches every floor, we check it on an example. We wrote one example week for a full-body gym routine (3 days). A test adds up the sets for each muscle and checks the floor is met.

**The test.** `check()` in `kb/templates.py` does two things:
1. If one muscle gets more than 11 sets in a single day, it reports it (soft guide).
2. If one muscle gets less than 4 sets in the whole week, it reports it.

If anything is wrong, it raises an error listing every problem. A problem means the template must change. It never means lowering the floor.

**Numbers.** 3 days, roughly 7-9 exercises per day, 2-3 sets per exercise (our own design, not cited). The test `tests/test_kb_templates.py` passed 4 tests (user-run). The per-muscle weekly totals were not written into the log, so we do not quote them.

**Quiz.**
1. What are the two checks? *Weekly floor for each group; daily cap of 11 for each group.*
2. If a check fails, what changes? *The template, never the floor.*
3. Who chose the 2-3 sets per exercise? *We did. They are not cited numbers.*

---

## Step 7: Protein and energy ranges

**Plain picture.** People ask "how much protein?" The honest answer is a range, because the studies disagree about the exact point. The app shows the range, never one number.

**The protein range.** 1.6 to 2.2 grams per kilogram of body weight per day. For a 70 kg person that is 112 to 154 grams. Where the range comes from: a 2018 meta-analysis (Morton et al.) found gains stopped rising at about 1.62 g/kg/day, but the confidence band was 1.03 to 2.20 and the result was not statistically significant (p = 0.079). The authors suggest about 2.2 as a practical number. Because the band is so wide, a single number would look more precise than the evidence is.

**The calorie bands.** The app has its own calorie bands for cutting, maintaining, recomposing and bulking. When the API starts, `check_energy` in `services/api/app/nutrition.py` compares them with the cited ranges. If a band is outside the citation and has no declared reason, the API refuses to start. The bulk band (5-10% above maintenance) is stricter than the cited 10-20%, so it is declared as stricter and the app's limit wins.

**Limits.** The cited energy papers are written for natural bodybuilders. Using them for everyday lifters is our extrapolation, and the file says so.

**Quiz.**
1. Why a range and not 1.8? *The evidence band is wide; one number would invent precision.*
2. What happens if a calorie band is outside the cited range with no declared reason? *The API will not start.*
3. Why is 2.3 to 3.1 g/kg not used? *It is for lean people in a deficit, and the papers use the unit differently.*

---

## Step 8: The eval set, written before any search code

**Plain picture.** Before building a search, we wrote a test: 33 questions, each with the right answers marked by hand. This is an exam for the search, and we wrote it before the search existed so that we could not tune the answers to pass.

**The questions.** 17 "planner-style" ("dumbbell press for chest"), 3 "synonym" ("skull crusher", where the library calls it EZ-Bar Skullcrusher), 4 "exact name" ("Romanian Deadlift"), and 9 "guideline" questions ("how many sets per week does a muscle need to grow"). The file is `evals/rag.jsonl`. Its fingerprint (SHA-256) was logged before any search ran: `ac22cc70...a450`.

**The score, in plain words.** For each question we look at the top 5 results. `recall@5` asks: of the right answers, how many did we get in the top 5? If a question has 12 right answers, the top 5 can hold at most 5, so we divide by the smaller number, `min(12, 5) = 5`. `hit@5` asks: did we find at least one right answer?

**Why this matters.** A test written after the search is tuned to it. A test written first, with its fingerprint logged, shows that the answers were not changed to fit the result. Gold answers were not edited after the first run.

**Quiz.**
1. Why write the test before the search? *So the answers cannot be tuned to the search.*
2. Why divide by `min(gold, 5)`? *The top 5 can never hold more than 5 answers.*
3. What is a fingerprint for? *It proves the test file did not change after it was logged.*

---

## Step 9: Turning text into numbers (the embedding model)

**Plain picture.** A computer cannot compare meanings directly. An embedding model reads a text and writes down a list of 384 numbers that acts like a fingerprint of its meaning. Two texts about the same idea get fingerprints that point the same way. We use bge-small, a small model, in a compressed form (int8, which stores each number in one byte instead of four).

**How a text becomes numbers.**
1. The text is split into word pieces ("tokens") and each piece gets an id.
2. The model reads all the pieces and produces one vector per piece.
3. We keep only the first piece's vector (CLS pooling). The model is trained so that this vector summarises the whole text. The setting is in BAAI's `1_Pooling/config.json`.
4. We scale the vector to length 1 (L2 normalisation), so two vectors can be compared by their angle.

Queries get a short fixed prefix ("Represent this sentence for searching relevant passages: ") before embedding. Chunks never get it. This is what the model's authors ask for.

**Why one text at a time.** We found that the compressed model gives slightly different numbers (about 0.001 per number) depending on how many texts are processed together. So index and queries are both embedded one at a time, to take the same path.

**Why the same model for chunks and queries.** Numbers from two different models cannot be compared. The index stores a fingerprint of the model file; if the file changes, the index refuses to load (`corpus_hash` in `kb/manifest.json`).

**Numbers.** 893 vectors, each of 384 numbers, stored in `kb/embeddings.npy` (1,371,648 bytes, about 1.4 MB). Model file 34,014,426 bytes. Its SHA-256 is in `kb/sources.md`. Licence MIT.

**Quiz.**
1. What does a 384-number vector stand for? *A summary of the meaning of a text.*
2. Why do chunks and queries use the same model? *Vectors from different models cannot be compared.*
3. Why do queries get a prefix and chunks do not? *The model was trained that way; the prefix tells it "this is a search".*

---

## Step 10: Keyword search (BM25)

**Plain picture.** Ctrl-F with good manners. It counts how many query words appear in each chunk. A word that appears in only a few chunks counts for more than a word that appears everywhere ("the" is worth nothing; "Romanian" is worth a lot). A longer chunk is not rewarded just for being long.

**How the score works.** For each query word `t` in a chunk `d`:
- `idf(t)`: how rare the word is across all 893 chunks. Rare means high.
- how often the word appears in the chunk, with diminishing returns (the 10th mention adds less than the 1st).
- a length adjustment, so long chunks do not win by size.

We use the standard settings of the library (`k1 = 1.5`, `b = 0.75`); we did not tune them.

**What it is good at.** Exact names. "Romanian Deadlift" matches the chunk with exactly those words.

**What it is bad at.** Meaning. "skull crusher" does not match "Skullcrusher", because the two words are different strings. This is the main reason it scored lowest on synonym questions (0.667).

**Numbers.** Keyword search alone, recall@5 0.833 (after the fix, run 2).

**Quiz.**
1. Why is a rare word worth more? *It separates chunks; a common word appears everywhere and tells you nothing.*
2. Why does BM25 miss "skull crusher"? *It matches exact words, and the library spells it differently.*
3. Why only candidates with score above zero? *A chunk with none of the query words has no reason to be a result.*

---

## Step 11: Combining the two searches (hybrid, RRF)

**Plain picture.** Two judges rank the same 893 cards. One judges by keywords, one by meaning. Their scores are on different scales, so we do not add them. We combine their rankings: a card gets points for its rank in each list. Being near the top in both lists is best.

**The formula.** For each card, add `1 / (60 + rank)` for each list it appears in, with rank 1 being the top. The number 60 is standard in the literature (Cormack et al., 2009) and was fixed before our first run.

Example: a card ranked 1st by keywords and 3rd by meaning gets `1/61 + 1/63 = 0.0323`. A card ranked 2nd by keywords only gets `1/62 = 0.0161`. The first card wins because two judges agree.

**Why we keep only the top 20 from each.** Enough to catch a card one judge ranks mid-table, not so many that low ranks matter. Not tuned.

**Numbers.** Hybrid recall@5 **0.903** (run 2), against 0.833 for keywords and 0.841 for meaning alone. Every query had at least one right answer in the top 5 (hit@5 = 1.000).

**Honest caveat.** The gap between hybrid and meaning-only is +0.062 with a confidence interval of -0.017 to +0.159. That interval includes zero, so with 33 questions we cannot be sure hybrid beats meaning-only. The gap against keywords-only is clearer: +0.070 with an interval from +0.011 to +0.147.

**Quiz.**
1. Why use ranks instead of scores? *The two scores are on different scales.*
2. What does a card in both lists get? *Two points, one from each rank.*
3. Why not tune the 60? *33 questions are too few to tune anything without fooling ourselves.*

---

## Step 12: Filters and the query manager

**Plain picture.** Before the judges look, we throw out cards that cannot be an answer. A request for "cable, beginner, back" should never return a dumbbell exercise or an advanced exercise.

**The filters.** Muscle, equipment and level. A card survives only if it is an exercise, matches the muscle, matches the equipment and matches the level. The level rule is ours: a beginner sees beginner exercises only, and an intermediate sees beginner and intermediate.

**Why the planner pool.** Early on, a filtered search returned a stationary bike and a clean-shrug variant as "back" exercises. The fix: filtered searches only look at the strength and powerlifting exercises (the 622 plannable ones) unless asked otherwise. We found this in run 1 and fixed it. Run 2 is reported next to run 1, not in place of it.

**The query manager.** It takes a body region, the user's equipment and level, and splits the request into one search per muscle group. "Back" becomes "middle back", "traps", "lats". Equipment words are looked up in a fixed list; unknown words are thrown away and never sent to the search, for privacy.

**Numbers.** Run 1 recall@5 0.891 (hybrid), run 2 0.903 after the pool fix.

**Quiz.**
1. Why do we throw out cards before ranking? *They cannot be answers, so ranking them wastes effort and can push out real answers.*
2. Why are unknown equipment words dropped? *Users' words are not sent to the search or logged.*
3. What was the pool bug? *Filtered searches included cardio and olympic lifts.*

---

## Step 13: Measuring the result

**Plain picture.** We ran the 33 questions three ways (keywords only, meaning only, both) and counted the right answers in the top 5. Then we asked: is the difference big enough to be trusted, or could it be luck?

**The confidence interval, plainly.** We take the 33 question scores for hybrid, subtract the 33 scores for the other method, and resample those differences 10,000 times, each time picking 33 with replacement. The middle 95% of the resampled averages is the interval. If it includes zero, the difference could be luck.

**Results (run 2).**

| | Keywords | Meaning | Both (hybrid) |
|---|---|---|---|
| recall@5 | 0.833 | 0.841 | **0.903** |
| hit@5 | 0.970 | 0.939 | **1.000** |

By question type, hybrid: exact names 1.00, guidelines 1.00, planner-style 0.847, synonyms 0.800.

**The gate we set beforehand.** Hybrid must reach 0.85 and beat each single method. It did, in both runs. We also said beforehand that a result below the gate would be recorded as a negative result, not hidden.

**Remaining misses.** Some "wrong" answers are reasonable answers that the test did not list as right (for example a cable crunch variant missing from the gold list). Gold was not edited after run 1, so these stand.

**Quiz.**
1. What does "the interval includes zero" mean? *The difference might be luck; we cannot tell.*
2. Why keep run 1 next to run 2? *So nobody can hide the bug we fixed.*
3. Why do the small question-type groups get no interval? *Three or four questions cannot give a meaningful one.*

---

## Step 14: What we did not build (on purpose)

- **Semantic cache** (remember past answers to speed up repeat questions). Deferred. Its false-hit test is not passed and is never reported as passed.
- **Intent router** (work out whether a question is a plan, a lookup or off-topic). Not needed while no free text reaches the search.
- **The full corpus** (25-40 documents) and the 40-question eval. We have 7 documents and 33 questions for the demo.
- **Peer-reviewed checks** of the Pelland numbers, and reading the unread parts of several papers. Listed in `docs/phase-plan.md` P5.

Each item is listed as TBD in `docs/phase-plan.md`, so none is forgotten.

---

# Glossary

- **Chunk**: one piece of text we store and search as a unit. Here, one exercise or one document section.
- **Token**: a small piece of text the model reads (roughly a word or part of one).
- **Embedding**: a list of numbers (here 384) that summarises the meaning of a text.
- **Vector**: a list of numbers; an embedding is one.
- **Cosine similarity**: how closely two vectors point the same way (1 = same, 0 = unrelated).
- **Index**: the stored chunks, their vectors and a fingerprint, ready to search.
- **Corpus**: the full set of chunks.
- **Keyword search (BM25)**: ranks chunks by shared, rare words.
- **Dense search**: ranks chunks by vector closeness.
- **Hybrid search**: both kinds of search, combined.
- **RRF (Reciprocal Rank Fusion)**: combines rankings using `1/(60 + rank)`.
- **Filter**: a rule that removes chunks that cannot be answers (muscle, equipment, level).
- **Plannable pool**: the 622 strength and powerlifting exercises used by default.
- **recall@5**: share of the right answers found in the top 5, divided by the smaller of 5 and the number of right answers.
- **hit@5**: did the top 5 contain at least one right answer?
- **Gold answers**: the right answers, written by hand before the test.
- **Bootstrap interval**: a range built by resampling the test results to see if a difference could be luck.
- **Fractional set**: a set counted 1 for a main muscle and 0.5 for a helper muscle (our rule).
- **Typical minimum**: a population average from research, never a personal prescription.
- **Preprint**: a paper not yet peer-reviewed.
- **Fingerprint (SHA-256)**: a code that changes if a file changes by even one byte.
- **Quantisation (int8)**: storing model numbers in one byte instead of four, to make the model smaller.

---

# Viva: 10 questions

1. **What is a chunk, and why do we not store whole documents as one chunk?** A chunk is one searchable piece. A whole document mixes topics, so its one summary vector is blurry, and a long text is cut at 512 tokens.
2. **How do we split exercises and documents into chunks?** Each exercise is one chunk (876). Our documents are split on `#` and `##` headings (17 chunks). Smaller headings stay inside their parent.
3. **Why not split exercises into one chunk per instruction step?** A single step loses the exercise name, muscles and equipment, and the search could not tell which exercise it belongs to.
4. **What chunk length did we use, and how do you know?** We did not set a length; the split is by exercise and by heading. We measured the result: median 153 tokens, range 25 to 707, over 893 chunks, counted with the model's own tokenizer.
5. **Why is the limit 512 tokens and what happens past it?** The embedding model reads at most 512 tokens; anything beyond is cut off without an error. Two of our 893 chunks exceed it (measured). Their last part is never searched by meaning; keyword search still sees every word.
6. **Why 384 numbers per chunk?** That is the size of bge-small's output (verified from its configuration file). A bigger model would give bigger vectors and cost more memory and time.
7. **Why keyword and meaning search together?** Keywords win on exact names; meaning wins on paraphrases. On our 33 questions hybrid scored 0.903 against 0.833 and 0.841. The gap over meaning-only is not statistically clear with 33 questions.
8. **What is the "same model" rule and how is it enforced?** Chunks and queries must be embedded by the same model file. The index stores the model's fingerprint, and the loader refuses to run if it changes.
9. **Where does the 4-sets-per-week floor come from, and what are its limits?** It is Table 2A of Pelland et al. 2024, a preprint, which is an average across studies of mostly young adults over about 10 weeks, not muscle-specific. The app calls it a typical minimum.
10. **What did the first test show, and what did you do?** Filtered searches leaked cardio and olympic exercises. We fixed the filter, reran, and reported the second run next to the first. The gold answers were not changed.
