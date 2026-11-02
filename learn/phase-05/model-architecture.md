# P5 models and search path, from zero

P5 uses one neural model, the embedding model bge-small, and two plain (non-learned) ranking methods, BM25 and RRF. No language model writes text in P5; that is P6. This file follows one question through the whole system so you can see every part working together.

**Source key.** Numbers come from `kb/manifest.json` (index totals), `services/api/app/rag/embed.py`, `retrieve.py`, `query.py` and `corpus.py` (the code), `kb/sources.md` (the model's facts and fingerprints), and `evals/results/rag_run2.json` (the results). Anything I could not confirm from a file in the repo is marked **(not verified here)**.

---

## 1. The whole path, in one picture

Our planner asks: "chest, dumbbells, beginner". Here is what happens to that request:

```
"chest", equipment ["dumbbells"], level "beginner"
        │
        ▼  query.py: split into one search per muscle group
   "chest exercise"  + filters (muscle=chest, equipment=dumbbell, level=beginner)
        │
        ▼  retrieve.py: throw out cards that cannot match the filters
   a smaller set of cards (exercises, strength/powerlifting, chest, dumbbell, beginner)
        │
        ├──► BM25 (keywords) ──────────── top 20 cards ─┐
        │                                               ├─► RRF (combine ranks) ─► top 5 cards
        └──► dense (meaning) ─► top 20 cards ───────────┘
                 ▲
                 │  embed.py: the query became 384 numbers
              bge-small model
```

Each box is explained below. If you only remember one thing: **each card is turned into 384 numbers once, when we build the index. Each question is turned into 384 numbers at search time. Cards whose numbers point the same way as the question's numbers are the best matches.**

---

## 2. Embedding: turning text into 384 numbers

**Plain picture.** Imagine giving every piece of text a position on a very large map of meanings. Texts about the same idea land close together. The 384 numbers are the coordinates on that map.

**What the model is.** bge-small-en-v1.5 is a small neural network (a language model made only to produce these coordinates, not to write). It is from BAAI, licensed MIT, and we use a compressed version (int8: each number stored in one byte instead of four). Its file is 34,014,426 bytes. Its fingerprint (SHA-256) is `6c9c6101...dfe4`, recorded in `kb/sources.md`.

**Facts we checked from files.**
- Output size: 384 numbers per text (`DIM = 384` in `embed.py`; dimension in the model's configuration, `kb/sources.md`).
- Maximum input: 512 tokens (`MAX_LEN = 512` in `embed.py`). A token is roughly a word or part of one.
- Pooling: the first token's output is the summary (CLS pooling, from BAAI's `1_Pooling/config.json`).
- Query prefix: "Represent this sentence for searching relevant passages: " is added to queries only.

**What the code does, step by step** (`Embedder._embed` in `embed.py`):
1. The tokenizer (`tokenizer.json`, 711,396 bytes) turns the text into token ids and adds padding and truncation at 512.
2. The model runs and gives one vector per token.
3. We keep the first token's vector (CLS).
4. We divide it by its own length, so every vector has length 1. Then "how similar are two texts" becomes a simple dot product.

**Why one text at a time.** The compressed model gives numbers that differ by about 0.001 depending on how many texts are processed together, because of padding. We found this when a test failed. The fix is that cards and questions both go through the model one at a time, so they take the same path.

**Why not a bigger model?** A larger model would capture meaning more finely but would be several times bigger and slower. Our corpus is 893 cards; this model is enough for the demo and fits the no-torch, small-image rule in ADR-028.

**What breaks it, and the guard.**

| What could go wrong | Guard in the code |
|---|---|
| Cards embedded by a different model file than the question | the index stores the model's fingerprint; `corpus.py` recomputes it and refuses to load if it differs |
| Chunks longer than 512 tokens get cut | **measured**: 2 of 893 chunks (user-run, 2026-11-02): `ex:Power_Clean` (707 tokens, about 195 cut) and `ex:Clean_and_Jerk` (565, about 53 cut). `Clean_and_Jerk` is olympic weightlifting (outside the default pool); `Power_Clean` is strength (inside it, truncated vector). No claim about the planner, P6 does not exist. Left as is by decision, ADR-048 Amendment 4 |
| Passages get the query prefix | never done; `embed_passages` does not add it |
| Vectors not length 1 | `build_index.py` checks this when it builds the index |

---

## 3. The index on disk

Built once, offline (by `kb/build_index.py`), committed to the repo:

| File | What it holds | Size |
|---|---|---|
| `kb/chunks.jsonl` | one line of JSON per chunk: id, text, muscle, equipment, level and more | text file |
| `kb/embeddings.npy` | the 893 vectors, each 384 numbers | 1,371,648 bytes |
| `kb/manifest.json` | counts (893 chunks, 876 exercise, 17 guideline), dimensions, fingerprints, `corpus_hash` | small |

Why store the vectors instead of computing them at each search? The corpus is fixed, so computing it once is cheaper. Only the question is embedded at search time: one short text through the model.

**The corpus fingerprint.** `corpus_hash` is a SHA-256 of the chunk text plus both model fingerprints. Current value `9ce6bc82...b8b0`. If anyone edits a chunk or swaps the model, the hash changes and the loader refuses to run until the index is rebuilt.

---

## 4. Dense search: find the closest meanings

**Plain picture.** We have 893 points on the meaning map and one question point. Dense search measures how close each card's point is to the question's point, and keeps the 20 closest.

**The arithmetic.** Each card's vector and the question's vector are length 1. Their dot product is then the cosine similarity: 1 means same direction, 0 means unrelated. The code is one line (`corpus.embeddings @ query_vec` in `retrieve.py`), a single multiplication of a 893-by-384 table by a 384-number vector. That is about 343,000 multiplications, which is fast.

**A filter is applied first:** cards that do not match are given a score of minus infinity, so they never reach the top 20.

**Why it helps.** "Press-up on the floor" finds "Pushups" even though no word matches exactly. That is the meaning doing the work.

**Why it fails.** It can miss exact names. Our results: dense search has recall@5 0.841 overall, but 0.800 on synonyms and 0.778 on guideline questions.

---

## 5. Keyword search: BM25

**Plain picture.** Find the cards that share words with the question, and give more weight to rare words. "Romanian" is rare, so a card with that word scores high; "the" is in almost every card, so it scores nothing.

**How it works, in words.** For each word in the question and each card:
- the rarer the word across all cards, the more it is worth;
- a word that appears again in the same card is worth a little more, but with diminishing returns;
- long cards are slightly penalised, so they do not win by size.

We use the library's standard settings. The formula itself is in the `rank_bm25` library; we did not re-read its source for this document **(not verified here)**.

**Tokens.** The text is lowercased and split into runs of letters and digits (`tokens()` in `retrieve.py`). No word-stemming, no synonyms. That is why "skull crusher" does not match "Skullcrusher".

**Why it helps.** Exact names. Romanian Deadlift scored 1.00 exact.

**Why it fails.** Meaning. Synonym questions scored 0.667.

A card with no matching word gets a score of 0 and is not a candidate at all. Dense search has no such cut-off, so it always returns 20 cards if 20 pass the filter.

---

## 6. RRF: combining the two rankings

**Plain picture.** Two judges rank the cards. Their scores are on different scales, so we ignore the scores and use only the ranks.

**The formula.** For each card, sum `1 / (60 + rank)` over the lists where it appears. Rank 1 is the top.

| Card | BM25 rank | Dense rank | Score |
|---|---|---|---|
| A | 1 | 3 | 1/61 + 1/63 = 0.0323 |
| B | 2 | (not in top 20) | 1/62 = 0.0161 |
| C | (not in top 20) | 1 | 1/61 = 0.0164 |

A wins because two judges agree. C beats B although B is ranked higher by BM25, because a rank 1 in one list is worth slightly more than a rank 2 in another. Those small differences are why the number 60 matters: a bigger number flattens the differences.

**Why 60?** It is the value from the original paper (Cormack, Clarke and Buettcher, 2009). We fixed it before our first run and did not change it.

**Why no weight between the two judges?** A weight would have to be tuned, and 33 questions are too few to tune anything fairly.

**Code.** `search(mode="hybrid")` in `retrieve.py` builds the combined score and sorts by score, then by card position so ties always come out the same way.

---

## 7. The filters, then the search

Filters run before both judges, in `Retriever.mask` (`retrieve.py`). A card survives only if all of these hold:
- it is an exercise (guidelines are removed under any filter);
- it is in the plannable pool (strength or powerlifting), unless the search opts out;
- its main muscles include the requested muscle;
- its equipment matches exactly;
- its level is the requested one.

With no filter at all, every card is allowed, including guidelines.

**The bug we fixed.** Before the pool rule was in `mask`, filtered searches returned cardio and olympic exercises, because the rule was written in the ingest code and the ADR but not used by the search. Run 1 caught it. Run 2 (0.903) is after the fix.

---

## 8. What the system does not do

- It does not rank exercises by quality. It ranks by how well they match the question. Choosing the best of seven dumbbell presses is the planner's job (P6).
- It does not understand free text. Only the closed vocabulary of muscle, equipment and level reaches the search. The intent router that would handle free text is deferred, and it is only safe to defer while the planner sends no free text (to be checked in P6).
- It does not know if the library's labels are wrong. A wrong muscle label becomes a wrong result.
- With 33 questions, small differences are not reliable. The gap between hybrid and meaning-only has an interval of -0.017 to +0.159.

---

# Glossary

- **Embedding model**: a neural network that turns text into a list of numbers that reflects its meaning.
- **bge-small-en-v1.5**: the embedding model we use. 384 outputs, MIT licence.
- **int8**: storing each model number in one byte instead of four, to make the file smaller.
- **Token**: a small piece of text, roughly a word or part of one. The model reads at most 512.
- **Tokenizer**: the program that splits text into tokens.
- **CLS pooling**: taking the first token's output as the whole text's summary.
- **Vector**: a list of numbers; here 384 of them.
- **L2 normalisation**: dividing a vector by its own length so it has length 1.
- **Dot product**: multiply matching numbers and add them up; for unit vectors it equals cosine similarity.
- **Cosine similarity**: how closely two vectors point the same way (1 = same direction).
- **Index**: the stored cards, vectors and fingerprints.
- **Fingerprint (SHA-256)**: a code that changes if a file changes at all.
- **Corpus hash**: a fingerprint of all chunk text plus both model fingerprints.
- **BM25**: a keyword ranking that rewards rare words and penalises padding-length cards.
- **RRF**: Reciprocal Rank Fusion; combines rankings with `1/(60 + rank)`.
- **Hybrid search**: BM25 and dense search combined with RRF.
- **Filter**: a rule that removes cards that cannot be answers.
- **Plannable pool**: the 622 strength and powerlifting exercises used by default.
- **Negative result**: a test outcome that does not support the hoped-for claim, reported as it is.

---

# Viva: 10 questions

1. **What happens to a planner request between the planner and the result?** It is split into one search per muscle group, filtered, searched by keywords and by meaning, and the two rankings are combined with RRF into the top 5.
2. **What are the 384 numbers?** The model's summary of a text's meaning. Similar texts get numbers that point the same way.
3. **Why are the card vectors computed once but the question vector at search time?** The cards do not change, so computing them once is cheaper. A question is new each time.
4. **Why do we keep the first token's output as the summary?** The model is trained so that this position summarises the whole text (CLS pooling, from the model's configuration).
5. **Why do questions get a prefix and cards do not?** The model's authors ask for it: it tells the model "this is a search", so it can match questions to passages.
6. **What is the 512 limit and what happens past it?** The model reads at most 512 tokens and cuts off the rest without an error. Measured: 2 of our 893 cards go past it; the median card is 153 tokens and the longest 707.
7. **Why does BM25 miss "skull crusher"?** It matches exact words, and the library spells the exercise "Skullcrusher".
8. **Why do we use ranks instead of scores in RRF?** The two scores are on different scales, so adding them would be meaningless without a tuned weight.
9. **Why not tune the 60, or the weight between the two searches?** With 33 questions, any tuning would fit the test rather than the real behaviour.
10. **Which of these results are solid, and which are not?** Hybrid beats keywords-only with a clear interval (+0.070, from +0.011 to +0.147). Hybrid beats meaning-only on the point estimate, but the interval includes zero (+0.062, from -0.017 to +0.159), so that part is not settled.
