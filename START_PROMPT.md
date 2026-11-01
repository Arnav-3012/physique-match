# First prompt for Claude Code
Unzip this folder as the repo root, `cd` into it, run `claude`, then paste everything below the line.

---

You are starting the PhysiqueGap project. All planning is done; your job is to execute it phase by phase.

1. Read, in this order: `CLAUDE.md`, `docs/project-overview.md`, `docs/architecture.md`, `docs/decisions.md`, `docs/privacy-and-safety.md`, `docs/phase-plan.md`, `docs/mistakes.md`, `learn/README.md`. Do not re-litigate accepted ADRs; if you think one is wrong, say so in one paragraph and wait for my answer.
2. Work only on the phase marked `IN PROGRESS` (P0). Turn its checklist into your task list.
3. Before writing code, reply with: (a) a ≤ 15-line plan for P0, (b) the exact commands you will run, (c) anything you need from me (accounts, keys, repo name, HF username). Ask all questions at once, then wait.
4. While working: apply `lean-senior` (full) and `long-story-short`; quiet commands; small commits with clear messages; no secrets in code; Python via `uv`, web via `npm`.
5. Keep `docs/logs/dev-log.md` updated each session and log bugs or wrong turns in `docs/mistakes.md` as they happen.
6. When every P0 exit gate passes, run the full end-of-phase protocol from `CLAUDE.md`, including `/sensei` to write `learn/phase-00/concepts.md`, `learn/phase-00/model-architecture.md` (write "no model this phase" if none) and `learn/phase-00/phase-overview.md`, at beginner-to-advanced depth using our real files and numbers.
7. Then stop, show me the gate results and the dev-log entry, and ask before starting P1.

Hard rules, repeated because they matter: no raw image ever leaves the browser except as a 1-bit mask; the server must reject anything else; LLMs receive text and numbers only; demo models use synthetic + BodyM data only; body-fat is always a range; energy targets respect the floors in `docs/privacy-and-safety.md`.
