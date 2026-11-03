# UI plan (P7, ADR-059)

Reference: **Linear** tokens (`awesome-design-md/references/linear.app.md`) + the layout rhythm of xnrgyclub.com
(giant display headline, two actions under it, short numbered sections, a four-tile "ways" grid, a stats strip).
Constraint: **no photos or videos from the owner.** Every visual is code: inline SVG, CSS gradients, one WebGL scene.
Dribbble fitness-app tag could not be fetched (empty response); only patterns from general knowledge are borrowed
(progress rings, one big numeral per card, count-up, bar fill). Screenshots from the owner may refine the palette.

## Pages
| Route | Purpose | Notes |
|---|---|---|
| `/` Landing | Hero, four tiles (Scan, Goal, Gaps, Plan), "what leaves your device" strip, numbers strip from our own logs, final CTA | 3D hero, lazy-loaded |
| `/guide` | How to take a good scan: clothing per scope, lighting, distance, 4 turns, what the scores mean, confidence tiers | static, SVG pose diagrams |
| `/privacy` | Rendered from `docs/privacy-and-safety.md` wording, never paraphrased into something weaker | static; same text the consent screen uses |
| `/app` | Consent -> scope -> capture -> review -> goal -> profile -> report -> plan (+ PDF) | the existing flow, restyled; **no 3D here** |

## Tokens (Linear, dark-first; light theme follows `prefers-color-scheme` later, not in the demo)
- Canvas `#08090a`, panel `#0f1011`, card `#191a1b`, hover `#28282c`.
- Text `#f7f8f8` / `#d0d6e0` / `#8a8f98` / `#62666d`.
- Accent indigo `#5e6ad2` (fills), `#7170ff` (links, active), `#828fff` (hover).
- Status: strong `#27a644`, caution amber `#e5a50a`, low-confidence grey `#8a8f98`. Never red for body numbers (no judgement colour on a body).
- Borders `rgba(255,255,255,0.05)` / `0.08`; overlay `rgba(0,0,0,0.85)`.
- Gradients: hero glow = radial `rgba(113,112,255,0.18)` to transparent behind the figure; bar fill = linear `#5e6ad2` -> `#828fff`; card top edge = 1px linear white 0.12 -> 0.
- Type: Inter Variable, self-hosted (`@fontsource-variable/inter`, no Google Fonts request: it would leak the visitor's IP to a third party). Features `cv01`, `ss03`. Display 72/64/48 px, weight 510, tracking -1.58/-1.41/-1.06 px; body 16 px / 1.5; mono for numbers: tabular figures.
- Radius 8 (cards 12, pills 999). Shadows: multi-layer with an inset 1px highlight (Linear/Raycast style).

## Motion (Framer Motion, `motion` package)
- Page load: staggered fade-up, 24 px, 0.5 s, ease `[0.22, 1, 0.36, 1]`, 60 ms stagger.
- Scroll reveals: `whileInView`, once.
- Numbers count up on first view; bars fill from 0 with a 0.8 s spring; rings draw via `stroke-dashoffset`.
- Step changes in `/app`: shared-axis slide + fade, 0.25 s.
- Buttons: 1.02 scale on hover, 0.98 on press; focus ring `#7170ff` 2 px.
- `prefers-reduced-motion`: all of it off (instant states). Required, not optional.

## 3D hero (landing only)
- `three` + `@react-three/fiber` + `@react-three/drei`, dynamic `import()` on the landing route only, so the app bundle and the capture screens never carry it (MediaPipe and the definition model need the GPU there).
- Procedural mannequin from capsules and spheres (no model file, no asset licence): torso, limbs, no head detail, matte material in `#191a1b` with an indigo rim light. Slow idle rotation; pointer moves it a few degrees; regions pulse in sequence (shoulders, waist, thighs) to hint at "gaps".
- Fallback: an inline SVG silhouette if WebGL is unavailable, on `prefers-reduced-motion`, or when `navigator.hardwareConcurrency <= 4` on a phone-width screen.
- Budget: three adds about 150 kB gzip on the landing route only; lazy so the first paint is the text.

## Screens of `/app` (restyle only; what is sent to the server does not change)
Consent, scope, capture, review keep their logic. New: goal -> **Continue** -> profile (sex with no default, height, weight, days, equipment checklist, optional tape waist) -> report -> plan. Report: gap bars with reliability tier, priority chips, "typical minimum" per region, body-fat band, leanness and adiposity as goal-vs-current, N/A rows with reasons, warnings and notes as given, `served_by` badge. Plan: day cards, weekly volume vs baseline, shortfalls named, PDF made in the browser.

## Results page (owner request 2026-11-03): after the scan and consent, one page with two parts
Route `/app/results`, reached from the review screen's Continue after goal and profile are set. Desktop: two columns (report left, assistant right); phone: report first, assistant as a bottom sheet.
1. **Goal vs current report** (the main part). Hero row: three big numerals with count-up (largest priority gap, goal body-fat band vs current range, V-taper index goal vs current). Below: a **paired bar chart per region**, current and goal drawn on the same axis with the gap as a tinted segment between them, sorted by priority, each row with its reliability tier chip and its "typical minimum" weekly volume. A silhouette with the same regions tinted by gap size sits beside it. Then: leanness and adiposity as goal vs current with the "experimental" badge, the realism note, warnings, N/A rows with their reasons, and the plan as day cards. Charts are hand-drawn SVG (no chart library), animated bar fill and tooltips.
2. **Assistant (chat style).** Messages and cards appear in sequence ("here is your report", "here is your plan"), then tappable question chips ("why this plan", "why these exercises", "what is a typical minimum", "why no calorie numbers", "what does experimental mean"). Answers are written by code from the planner's own facts, so no free text reaches a model and there is no extra token cost. **Typed questions are in the demo (owner, 2026-11-03), behind `POST /chat` and ADR-060**, built in this order: (1) `evals/chat.jsonl` written and frozen first (on-topic, off-topic, prompt injection, self-harm / eating-disorder, medical, "make me X kg in Y days", drug and supplement asks, under-18, another person's body, very long input), with its hash logged; (2) deterministic gate before any model: length cap (300 chars), no URLs or HTML, the frozen unsafe patterns from `evals/patterns.py`, a topic allow-list check (embedding router over the question chips and the KB titles), fixed refusal texts written by code; (3) the model sees only the report's facts block plus the question, no history beyond the last 2 turns, answers as JSON with a closed `kind` (`answer` / `refuse`), no tools; (4) the same text-safety check as the plan writer on the output, a failure becomes a fixed "try one of these questions" message; (5) nothing is stored, and the question text and the answer are never logged (counts and outcome only); (6) a per-client limit tighter than `/plan`, and a 429 or provider failure shows "chat is busy, tap a question instead", so the chips always work. Token cost about 1.5k per question: on Groq's free tier (8,000 tokens a minute) roughly 4 typed questions a minute across all users, stated in the UI as a limit, not hidden. ED history or under-18 profile: typed chat is disabled and chips only, with the supportive note.
3. States: loading skeleton while `/analyze` and `/plan` run, "waking server" after 3 s, error card with retry, 429 message with the Retry-After.

## Safety copy that must stay visible and legible (docs/privacy-and-safety.md)
"experimental", "appearance-based", "approximate, fitness-industry categories", "back width (estimate)", low-confidence markers, the privacy line, the ED / under-18 notes. Minimum contrast 4.5:1 on these (checked against `#08090a`: `#8a8f98` is 6.2:1, `#62666d` is 3.2:1 and is **not** allowed for them).

## Build order
0. Done 2026-11-03 (written, not yet built or viewed): foundation, landing, guide, privacy, 3D hero with SVG fallback; `/app` wrapped in a temporary light wrapper (`.legacy-light`) until its screens are restyled. No timing claims ("about 2 minutes") until measured. **When `/chat` ships, `Privacy.tsx` and the consent text must say typed questions go to a language model, and the line "only numbers go to them" changes.**
1. Foundation: deps, tokens in `index.css` (`@theme`), fonts, router, layout shell, Vercel SPA rewrite.
2. Landing (CSS/SVG first, then the 3D hero), Guide, Privacy.
3. Restyle consent, scope, capture, review, goal (no logic change).
4. API client: `/analyze`, `/plan`; profile screen; report and plan screens; PDF.
5. Public API: `X-Forwarded-For` rate-limit key, Upstash limiter, CORS = the Vercel origin, make the service public.
6. Vercel project (root `apps/web`), `VITE_API_URL`, full-flow gate from the public URL.
