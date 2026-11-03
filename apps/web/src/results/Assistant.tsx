import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useEffect, useRef, useState } from "react";
import { answer, QUESTIONS, type Facts, type QuestionId } from "./answers";

interface Msg {
  id: number;
  from: "bot" | "me";
  paras: string[];
}

/** A chat-style guide to the report. Answers are written by code from the user's own results (ADR-059), so the
 * chips always work and nothing here reaches a language model. Typed questions arrive with their own safety layer. */
export default function Assistant({ facts }: { facts: Facts }) {
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [asked, setAsked] = useState<Set<QuestionId>>(new Set());
  const seq = useRef(0);
  const box = useRef<HTMLDivElement>(null);
  const reduce = useReducedMotion();

  const push = (from: Msg["from"], paras: string[]) => setMsgs((m) => [...m, { id: seq.current++, from, paras }]);

  useEffect(() => {
    const t = [
      setTimeout(() => push("bot", ["Your report is on the left: size gaps, body-fat range and calories."]), 400),
      setTimeout(
        () => push("bot", [facts.plan ? "Your weekly plan is below it. Ask me how any of it was built." : "Your size gaps are ready. The plan could not be built just now."]),
        1300,
      ),
    ];
    return () => t.forEach(clearTimeout);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    box.current?.scrollTo({ top: box.current.scrollHeight, behavior: reduce ? "auto" : "smooth" });
  }, [msgs.length, reduce]);

  const ask = (id: QuestionId, text: string) => {
    setAsked((s) => new Set(s).add(id));
    push("me", [text]);
    setTimeout(() => push("bot", answer(id, facts)), 450);
  };

  return (
    <section className="flex h-full flex-col rounded-2xl border border-line bg-card/60" aria-label="Report assistant">
      <header className="flex items-center gap-3 border-b border-line-soft px-5 py-4">
        <span className="grid size-8 place-items-center rounded-full bg-accent/15 text-accent-2" aria-hidden>
          <span className="size-2 rotate-45 bg-accent" />
        </span>
        <div>
          <p className="display text-xl leading-none">Ask about your report</p>
          <p className="mt-1 text-xs text-ink-3">Answers come from your own numbers</p>
        </div>
      </header>

      <div ref={box} className="min-h-[14rem] flex-1 space-y-3 overflow-y-auto px-5 py-5 lg:max-h-[46vh]" aria-live="polite">
        <AnimatePresence initial={false}>
          {msgs.map((m) => (
            <motion.div
              key={m.id}
              initial={reduce ? false : { opacity: 0, y: 12, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
              className={`max-w-[92%] space-y-2 rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                m.from === "me" ? "ml-auto rounded-br-md bg-accent text-on-accent" : "rounded-bl-md border border-line bg-raised/70 text-ink-2"
              }`}
            >
              {m.paras.map((p, i) => (
                <p key={i}>{p}</p>
              ))}
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      <div className="border-t border-line-soft px-5 py-4">
        <div className="flex flex-wrap gap-2">
          {QUESTIONS.map((q) => (
            <button
              key={q.id}
              onClick={() => ask(q.id, q.text)}
              className={`rounded-full border px-3.5 py-2 text-xs transition duration-300 ${
                asked.has(q.id) ? "border-line text-ink-3" : "border-accent/50 text-accent-2 hover:bg-accent/10"
              }`}
            >
              {q.text}
            </button>
          ))}
        </div>
        <input
          disabled
          aria-label="Typed questions"
          placeholder="Tap a question above"
          className="mt-3 w-full cursor-not-allowed rounded-xl border border-line bg-raised/40 px-4 py-3 text-sm text-ink-3 placeholder:text-ink-4"
        />
      </div>
    </section>
  );
}
