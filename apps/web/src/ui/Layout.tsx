import { motion, useReducedMotion, type Variants } from "motion/react";
import type { ReactNode } from "react";

const container: Variants = { hidden: {}, show: { transition: { staggerChildren: 0.09, delayChildren: 0.05 } } };

/** Entrance variant shared by every stagger child; no movement under prefers-reduced-motion. */
export function useItem(): Variants {
  const reduce = useReducedMotion();
  return {
    hidden: reduce ? { opacity: 1 } : { opacity: 0, y: 22 },
    show: { opacity: 1, y: 0, transition: { duration: reduce ? 0 : 0.6, ease: [0.22, 1, 0.36, 1] } },
  };
}

/** One /app step: children that are `Heading`, `Panel` or `motion` items with `useItem()` fade up in sequence.
 * `aside` (wide screens only) holds a sticky decorative figure. */
export function Screen({ children, aside, labelledBy }: { children: ReactNode; aside?: ReactNode; labelledBy?: string }) {
  const main = (
    <motion.section variants={container} initial="hidden" animate="show" aria-labelledby={labelledBy} className="min-w-0">
      {children}
    </motion.section>
  );
  if (!aside) return <div className="relative z-10 mx-auto max-w-2xl px-4 pb-10 pt-10 sm:px-6">{main}</div>;
  return (
    <div className="relative z-10 mx-auto max-w-6xl px-4 pb-10 pt-10 sm:px-6 lg:grid lg:grid-cols-[minmax(0,40rem)_1fr] lg:gap-16">
      {main}
      <aside className="hidden lg:block" aria-hidden>
        <div className="sticky top-10">{aside}</div>
      </aside>
    </div>
  );
}

/** A stagger container for layouts that `Screen` does not fit (the results page). */
export function Stagger({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <motion.div variants={container} initial="hidden" animate="show" className={className}>
      {children}
    </motion.div>
  );
}

export function Heading({ eyebrow, title, id, children }: { eyebrow?: string; title: string; id?: string; children?: ReactNode }) {
  return (
    <motion.header variants={useItem()} className="mb-8">
      {eyebrow && <p className="eyebrow">{eyebrow}</p>}
      <h1 id={id} className="display mt-3">
        {title}
      </h1>
      {children && <div className="mt-4 max-w-xl leading-relaxed text-ink-2">{children}</div>}
    </motion.header>
  );
}

export function Panel({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <motion.div variants={useItem()} className={`relative overflow-hidden rounded-2xl border border-line bg-card/50 p-6 sm:p-8 ${className}`}>
      <div className="absolute inset-x-8 top-0 h-px bg-gradient-to-r from-transparent via-accent/60 to-transparent" aria-hidden />
      {children}
    </motion.div>
  );
}

/** Wrapper for the button row (or any block) so it joins the stagger. */
export function Item({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <motion.div variants={useItem()} className={className}>
      {children}
    </motion.div>
  );
}

export function Notice({ tone = "info", title, children, role }: { tone?: "info" | "caution"; title?: string; children: ReactNode; role?: "alert" }) {
  const bar = tone === "caution" ? "bg-caution" : "bg-accent";
  return (
    <div role={role} className="relative rounded-xl border border-line bg-raised/50 py-4 pl-6 pr-5 text-sm leading-relaxed text-ink-2">
      <span className={`absolute inset-y-3 left-0 w-[3px] rounded-full ${bar}`} aria-hidden />
      {title && <p className="mb-1 font-medium text-ink">{title}</p>}
      {children}
    </div>
  );
}
