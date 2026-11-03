import type { ReactNode } from "react";
import Reveal from "./Reveal";

export function Page({ eyebrow, title, intro, children }: { eyebrow: string; title: string; intro: string; children: ReactNode }) {
  return (
    <main className="mx-auto max-w-3xl px-4 pb-24 pt-14 sm:px-6">
      <Reveal>
        <p className="eyebrow">{eyebrow}</p>
        <h1 className="display mt-4 text-5xl sm:text-7xl">{title}</h1>
        <p className="mt-4 text-lg leading-relaxed text-ink-2">{intro}</p>
      </Reveal>
      <div className="mt-12 space-y-4">{children}</div>
    </main>
  );
}

export function Section({ n, title, children }: { n?: string; title: string; children: ReactNode }) {
  return (
    <Reveal>
      <section className="card p-6">
        <div className="flex items-baseline gap-3">
          {n && <span className="font-mono text-xs text-accent-2">{n}</span>}
          <h2 className="display text-3xl">{title}</h2>
        </div>
        <div className="mt-3 space-y-3 text-[0.95rem] leading-relaxed text-ink-2">{children}</div>
      </section>
    </Reveal>
  );
}

export const List = ({ items }: { items: ReactNode[] }) => (
  <ul className="space-y-2">
    {items.map((it, i) => (
      <li key={i} className="flex gap-3">
        <span className="mt-2 size-1.5 shrink-0 rounded-full bg-accent-2" aria-hidden />
        <span>{it}</span>
      </li>
    ))}
  </ul>
);
