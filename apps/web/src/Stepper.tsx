const STEPS = ["Consent", "Option", "Scan", "Review", "Goal", "Profile", "Results"];

/** Where the user is in the /app flow. Decoration for orientation only; no state is read from it. */
export default function Stepper({ current }: { current: number }) {
  return (
    <nav aria-label="Progress" className="no-print relative z-10 mx-auto max-w-2xl px-4 pt-6">
      <ol className="flex gap-1.5">
        {STEPS.map((label, i) => (
          <li key={label} className="flex-1" aria-current={i === current ? "step" : undefined}>
            <div
              className="h-[3px] rounded-full transition-colors duration-500"
              style={{ background: i <= current ? "var(--color-accent)" : "var(--color-line)" }}
            />
            <p className={`mt-2 hidden text-[0.68rem] uppercase tracking-[0.16em] sm:block ${i === current ? "text-accent-2" : "text-ink-4"}`}>
              {label}
            </p>
          </li>
        ))}
      </ol>
      <p className="mt-2 text-xs uppercase tracking-[0.16em] text-accent-2 sm:hidden">
        {current + 1} of {STEPS.length} · {STEPS[current]}
      </p>
    </nav>
  );
}
