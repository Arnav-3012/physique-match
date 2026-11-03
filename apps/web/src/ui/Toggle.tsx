/** Small switch: a real checkbox, visually hidden, with a gold track. */
export default function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <label className="group inline-flex cursor-pointer items-center gap-3 text-sm text-ink-2">
      <input type="checkbox" className="sr-only" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span className="relative h-5 w-9 rounded-full bg-line transition duration-300 group-has-[:checked]:bg-accent group-has-[:focus-visible]:outline-2 group-has-[:focus-visible]:outline-offset-2 group-has-[:focus-visible]:outline-accent-2" aria-hidden>
        <span className="absolute left-0.5 top-0.5 size-4 rounded-full bg-ink transition duration-300 group-has-[:checked]:translate-x-4 group-has-[:checked]:bg-on-accent" />
      </span>
      {label}
    </label>
  );
}
