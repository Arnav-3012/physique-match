import type { ReactNode } from "react";

interface Props {
  type: "radio" | "checkbox";
  name?: string;
  checked: boolean;
  onChange: () => void;
  title: ReactNode;
  description?: ReactNode;
  disabled?: boolean;
  compact?: boolean;
}

/** A selectable card: a real (visually hidden) input inside a label, so keyboard and screen readers work. */
export default function Choice({ type, name, checked, onChange, title, description, disabled, compact }: Props) {
  return (
    <label className={`group relative block h-full ${disabled ? "cursor-not-allowed opacity-45" : "cursor-pointer"}`}>
      <input type={type} name={name} checked={checked} onChange={onChange} disabled={disabled} className="sr-only" />
      <span
        className={`flex h-full gap-4 rounded-xl border border-line bg-card/70 transition duration-300 hover:border-accent/50 group-has-[:checked]:border-accent group-has-[:checked]:bg-accent/[0.07] group-has-[:focus-visible]:outline-2 group-has-[:focus-visible]:outline-offset-2 group-has-[:focus-visible]:outline-accent-2 ${compact ? "items-center px-4 py-3" : "items-start p-5"}`}
      >
        <span
          className={`${compact ? "" : "mt-0.5"} grid size-[18px] shrink-0 place-items-center border border-ink-3 transition duration-300 group-has-[:checked]:border-accent group-has-[:checked]:bg-accent ${type === "radio" ? "rounded-full" : "rounded-[5px]"}`}
          aria-hidden
        >
          {type === "radio" ? (
            <span className="size-[7px] scale-0 rounded-full bg-on-accent transition duration-300 group-has-[:checked]:scale-100" />
          ) : (
            <svg viewBox="0 0 12 12" className="size-3 scale-0 text-on-accent transition duration-300 group-has-[:checked]:scale-100">
              <path d="M2.5 6.3 5 8.8l4.5-5.3" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          )}
        </span>
        <span className="min-w-0">
          <span className="block font-medium text-ink">{title}</span>
          {description && <span className="mt-1 block text-sm leading-relaxed text-ink-2">{description}</span>}
        </span>
      </span>
    </label>
  );
}
