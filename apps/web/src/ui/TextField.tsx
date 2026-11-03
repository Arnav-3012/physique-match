import { useId, type InputHTMLAttributes } from "react";

type Props = { label: string; hint?: string; suffix?: string; fieldClassName?: string } & Omit<InputHTMLAttributes<HTMLInputElement>, "id">;

/** Labelled input with a gold focus state. `hint` is wired to the input with aria-describedby.
 * `fieldClassName` sizes the input box only, so the hint can run the full width. */
export default function TextField({ label, hint, suffix, className, fieldClassName = "", ...input }: Props) {
  const id = useId();
  return (
    <div className={className}>
      <label htmlFor={id} className="block text-[0.7rem] font-medium uppercase tracking-[0.18em] text-ink-3">
        {label}
      </label>
      <div className={`relative mt-2 ${fieldClassName}`}>
        <input
          id={id}
          aria-describedby={hint ? `${id}-hint` : undefined}
          className="w-full rounded-xl border border-line bg-raised/60 px-4 py-3.5 text-lg text-ink placeholder:text-ink-4 outline-none transition duration-300 focus:border-accent focus:bg-raised focus:shadow-[0_0_0_4px_rgb(200_168_106_/_0.12)]"
          {...input}
        />
        {suffix && <span className="pointer-events-none absolute right-4 top-1/2 -translate-y-1/2 text-sm text-ink-3">{suffix}</span>}
      </div>
      {hint && (
        <p id={`${id}-hint`} className="mt-2 text-sm text-ink-3">
          {hint}
        </p>
      )}
    </div>
  );
}
