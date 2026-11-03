import { Link } from "react-router-dom";
import { Figure, RatioReadout, useAutoMorph } from "./Morph";
import Reveal from "./Reveal";
import Story from "./Story";

const WAYS = [
  { n: "01", title: "Scan", text: "Four slow turns in front of your camera, or file upload. Everything is read on your device." },
  { n: "02", title: "Goal", text: "Upload a goal photo (front only is fine), pick a body-fat band and the regions you care about." },
  { n: "03", title: "Gaps", text: "Per-region size gaps, each with a confidence tier, plus a body-fat range and a V-taper check." },
  { n: "04", title: "Plan", text: "A weekly plan that never drops a region below its cited minimum, with sources." },
];

const PILLARS = [
  { word: "Private", text: "Your photo is read on your device. Only an outline and your numbers are sent." },
  { word: "Per region", text: "Every part of your frame gets its own gap and its own confidence, not one vague score." },
  { word: "Sourced", text: "Weekly minimums and exercises come from cited research, shown in your plan." },
];

function Hero() {
  const p = useAutoMorph();
  return (
    <section className="mx-auto grid max-w-6xl items-center gap-10 px-4 pb-20 pt-14 sm:px-6 md:grid-cols-[1.25fr_1fr] md:pt-24">
      <div>
        <Reveal>
          <p className="eyebrow">Body composition, measured privately</p>
        </Reveal>
        <Reveal delay={0.06}>
          <h1 className="display mt-6 text-[3.4rem] sm:text-7xl lg:text-[6.6rem]">
            The distance between <em className="text-accent-2">where you are</em> and where you&apos;re going.
          </h1>
        </Reveal>
        <Reveal delay={0.12}>
          <p className="mt-7 max-w-xl text-lg leading-relaxed text-ink-2">
            Scan yourself in your browser, add a goal photo, and get per-region gaps, a body-fat range and a weekly
            training plan. Only a black-and-white outline and the numbers you type leave your device.
          </p>
        </Reveal>
        <Reveal delay={0.18} className="mt-9 flex flex-wrap gap-3">
          <Link to="/app" className="btn btn-primary">
            Start your scan
          </Link>
          <Link to="/guide" className="btn btn-ghost">
            How it works
          </Link>
        </Reveal>
        <Reveal delay={0.24}>
          <p className="mt-6 text-sm text-ink-3">18+ only · no account · nothing to install</p>
        </Reveal>
      </div>
      <Reveal delay={0.1} className="relative">
        <div className="mx-auto h-[440px] md:h-[600px]">
          <Figure p={p} />
        </div>
        <p className="mt-2 flex items-baseline justify-center gap-3 text-sm text-ink-3">
          Shoulder to waist
          <span className="display text-3xl text-accent-2">
            <RatioReadout p={p} />
          </span>
          <span className="text-xs">illustration</span>
        </p>
      </Reveal>
    </section>
  );
}

export default function Landing() {
  return (
    <main>
      <Hero />

      <section className="rule">
        <div className="mx-auto grid max-w-6xl px-4 sm:px-6 md:grid-cols-3">
          {PILLARS.map((n, i) => (
            <Reveal key={n.word} delay={i * 0.06} className={`px-4 py-14 text-center md:px-10 ${i ? "md:border-l md:border-line" : ""}`}>
              <p className="display text-5xl text-accent-2">{n.word}</p>
              <p className="mx-auto mt-3 max-w-xs text-sm leading-relaxed text-ink-2">{n.text}</p>
            </Reveal>
          ))}
        </div>
      </section>

      <Story />

      <section className="rule mx-auto max-w-6xl px-4 py-24 sm:px-6">
        <Reveal>
          <p className="eyebrow">Four steps</p>
          <h2 className="display mt-4 max-w-2xl text-4xl sm:text-6xl">From a scan to a plan.</h2>
        </Reveal>
        <ol className="mt-12">
          {WAYS.map((w, i) => (
            <Reveal key={w.title} delay={i * 0.05}>
              <li className="rule group grid items-baseline gap-2 py-7 transition-[padding] duration-300 hover:pl-3 md:grid-cols-[6rem_14rem_1fr]">
                <span className="font-mono text-xs text-accent">{w.n}</span>
                <h3 className="display text-4xl">{w.title}</h3>
                <p className="max-w-xl text-ink-2">{w.text}</p>
              </li>
            </Reveal>
          ))}
        </ol>
      </section>

      <section className="bg-emerald">
        <div className="mx-auto grid max-w-6xl gap-12 px-4 py-24 sm:px-6 md:grid-cols-[1fr_1.2fr]">
          <Reveal>
            <p className="eyebrow">Privacy</p>
            <h2 className="display mt-4 text-4xl sm:text-6xl">You watch a shadow, not yourself.</h2>
            <p className="mt-6 max-w-md text-ink-2">
              Photos are processed on this device. No photo is uploaded or saved. The server receives only a
              black-and-white outline, body joint positions and the numbers you enter.
            </p>
            <Link to="/privacy" className="btn btn-ghost mt-8">
              Read the full privacy page
            </Link>
          </Reveal>
          <Reveal delay={0.08} className="grid gap-8 sm:grid-cols-3">
            {[
              ["Stays on your device", "Camera frames, your face, your photo and your goal photo. Frames are released once the outlines are made."],
              ["Sent to the server", "A 1-bit outline with the head removed, 33 joint positions per view, and your typed numbers."],
              ["Not stored by our code", "The server holds the request in memory to compute your result. The host keeps its own technical logs."],
            ].map(([title, text]) => (
              <div key={title} className="border-t border-accent/50 pt-4">
                <h3 className="text-sm font-medium text-accent-2">{title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-ink-2">{text}</p>
              </div>
            ))}
          </Reveal>
        </div>
      </section>

      <section className="mx-auto max-w-4xl px-4 py-32 text-center sm:px-6">
        <Reveal>
          <h2 className="display text-5xl sm:text-7xl">Ready to see your gaps?</h2>
          <p className="mx-auto mt-6 max-w-md text-ink-2">No account, nothing to install. Not medical advice.</p>
          <Link to="/app" className="btn btn-primary mt-9">
            Start your scan
          </Link>
        </Reveal>
      </section>
    </main>
  );
}
