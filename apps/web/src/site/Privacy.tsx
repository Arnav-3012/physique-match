import { PRIVACY_COPY } from "../capture/privacyCopy";
import { List, Page, Section } from "./Page";

/** Wording follows docs/privacy-and-safety.md. The core paragraph is PRIVACY_COPY itself (a test pins it). */
export default function Privacy() {
  return (
    <Page
      eyebrow="Privacy"
      title="Your photo never leaves your device"
      intro="What runs where, what is sent, and what we can and cannot promise."
    >
      <Section title="In one paragraph">
        <blockquote className="border-l-2 border-accent-2 pl-4 text-ink">{PRIVACY_COPY}</blockquote>
      </Section>

      <Section title="What the server receives">
        <List
          items={[
            "A 1-bit black-and-white outline per view, with the head removed.",
            "33 joint positions per view.",
            "The numbers you type: height, weight, age, your sex and the goal's sex (as choices), tape waist if you add it, your goal and your equipment.",
            "The server rejects anything that is not a binary mask, and anything over 512 KB.",
          ]}
        />
      </Section>

      <Section title="What we do not do">
        <List
          items={[
            "No account, and no scan stored by our code. The server computes your result from the request in memory and returns it.",
            "No photo goes to any language model. Free hosted models may train on prompts, so only numbers go to them.",
            "No photo, thumbnail or video frame is written to your browser's storage or to a file by this app, by design.",
          ]}
        />
        <p className="text-ink-3">
          Honest limit: &ldquo;by design&rdquo; rests partly on a code review, not a runtime observation. The check that
          confirms it in a real browser, and the check that the hosting service&apos;s logs hold no outline, are run before
          launch and their results will be added here.
        </p>
      </Section>

      <Section title="The leanness model runs in your browser">
        <p>
          It is off unless you turn it on. When on, small models read the front photo of you and of your goal on your
          device and produce two numbers each, an overall leanness and muscularity estimate and an adiposity estimate;
          if you add back photos, a third rough back-development band is read from those. All of these numbers, and a
          back photo&apos;s outline, stay in your browser and are not sent. It is experimental, not yet validated on real photos, and it never decides whether you cut, recomp or
          bulk.
        </p>
      </Section>

      <Section title="Clothing and what is asked">
        <p>
          You choose what to compare before the camera starts. The full-body option is for adults only and the server
          rejects it for anyone under 18. Nothing is required: the clothed scan always works, and N/A always comes with
          a reason.
        </p>
      </Section>

      <Section title="Sex fields">
        <p>
          Sex is only used for body-fat %, calories, back width and the V-taper ceiling. You type it, it is sent as a
          choice, it is not stored, and it is never read from your photo. Leave it unset and everything else still works.
        </p>
      </Section>

      <Section title="Hosting and logs">
        <p>
          The API runs on Google Cloud Run. The host writes its own technical request logs (time, address, status),
          outside our code, so the host can see your IP address. Rate-limit counters hold a client key and timestamps for
          about a minute and no body.
        </p>
      </Section>

      <Section title="Body-image safety">
        <List
          items={[
            "Language is about development opportunities, never flaws.",
            "Goal photos are often posed, pumped, lit or edited, and the report says so once.",
            "No supplement or drug advice.",
            "Calorie deficits are capped at 20% and about 1% of body weight a week, never below your estimated resting rate; surpluses at 10%.",
            "No cutting plan if your BMI is under 18.5 or your estimated body fat is below the safety floor. You get maintenance and strength work, and a note to see a professional.",
            "An optional eating-disorder question turns off calorie numbers and shows a supportive note.",
          ]}
        />
        <p>Every report ends with: not medical advice; see a professional for injuries or health conditions.</p>
      </Section>
    </Page>
  );
}
