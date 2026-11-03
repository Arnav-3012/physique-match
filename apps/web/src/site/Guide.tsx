import { Link } from "react-router-dom";
import { List, Page, Section } from "./Page";

export default function Guide() {
  return (
    <Page
      eyebrow="Guide"
      title="How to get a good scan"
      intro="These steps follow the capture flow in the app, in order."
    >
      <Section n="01" title="Choose what to compare">
        <p>The app asks once, before the camera starts. Nothing is forced, and the clothed scan always works.</p>
        <List
          items={[
            <>
              <strong className="text-ink">Arms, shoulders and legs:</strong> a sleeveless or tank top and shorts. No
              shirtless scan. Definition is not scored because the scan does not show the abdomen.
            </>,
            <>
              <strong className="text-ink">Full body (18+ only):</strong> men shirtless with shorts, women a sports bra
              with shorts, or a bare torso or crop top by choice. Gives one overall leanness estimate. For a bra scan the
              chest estimate is labelled approximate.
            </>,
            <>
              <strong className="text-ink">Clothed scan:</strong> fitted clothes. Size gaps and a body-fat range only;
              every definition score reads N/A with the reason &ldquo;clothed scan&rdquo;.
            </>,
          ]}
        />
        <p>You never need to remove clothes. Wear fitted gym wear: loose clothing makes the outline balloon and the results less accurate.</p>
      </Section>

      <Section n="02" title="Set up the space">
        <p>These are general tips for a cleaner outline, not measured limits.</p>
        <List
          items={[
            "Stand where your whole body is in frame, head to feet, with a plain wall or open space behind you.",
            "Use even light from the front. Avoid a bright window behind you.",
            "Prop the phone or laptop at about waist height and step back far enough to fit your full height.",
            "Keep your feet and hands inside the frame in every view.",
          ]}
        />
      </Section>

      <Section n="03" title="Take the four turns">
        <p>Front, left, back, right. You pace it yourself: hold still on the cue, the app picks the sharpest frame for each view and beeps when it has it. You can retake any view. You watch your live silhouette, not your camera feed (a toggle shows the feed for alignment).</p>
        <p>You can upload photos instead of using the camera. The same on-device processing runs on them.</p>
      </Section>

      <Section n="04" title="Add a goal and your numbers">
        <List
          items={[
            "Upload a goal photo. Front-only is fine; a side view is optional. Use a photo you have the rights to. It stays on your device.",
            "Say whose body it is (the goal's sex), choose a body-fat band and the regions you want to prioritise.",
            "Enter height, weight, age and your sex. Sex is never guessed from a photo and never filled in for you; leave it unset and you still get size gaps and a training plan, but no body-fat %, calorie targets or back width.",
            "Tick your equipment so the plan only uses what you have.",
          ]}
        />
      </Section>

      <Section n="05" title="Read your results">
        <List
          items={[
            <>
              <strong className="text-ink">Size gaps:</strong> goal minus current for each region, as a share of height.
              Each row has a confidence tier. Girth regions are less certain when the goal photo is front-only.
            </>,
            <>
              <strong className="text-ink">Body-fat range:</strong> always a range with the method named, never a single
              number. A goal photo&apos;s value is an appearance-based estimate.
            </>,
            <>
              <strong className="text-ink">Back width (estimate):</strong> inferred from chest, shoulder and waist
              ratios. It is not a measure of lat muscle and it is not ranked. Upper back size cannot be measured from a
              front photo.
            </>,
            <>
              <strong className="text-ink">Leanness and adiposity (experimental):</strong> shown only as goal against
              current. Experimental, and not yet validated on real photos.
            </>,
            "Goal photos are often posed, pumped, lit or edited. If a gap is larger than natural training is likely to close, the report says so and sets an intermediate target.",
          ]}
        />
      </Section>

      <Section title="What this is not">
        <p>
          It is not medical advice, and it never recommends supplements or drugs. Calorie targets are capped (a deficit of at most 20%, never below your estimated resting rate), and anyone with an eating-disorder history gets the training plan only. See a professional for injuries or health conditions.
        </p>
      </Section>

      <div className="flex flex-wrap gap-3 pt-4">
        <Link to="/app" className="btn btn-primary">
          Start your scan
        </Link>
        <Link to="/privacy" className="btn btn-ghost">
          Privacy
        </Link>
      </div>
    </Page>
  );
}
