import { useState } from "react";
import Choice from "../ui/Choice";
import { Heading, Item, Panel, Screen } from "../ui/Layout";
import SideFigure from "../ui/SideFigure";
import { PRIVACY_COPY } from "./privacyCopy";

export default function Consent({ onAccept }: { onAccept: () => void }) {
  const [adult, setAdult] = useState(false);

  return (
    <Screen labelledBy="consent-h" aside={<SideFigure caption="You watch a shadow, not yourself" />}>
      <Heading eyebrow="Step 1 · Before we start" title="Your photo stays with you" id="consent-h" />
      <Panel>
        <blockquote className="border-l-2 border-accent pl-5 text-lg leading-relaxed text-ink">{PRIVACY_COPY}</blockquote>
        <p className="mt-6 leading-relaxed text-ink-2">
          Before anything is sent you can preview exactly what: the review step shows the outlines. Next you choose what
          to compare, which decides what to wear; you never have to remove clothes.
        </p>
        <div className="mt-7">
          <Choice
            type="checkbox"
            checked={adult}
            onChange={() => setAdult(!adult)}
            title="I am 18 or older and I agree to the above."
          />
        </div>
      </Panel>
      <Item className="mt-8">
        <button className="btn btn-primary disabled:opacity-40" disabled={!adult} onClick={onAccept}>
          Continue
        </button>
      </Item>
    </Screen>
  );
}
