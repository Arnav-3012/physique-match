import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { BAND_LABELS, EXPERIMENTAL_BADGE } from "./bands";
import DefinitionBands from "./DefinitionBands";

const REGIONS = /\b(abs|chest|shoulders|arms|back|legs)\b/i;
const html = (definition: number, adiposity: number) =>
  renderToStaticMarkup(createElement(DefinitionBands, { scores: { definition, adiposity } }));

describe("the definition UI shows two bands and no per-region definition (ADR-053 Amendment 6)", () => {
  const all = [1, 2, 3, 4, 5].flatMap((d) => [1, 2, 3, 4, 5].map((a) => [d, a] as const));

  it("names no region for any pair of scores", () => {
    for (const [d, a] of all) expect(html(d, a)).not.toMatch(REGIONS);
  });

  it("renders exactly two rows, the overall band and the adiposity band, for any pair of scores", () => {
    for (const [d, a] of all) {
      const out = html(d, a);
      expect(out.match(/<dt>/g)).toHaveLength(2);
      expect(out).toContain(BAND_LABELS.definition.replaceAll("&", "&amp;")); // the markup escapes the ampersand
      expect(out).toContain(BAND_LABELS.adiposity);
    }
  });

  it("shows low / mid / high words, not the raw class numbers", () => {
    const out = html(2, 5);
    expect(out).toMatch(/>low</);
    expect(out).toMatch(/>high</);
    expect(out).not.toMatch(/>[1-5]</);
  });

  it("keeps the experimental badge and the adiposity caveat on screen", () => {
    const out = html(3, 3);
    expect(out).toContain(EXPERIMENTAL_BADGE);
    expect(out).toMatch(/waist-girth proxy/);
    expect(out).toMatch(/very muscular/);
    expect(out).toMatch(/Not medical advice/);
  });
});
