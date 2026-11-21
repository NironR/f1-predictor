// Bakes the latest race into index.html, so the page reads the same without scripts (and before
// they run). Run after build_site.py writes data.js: node docs/prerender.mjs
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
globalThis.window = {};
await import("./data.js");
const { pageSlots, raceSlots } = await import("./render.js");

const D = window.F1, latest = D.races.at(-1);
const slots = { ...pageSlots(D), ...raceSlots(D, latest) };
const file = join(here, "index.html");
let html = readFileSync(file, "utf8"), missing = [];
for (const [name, content] of Object.entries(slots)) {
  const re = new RegExp(`(<!--rp:${name}-->)[\\s\\S]*?(<!--/rp:${name}-->)`, "g");
  if (!re.test(html)) { missing.push(name); continue; }
  html = html.replace(re, (_, a, b) => a + content + b);
}
html = html.replace(/(data-rp="hero-label"[^>]*?aria-label=")[^"]*(")|(aria-label=")[^"]*("[^>]*?data-rp="hero-label")/,
  (m, a, b, c, d) => (a ? a + `The latest race: ${latest.name} ${latest.season}, predicted order beside the result` + b
                        : c + `The latest race: ${latest.name} ${latest.season}, predicted order beside the result` + d));
writeFileSync(file, html);
console.log(`prerendered ${latest.name} ${latest.season}${missing.length ? `; no marker for: ${missing.join(", ")}` : ""}`);
