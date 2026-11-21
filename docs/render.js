// Pure HTML builders for the race explorer, shared by the page (app.js) and the build-time
// prerender (prerender.mjs), so the page reads the same with or without scripts.

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const short = (name) => name.replace(" Grand Prix", " GP");
const pct = (x, d = 0) => `${(x * 100).toFixed(d)}%`;
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
export const day = (iso) => (iso ? `${+iso.slice(8, 10)} ${MONTHS[+iso.slice(5, 7) - 1]}` : "");
export const key = (r) => `${r.season}-${r.round}`;
const signed = (n) => (n > 0 ? `+${n}` : n < 0 ? `−${-n}` : "0");
const COMP = { SOFT: "Soft", MEDIUM: "Medium", HARD: "Hard" };

function ranks(values) {  // average ranks, ties shared
  const order = values.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
  const out = new Array(values.length);
  for (let i = 0; i < order.length;) {
    let j = i;
    while (j + 1 < order.length && order[j + 1][0] === order[i][0]) j++;
    for (let k = i; k <= j; k++) out[order[k][1]] = (i + j) / 2 + 1;
    i = j + 1;
  }
  return out;
}
function spearman(a, b) {
  const ra = ranks(a), rb = ranks(b), n = a.length;
  const ma = ra.reduce((s, v) => s + v, 0) / n, mb = rb.reduce((s, v) => s + v, 0) / n;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) { num += (ra[i] - ma) * (rb[i] - mb); da += (ra[i] - ma) ** 2; db += (rb[i] - mb) ** 2; }
  return num / Math.sqrt(da * db);
}
export function scores(race) {
  const f = race.finish;
  return { model: spearman(f.map((d) => d.pred), f.map((d) => d.actual)),
           grid: spearman(f.map((d) => d.grid), f.map((d) => d.actual)) };
}

// ---- hero window ----
export function heroRecent(D, race) {
  const recent = D.races.filter((r) => r.season === race.season && r.round <= race.round).slice(-4).reverse();
  return recent.map((r) => `<p${r === race ? ' class="is-on"' : ""}>${esc(short(r.name))}<small>Round ${r.round} · ${day(r.date)}</small></p>`).join("");
}
export function heroTop(race) {
  const head = `<div class="rp-top__row rp-top__row--head"><span>Pos</span><span>Driver</span><span>Grid</span><span>Predicted</span><span>Beat by</span></div>`;
  return head + race.finish.slice(0, 8).map((d) => {
    const g = d.pred - d.actual;
    const bar = `<svg viewBox="0 0 24 6" aria-hidden="true"><rect x="${g < 0 ? 12 - Math.min(12, -g * 3) : 12}" y="0" width="${Math.max(1, Math.min(12, Math.abs(g) * 3))}" height="6" rx="1" fill="currentColor"/></svg>`;
    return `<div class="rp-top__row rp-mono"><span>${d.actual}</span><span>${esc(d.name)}</span><span>${d.grid > 20 ? "Pits" : d.grid}</span><span>${d.pred}</span><span class="rp-gain ${g >= 0 ? "rp-up" : "rp-down"}${g >= 3 ? " rp-mark" : ""}">${signed(g)}${bar}</span></div>`;
  }).join("");
}

// ---- the race list ----
export function seasons(D, season, phone = false) {
  const all = [...new Set(D.races.map((r) => r.season))].reverse();
  return (phone ? "" : `<span class="ui-win__label">Seasons</span>`) + all.map((s) => {
    const n = D.races.filter((r) => r.season === s).length;
    return phone
      ? `<button class="rp-season" type="button" data-season="${s}" aria-pressed="${s === season}">${s}</button>`
      : `<button class="rp-btn ui-win__nav${s === season ? " is-on" : ""}" type="button" data-season="${s}" aria-pressed="${s === season}">${s}<em>${n}</em></button>`;
  }).join("");
}
export function rows(D, season, selected) {
  return D.races.filter((r) => r.season === season).slice().reverse().map((r) => {
    const s = scores(r), gap = s.model - s.grid;
    const [label, cls] = gap > 0.02 ? ["Beat the grid", " is-done"] : gap < -0.02 ? ["Behind the grid", " is-you"] : ["Level", ""];
    return `<li class="ui-win__row" data-row data-key="${key(r)}" role="button" tabindex="0" aria-pressed="${key(r) === selected}">`
      + `<span class="ui-avatar" data-tone="${(r.round % 5) + 1}">${r.round}</span>`
      + `<span class="ui-win__who">${esc(short(r.name))}</span>`
      + `<span class="ui-win__sub rp-mono">Model ${s.model.toFixed(2)} · grid ${s.grid.toFixed(2)}</span>`
      + `<span class="ui-win__st${cls}" data-st>${label}</span>`
      + `<span class="ui-win__age">${day(r.date)}</span></li>`;
  }).join("");
}

// ---- the open race: grid, prediction and result for every driver ----
export function open(race) {
  const s = scores(race), f = race.finish, n = Math.max(20, ...f.map((d) => Math.max(d.pred, d.actual, Math.min(d.grid, 22))));
  const x = (p) => `${((Math.min(p, n) - 1) / (n - 1)) * 100}%`;
  const head = `<div class="rp-head"><b>${esc(race.name)} ${race.season}</b><span>${esc(race.circuit.replace(/_/g, " "))} · ${race.laps} laps</span></div>`;
  const sum = `<div class="rp-open__sum"><span>Rank correlation <b class="rp-mono">${s.model.toFixed(2)}</b> model, <b class="rp-mono">${s.grid.toFixed(2)}</b> grid</span>`
    + `<span class="rp-key"><i class="rp-dot rp-dot--grid"></i>Grid</span><span class="rp-key"><i class="rp-dot rp-dot--ring"></i>Predicted</span><span class="rp-key"><i class="rp-dot"></i>Result</span></div>`;
  const body = f.map((d) => {
    const g = d.pred - d.actual, lo = Math.min(d.pred, d.actual), hi = Math.max(d.pred, d.actual);
    return `<div class="rp-row${d.finished && g >= 3 ? " is-gain" : ""}"><span class="rp-row__pos rp-mono">${d.finished ? `P${d.actual}` : "DNF"}</span>`
      + `<span class="rp-row__name"><span class="rp-row__full">${esc(d.name)}</span><span class="rp-row__short">${esc(d.code)}</span></span>`
      + `<span class="rp-track"><i class="rp-span" style="left:${x(lo)};width:calc(${x(hi)} - ${x(lo)})"></i><i class="rp-g" style="left:${x(d.grid > 20 ? n : d.grid)}"></i><i class="rp-p" style="left:${x(d.pred)}"></i>`
      + (d.finished ? `<i class="rp-a" style="left:${x(d.actual)}"></i>` : "") + `</span>`
      + `<span class="rp-row__d rp-mono">${d.finished ? signed(g) : ""}</span></div>`;
  }).join("");
  return head + sum + `<div class="rp-rows">${body}</div>`;
}

// ---- the four model fragments ----
const fragHead = (title, race) => `<div class="rp-head"><b>${title}</b><span class="rp-sp"></span><span>${esc(short(race.name))} ${race.season}</span></div>`;

export function fragSC(race) {
  const s = race.sc, head = fragHead("Safety car risk per lap", race);
  if (!s.lap.length) return head + `<p class="rp-empty">No lap-by-lap scores for this race.</p>`;
  const byLap = new Map(s.lap.map((l, i) => [l, s.p[i]])), max = Math.max(0.05, ...s.p);
  const bars = Array.from({ length: race.laps }, (_, i) => {
    const l = i + 1, p = byLap.get(l);
    if (p === undefined) return `<i class="is-out" style="height:100%"></i>`;
    return `<i${s.starts.includes(l) ? ' class="is-sc"' : ""} style="height:${Math.max(2, (p / max) * 100)}%"></i>`;
  }).join("");
  const avg = s.p.reduce((a, b) => a + b, 0) / s.p.length;
  const hit = s.starts.map((o) => byLap.get(o)).filter((v) => v !== undefined);
  const note = s.starts.length
    ? `${s.starts.length === 1 ? "One safety car" : `${s.starts.length} safety cars`} started (lap ${s.starts.join(", ")}). Risk on ${s.starts.length === 1 ? "that lap" : "those laps"}: ${pct(hit.reduce((a, b) => a + b, 0) / Math.max(hit.length, 1), 1)}, against ${pct(avg, 1)} for a lap.`
    : `No safety car. Risk averaged ${pct(avg, 1)} a lap.`;
  return head + `<div class="rp-bars">${bars}</div><div class="rp-axis rp-mono"><span>Lap 1</span><span>Lap ${race.laps}</span></div><p class="rp-note">${note}</p>`;
}

export function fragTyres(race) {
  const t = race.tyres || {}, head = fragHead("Tyre wear, seconds per lap of age", race);
  const comps = Object.keys(COMP).filter((c) => t[c]);
  if (!comps.length) return head + `<p class="rp-empty">No dry-tyre wear measured for this race.</p>`;
  const vals = comps.flatMap((c) => [t[c].measured - t[c].se, t[c].measured + t[c].se, t[c].pre ?? t[c].measured]);
  const lo = Math.min(0, ...vals), hi = Math.max(0.05, ...vals), x = (v) => `${((v - lo) / (hi - lo)) * 100}%`;
  return head + `<div class="rp-lines">` + comps.map((c) => {
    const d = t[c];
    return `<div class="rp-line"><span>${COMP[c]}</span><span class="rp-scale"><i class="rp-zero" style="left:${x(0)}"></i>`
      + `<i class="rp-se" style="left:${x(d.measured - d.se)};width:calc(${x(d.measured + d.se)} - ${x(d.measured - d.se)})"></i>`
      + (d.pre !== null ? `<i class="rp-p" style="left:${x(d.pre)}"></i>` : "") + `<i class="rp-a" style="left:${x(d.measured)}"></i></span>`
      + `<span class="rp-line__v rp-mono">${d.measured >= 0 ? "+" : "−"}${Math.abs(d.measured).toFixed(3)}</span></div>`;
  }).join("") + `</div><div class="rp-axis"><span class="rp-key"><i class="rp-dot rp-dot--ring"></i>Before the race</span><span class="rp-key"><i class="rp-dot"></i>Measured</span></div>`;
}

export function fragTeams(race) {
  const teams = race.teams || [], head = fragHead("Team wear against the field", race);
  if (!teams.length) return head + `<p class="rp-empty">No team wear measured for this race.</p>`;
  const m = Math.max(0.02, ...teams.map((d) => Math.abs(d.tw))), top = teams.reduce((a, b) => (Math.abs(b.tw) > Math.abs(a.tw) ? b : a));
  return head + `<div class="rp-lines">` + teams.map((d) => {
    const w = (Math.abs(d.tw) / m) * 50;
    return `<div class="rp-line"><span>${esc(d.team)}</span><span class="rp-div"><i${d === top ? ' class="is-top"' : ""} style="left:${d.tw < 0 ? 50 - w : 50}%;width:${Math.max(w, 0.5)}%"></i></span>`
      + `<span class="rp-line__v rp-mono">${d.tw >= 0 ? "+" : "−"}${Math.abs(d.tw).toFixed(3)}</span></div>`;
  }).join("") + `</div>`;
}

export function fragPits(race) {
  const rows = race.pit_window || [], head = fragHead("First stop: real and called", race);
  if (!rows.length) return head + `<p class="rp-empty">No replayed first stops for this race. The replay covers green-flag stops in dry races from 2023.</p>`;
  const x = (l) => `${((l - 1) / (race.laps - 1)) * 100}%`;
  return head + `<div class="rp-lines rp-pits">` + rows.map((d) => {
    const lo = Math.min(d.actual, d.assistant), hi = Math.max(d.actual, d.assistant);
    return `<div class="rp-line"><span class="rp-mono">${esc(d.code)}</span><span class="rp-scale">`
      + `<i class="rp-se" style="left:${x(lo)};width:calc(${x(hi)} - ${x(lo)})"></i><i class="rp-p" style="left:${x(d.assistant)}"></i><i class="rp-a" style="left:${x(d.actual)}"></i></span>`
      + `<span class="rp-line__v rp-mono">lap ${d.actual} · ${d.assistant}</span></div>`;
  }).join("") + `</div><div class="rp-axis"><span class="rp-key"><i class="rp-dot"></i>Real stop</span><span class="rp-key"><i class="rp-dot rp-dot--ring"></i>Assistant's call</span></div>`;
}

// ---- headline figures from data/metrics.json ----
export function metrics(D) {
  const s = D.summary, f2 = (v) => v.toFixed(2), f3 = (v) => v.toFixed(3);
  return {
    "pill": `${s.finish.races} races scored`,
    "m-auc": f2(s.sc_lap.auc), "m-pit": pct(s.assistant.pit_within_3),
    "laps": (s.data?.laps ?? 0).toLocaleString("en-GB"),
    "r-finish": `${f3(s.finish.model_spearman)}<small>${s.finish.races} races, 2021 on</small>`,
    "r-finish-b": `${f3(s.finish.grid_spearman)}<small>Grid order</small>`,
    "r-sc": `${f2(s.sc_lap.auc)}<small>Lap 1, restarts, retirements, rain</small>`,
    "r-sc-b": `${f2(s.sc_lap.lap1_baseline_auc)}<small>Lap 1 against the rest</small>`,
    "r-tyre": `${f3(s.tyre_in_race.mae_after_40)} s<small>Updated from the race's own laps</small>`,
    "r-tyre-b": `${f3(s.tyre_in_race.mae_pre_race)} s<small>Estimate before the start</small>`,
    "r-pit": `${pct(s.assistant.pit_within_3)}<small>Live assistant, 2024 on</small>`,
    "r-pit-b": `${pct(s.assistant.pre_race_within_3)}<small>Plan made before the race</small>`,
    "r-call": `${pct(s.sc_calls.model)}<small>No better than the baseline</small>`,
    "r-call-b": `${pct(s.sc_calls.always_pit)}<small>Always pit</small>`,
  };
}

// The hero (always the latest race) and the headline figures, by data-rp slot.
export function pageSlots(D) {
  const latest = D.races.at(-1);
  return { "hero-race": `${esc(latest.name)} ${latest.season}`, "hero-recent": heroRecent(D, latest), "hero-top": heroTop(latest), ...metrics(D) };
}

// Everything that follows the picked race, by data-rp slot.
export function raceSlots(D, race) {
  return {
    "crumb": String(race.season), "seasons": seasons(D, race.season), "seasons-phone": seasons(D, race.season, true),
    "rows": rows(D, race.season, key(race)), "open": open(race),
    "f-sc": fragSC(race), "f-tyres": fragTyres(race), "f-teams": fragTeams(race), "f-pits": fragPits(race),
  };
}
