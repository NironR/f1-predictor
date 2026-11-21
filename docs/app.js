// The race explorer: picking a season or a race redraws the open race and the four model cards.
import { key, pageSlots, raceSlots } from "./render.js";

const D = window.F1;
const fill = (slots) => {
  for (const [name, html] of Object.entries(slots)) {
    for (const el of document.querySelectorAll(`[data-rp="${name}"]`)) el.innerHTML = html;
  }
};

function pick(race, focusRow = false) {
  fill(raceSlots(D, race));
  if (focusRow) document.querySelector(`[data-rp="rows"] [data-key="${key(race)}"]`)?.focus();
}

function pickSeason(season) {
  const races = D.races.filter((r) => r.season === season);
  if (races.length) pick(races.at(-1));
}

if (D) {
  fill(pageSlots(D));
  document.querySelector('[data-rp="hero-label"]')
    ?.setAttribute("aria-label", `The latest race: ${D.races.at(-1).name} ${D.races.at(-1).season}, predicted order beside the result`);
  pick(D.races.at(-1));

  for (const name of ["seasons", "seasons-phone"]) {
    document.querySelector(`[data-rp="${name}"]`)?.addEventListener("click", (e) => {
      const b = e.target.closest("[data-season]");
      if (b) pickSeason(+b.dataset.season);
    });
  }
  const list = document.querySelector('[data-rp="rows"]');
  const choose = (row) => {
    const race = D.races.find((r) => key(r) === row.dataset.key);
    if (race) pick(race, true);
  };
  list?.addEventListener("click", (e) => { const row = e.target.closest("[data-key]"); if (row) choose(row); });
  list?.addEventListener("keydown", (e) => {
    const row = e.target.closest("[data-key]");
    if (!row) return;
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); choose(row); }
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      (e.key === "ArrowDown" ? row.nextElementSibling : row.previousElementSibling)?.focus();
    }
  });
}
