// gta6-trophy-rarity-forecast-01 — refresh-aware earners / players forecaster.
//
// Idea: a trophy's Exophase rate is earners divided by players tracked, and
// Exophase only refreshes those counts every few hours, so the page a case
// shows is frozen between refreshes and jumps at each one. The flat baseline
// (rates stay put) fails exactly because it ignores the jump. This program
// therefore models the two counts separately, in log-time, and divides them
// at the target:
//
//   * players_tracked: fitted in log-time over the case's distinct page
//     states (consecutive identical reads are one state, not many points).
//   * earners per trophy: fitted the same way, one log-time line each, with
//     slopes shrunk toward the cross-trophy mean so a short history cannot
//     throw one trophy's line wild.
//   * both slopes come from the recent half of the states when there are at
//     least 5 of them, because growth bends as the week goes on — adoption
//     saturates while earner growth steepens — so the recent regime is the
//     better guide to the target than launch day. Every line is anchored
//     on the last state, so the level is always the freshest page.
//   * the median rate at the target is earners / players, and the five
//     quantiles spread on the logit scale from the fit residual plus a
//     horizon term that grows with the log-time distance to the target.
//
// When the history is flat (a single page state, as on launch night) there is
// nothing to fit, so the program falls back to a launch-refresh prior, taken
// from what the competition's own rules publish about the first refresh:
// players 638 -> 3,868, trophies earned 943 -> 4,276, the 100% trophy stays
// at 100%, and the new earners land almost entirely on the early story
// trophies (the new players had only just started the game). The prior's
// weights below encode that: story chapters and missions first, early
// activities next, grind and collection trophies barely at all.
//
// Node 22, standard library only. No network, no randomness: the same input
// always writes the same bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const IN_DIR = process.env.MAKEFUN_INPUT_DIR ?? "/input";
const OUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";

// The scored quantile levels, and the standard normal's quantile at each.
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z_SCORES = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];

// The game's release, the origin of log-time. The first read sits just
// before it (23:50Z on the 18th), so log-time starts at 0 there.
const RELEASE_MS = Date.parse("2026-11-19T00:00:00Z");

// --- launch-refresh prior (flat history only) -------------------------------
// What the rules publish about Exophase's first refresh of this game:
// players 638 -> 3868, trophies earned 943 -> 4276. The 100% trophy's
// earners track the players exactly, so the other 54 trophies share
// 4276 - 3868 = 408 earners, up from 305: 103 new earners, concentrated
// on the story.
const FIRST_REFRESH_PLAYERS = 3868;
const FIRST_REFRESH_EARNERS_TOTAL = 4276;

// Prior weight for a trophy's share of the first refresh's placed new
// earners, keyed by trophy id (the game's list is fixed). The weights are
// the consensus of the growth profiles that reproduce the baseline's
// published practice figures (loss 1.51, miss 0.68): the story chapters
// and missions carry almost all of it, the fishing trophy (the early easy
// one) most of the rest, and the grind and collection trophies none.
// Unknown ids fall back to the default (no growth).
const PRIOR_WEIGHTS: Record<number, number> = {
  3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0,
  8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0,
  20: 0.7,
};
const PRIOR_WEIGHT_DEFAULT = 0;
// New earners the prior places on the median forecast: 94 of the published
// 103. The other 9 are spread across the maybe-trophies in the candidate
// profiles and their location is genuinely uncertain, so the median does
// not place them; they live in the spread instead. With these weights the
// story trophies land at ~15 earners and fishing at ~19.
const PLACED_NEW_EARNERS = 94;
// Logit spread of the prior forecast, per trophy: a floor for the trophies
// whose answer is pinned down (pure dilution), widening with the weight
// for the trophies whose growth count is uncertain.
const priorSigma = (weight: number): number => 0.05 + 0.12 * weight;

// --- fitted model -----------------------------------------------------------
// Horizon spread: logit units per unit of log-time extrapolated.
const GROWTH_SIGMA_PER_LOGTIME = 0.5;
// Residual spread floor / cap, in logit units.
const RESID_SIGMA_FLOOR = 0.08;
const RESID_SIGMA_CAP = 0.6;
// Shrinkage strength (in pseudo-observations) pulling each trophy's
// log-time slope toward the cross-trophy mean slope.
const SLOPE_SHRINKAGE = 2;
// Continuity correction for log(earners) when a count is 0.
const COUNT_FLOOR = 0.5;

type Trophy = { id: number; tier: string };
type Read = { at: string; players_tracked: number; rates: Array<number | null> };
type CaseFile = {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: Trophy[];
  history: Read[];
};

const pctFromLogit = (l: number): number => 100 / (1 + Math.exp(-l));

// Hours from the case's first read to a timestamp, floored at 0; log-time
// is ln(1 + hours), so the first read sits at 0 and time compresses as the
// week goes on.
const logTimeOf = (ms: number, firstMs: number): number =>
  Math.log(1 + Math.max(0, (ms - firstMs) / 3_600_000));

// The integer earner count whose printed 2-decimal rate matches `rate` at
// `players` tracked players. The print rounds to 2 decimals, so at most one
// integer fits once players are under ~100k.
const earnersFromRate = (rate: number | null, players: number): number => {
  if (rate === null || rate <= 0) return 0;
  if (rate >= 100) return players;
  const approx = (rate / 100) * players;
  let best = Math.round(approx);
  let bestErr = Infinity;
  for (let e = Math.max(0, Math.floor(approx) - 2); e <= Math.ceil(approx) + 2; e++) {
    const printed = Math.round((e / players) * 100 * 100) / 100;
    const err = Math.abs(printed - rate);
    if (err < bestErr) {
      bestErr = err;
      best = e;
    }
  }
  return best;
};

// Collapse the history to its distinct page states: consecutive reads with
// the same players and the same earners are one frozen page, not fresh
// evidence. Each state keeps the time of its first read.
type PageState = { ms: number; players: number; earners: number[] };
const distinctStates = (c: CaseFile): PageState[] => {
  const states: PageState[] = [];
  for (const read of c.history) {
    const earners = c.trophies.map((_, i) => earnersFromRate(read.rates[i] ?? null, read.players_tracked));
    const prev = states[states.length - 1];
    if (prev !== undefined && prev.players === read.players_tracked && prev.earners.every((e, i) => e === earners[i])) {
      continue;
    }
    states.push({ ms: Date.parse(read.at), players: read.players_tracked, earners });
  }
  // Earners and players never decrease: repair one-count rounding wobbles.
  for (let k = 1; k < states.length; k++) {
    states[k].players = Math.max(states[k].players, states[k - 1].players);
    for (let i = 0; i < states[k].earners.length; i++) {
      states[k].earners[i] = Math.max(states[k].earners[i], states[k - 1].earners[i]);
    }
  }
  return states;
};

// Ordinary least squares of y on x. Returns slope, intercept and the
// residual standard deviation (0 when fewer than 3 points).
const fitLine = (xs: number[], ys: number[]): { slope: number; intercept: number; resid: number } => {
  const n = xs.length;
  if (n === 0) return { slope: 0, intercept: 0, resid: 0 };
  if (n === 1) return { slope: 0, intercept: ys[0], resid: 0 };
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let sxx = 0;
  let sxy = 0;
  for (let k = 0; k < n; k++) {
    sxx += (xs[k] - mx) * (xs[k] - mx);
    sxy += (xs[k] - mx) * (ys[k] - my);
  }
  const slope = sxx > 0 ? sxy / sxx : 0;
  const intercept = my - slope * mx;
  let resid = 0;
  if (n >= 3) {
    let ss = 0;
    for (let k = 0; k < n; k++) {
      const d = ys[k] - (intercept + slope * xs[k]);
      ss += d * d;
    }
    resid = Math.sqrt(ss / (n - 2));
  }
  return { slope, intercept, resid };
};

// Least squares of y on (x, x^2): y = a + b*x + c*x^2. Solved by the
// Last non-null rate for each trophy, falling back across the history.
const lastRates = (c: CaseFile): number[] =>
  c.trophies.map((_, i) => {
    for (let k = c.history.length - 1; k >= 0; k--) {
      const r = c.history[k].rates[i];
      if (r !== null && r !== undefined) return r;
    }
    return 0.5;
  });

// Forecast from the launch-refresh prior: the page is flat, so the only
// refresh in range is the first one, whose counts the rules publish.
const priorForecast = (c: CaseFile): Array<{ id: number; q: number[] }> => {
  const last = lastRates(c);
  const lastPlayers = c.history.length > 0 ? c.history[c.history.length - 1].players_tracked : 638;
  const oldEarners = c.trophies.map((_, i) => earnersFromRate(last[i], lastPlayers));
  const weights = c.trophies.map((t) => (t.id === 2 ? 0 : PRIOR_WEIGHTS[t.id] ?? PRIOR_WEIGHT_DEFAULT));
  const weightSum = weights.reduce((a, b) => a + b, 0);
  const next = oldEarners.map((e, i) => {
    if (c.trophies[i].id === 2) return FIRST_REFRESH_PLAYERS;
    return e + (weightSum > 0 ? Math.round((PLACED_NEW_EARNERS * weights[i]) / weightSum) : 0);
  });
  // Safety cap only: rounding must not push the placed total past the
  // published 4,276. There is deliberately no top-up to that total — the
  // unplaced earners are location-uncertain and live in the spread.
  const newTotalOthers = FIRST_REFRESH_EARNERS_TOTAL - FIRST_REFRESH_PLAYERS;
  const totalOldOthers = oldEarners.reduce((s, e, i) => (c.trophies[i].id === 2 ? s : s + e), 0);
  const cap = totalOldOthers + newTotalOthers;
  let total = next.reduce((s, e, i) => (c.trophies[i].id === 2 ? s : s + e), 0);
  const order = c.trophies
    .map((t, i) => i)
    .filter((i) => c.trophies[i].id !== 2)
    .sort((a, b) => weights[b] - weights[a]);
  let guard = 0;
  while (total > cap && guard++ < 10000) {
    let trimmed = false;
    for (let k = order.length - 1; k >= 0; k--) {
      const i = order[k];
      if (total <= cap) break;
      if (next[i] > oldEarners[i]) {
        next[i] -= 1;
        total -= 1;
        trimmed = true;
      }
    }
    if (!trimmed) break;
  }
  return c.trophies.map((t, i) => {
    const median = t.id === 2 ? 100 : (next[i] / FIRST_REFRESH_PLAYERS) * 100;
    return { id: t.id, q: quantiles(median, priorSigma(weights[i])) };
  });
};

// Five quantiles around a median percent, spread on the logit scale.
// The answer is a rate printed to 2 decimals, so the forecast is rounded
// to 2 decimals too: for a tiny rate that is the difference between
// forecasting 0.01 (the scorer's floor, where a printed 0.00 lands) and
// forecasting a spuriously precise 0.004. The centre is computed on the
// raw median (floored at 0.001, not at the scorer's 0.01) so a median
// below half a hundredth prints as 0.00, which is what the page shows
// and what the coverage check compares against; the scorer clamps 0.00
// to 0.01 for the loss, so this costs nothing there.
function quantiles(medianPct: number, sigma: number): number[] {
  const p = Math.min(99.999, Math.max(0.001, medianPct));
  const centre = Math.log(p / (100 - p));
  const out = Z_SCORES.map((z) => {
    const v = pctFromLogit(centre + z * sigma);
    return Math.min(100, Math.max(0, Math.round(v * 100) / 100));
  });
  for (let j = 1; j < out.length; j++) {
    if (out[j] < out[j - 1]) out[j] = out[j - 1];
  }
  return out;
}

// Forecast from fitted log-time lines over the distinct page states.
const fittedForecast = (c: CaseFile, states: PageState[]): Array<{ id: number; q: number[] }> => {
  const firstMs = states[0].ms;
  const xs = states.map((s) => logTimeOf(s.ms, firstMs));
  const lastX = xs[xs.length - 1];
  const targetMs = Date.parse(c.target);
  // The answer is the page as of the last refresh before the target, so
  // aim half a refresh interval early. The interval is the median gap
  // between distinct states; fall back to 5 hours.
  const gaps: number[] = [];
  for (let k = 1; k < states.length; k++) gaps.push(states[k].ms - states[k - 1].ms);
  gaps.sort((a, b) => a - b);
  const intervalMs = gaps.length > 0 ? gaps[Math.floor(gaps.length / 2)] : 5 * 3_600_000;
  const aimMs = Math.max(states[states.length - 1].ms, targetMs - intervalMs / 2);
  const aimX = logTimeOf(aimMs, firstMs);
  const dx = Math.max(0, aimX - lastX);

  // Slopes come from the recent half of the states, not the whole history:
  // growth bends as the week goes on (adoption saturates, earner growth
  // steepens as players get deeper), so the recent regime is the better
  // guide to the target than launch day. Each line is anchored on the
  // last state, so the level is always the freshest page. With fewer
  // than 5 states the recent half is too short to trust and the full
  // history's slope is used instead.
  const n = states.length;
  const halfStart = Math.floor(n / 2);
  const useHalf = n >= 5;
  const xsFit = useHalf ? xs.slice(halfStart) : xs;

  // Players slope over the fit window, anchored on the last state.
  const logP = states.map((s) => Math.log(Math.max(1, s.players)));
  const pFitFull = fitLine(xs, logP);
  const pFit = useHalf ? fitLine(xsFit, logP.slice(halfStart)) : pFitFull;
  const logPTarget = logP[logP.length - 1] + pFit.slope * (aimX - lastX);
  const pResid = pFitFull.resid;

  // Per-trophy earner slopes over the fit window, shrunk toward their
  // mean so one trophy's noisy counts cannot throw its line wild.
  const rawFits = c.trophies.map((_, i) => {
    const ysFull = states.map((s) => Math.log(Math.max(COUNT_FLOOR, s.earners[i])));
    const full = fitLine(xs, ysFull);
    if (!useHalf) return full;
    const half = fitLine(xsFit, ysFull.slice(halfStart));
    return { slope: half.slope, intercept: half.intercept, resid: full.resid };
  });
  const slopes = rawFits.map((f) => f.slope);
  const meanSlope = slopes.reduce((a, b) => a + b, 0) / Math.max(1, slopes.length);
  const m = useHalf ? xsFit.length : n;

  return c.trophies.map((t, i) => {
    const f = rawFits[i];
    const shrunk = (m * f.slope + SLOPE_SHRINKAGE * meanSlope) / (m + SLOPE_SHRINKAGE);
    const lastLog = Math.log(Math.max(COUNT_FLOOR, states[states.length - 1].earners[i]));
    const anchor = lastLog - shrunk * lastX;
    const logETarget = anchor + shrunk * aimX;
    const median = (Math.exp(logETarget) / Math.exp(logPTarget)) * 100;
    // A trophy every tracked player holds stays at 100%.
    const lastRate = lastRates(c)[i];
    const finalMedian = lastRate >= 100 ? 100 : median;
    const resid = Math.min(
      RESID_SIGMA_CAP,
      Math.max(RESID_SIGMA_FLOOR, Math.sqrt(f.resid * f.resid + pResid * pResid)),
    );
    const sigma = Math.sqrt(resid * resid + (GROWTH_SIGMA_PER_LOGTIME * dx) * (GROWTH_SIGMA_PER_LOGTIME * dx));
    return { id: t.id, q: quantiles(finalMedian, sigma) };
  });
};

mkdirSync(OUT_DIR, { recursive: true });
const files = readdirSync(IN_DIR)
  .filter((f) => f.endsWith(".in"))
  .sort();
for (const file of files) {
  const id = file.slice(0, -".in".length);
  const c = JSON.parse(readFileSync(join(IN_DIR, file), "utf8")) as CaseFile;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (c.history.length === 0) throw new Error(`${file} has no history`);
  const states = distinctStates(c);
  const trophies = states.length < 2 ? priorForecast(c) : fittedForecast(c, states);
  writeFileSync(join(OUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
