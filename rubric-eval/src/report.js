import { byGroup, divergences, stats, toLevel } from './agreement.js';

const TEN = { min: 1, max: 10 };
const SPELLING = /spell|capitali[sz]|punctuat|apostrophe/i;
const FIVE = { min: 1, max: 5 };

// Pairs the model's scores with the people's. The corpus scores 1-5 in half
// points (the mean of two raters), so doubling puts it on the 1-10 scale.
export function compare(results, essays) {
  const byId = new Map(essays.map((e) => [e.id, e]));
  const scored = results.filter((r) => r.score != null);
  const ten = scored.map((r) => ({ id: r.id, item: r.item, a: r.score, b: byId.get(r.id).human[r.item] * 2, reason: r.reason }));

  const level = (pick) => scored.map((r) => ({ item: r.item, a: toLevel(r.score), b: byId.get(r.id).raters[pick][r.item] }));
  const raters = essays.flatMap((e) => Object.keys(e.human).map((item) => ({ item, a: e.raters[0][item], b: e.raters[1][item] })));
  const items = new Set(scored.map((r) => r.item));
  const ratersHere = raters.filter((p) => items.has(p.item));

  // The same scores with the average gap taken off, to separate ranking from
  // calibration. The shift is worked out on these essays, so it flatters.
  const all = stats(ten, TEN);
  const shift = all.n ? -Math.round(all.bias) : 0;
  const shifted = ten.map((p) => ({ ...p, a: Math.min(TEN.max, Math.max(TEN.min, p.a + shift)) }));
  const count = (list, key) => Object.fromEntries([...Map.groupBy(list, (p) => p[key])].map(([k, v]) => [k, v.length]).sort((x, y) => x[0] - y[0]));

  return {
    calls: results.length,
    errors: results.filter((r) => r.score == null).length,
    longReasons: scored.filter((r) => r.sentences > 1).length,
    ten: { all, byItem: byGroup(ten, 'item', TEN), divergences: divergences(ten, 8) },
    shifted: { by: shift, all: stats(shifted, TEN) },
    spread: { model: count(ten, 'a'), human: count(ten, 'b') },
    spelling: Object.fromEntries([...items].map((item) => {
      const reasons = scored.filter((r) => r.item === item);
      return [item, { mention: reasons.filter((r) => SPELLING.test(r.reason ?? '')).length, of: reasons.length }];
    })),
    levels: {
      modelVsRater1: stats(level(0), FIVE),
      modelVsRater2: stats(level(1), FIVE),
      rater1VsRater2: stats(ratersHere, FIVE),
      byItem: Object.fromEntries([...items].map((item) => [item, {
        model: stats([...level(0), ...level(1)].filter((p) => p.item === item), FIVE),
        raters: stats(ratersHere.filter((p) => p.item === item), FIVE),
      }])),
    },
  };
}

const pct = (x) => (x == null ? 'n/a' : `${Math.round(x * 100)}%`);
const num = (x, d = 2) => (x == null ? 'n/a' : x.toFixed(d));
const signed = (x) => (x == null ? 'n/a' : `${x > 0 ? '+' : ''}${x.toFixed(2)}`);

export function renderReport({ label, model, rubric, essays, cmp }) {
  const t = cmp.ten;
  const l = cmp.levels;
  const scores = Array.from({ length: 10 }, (_, i) => i + 1);
  const lines = [
    `# ${model}`,
    '',
    `${essays.length} essays × ${rubric.items.length} items = ${cmp.calls} calls. ${cmp.errors} failed after a retry. ${cmp.longReasons} reasons ran past one sentence.`,
    '',
    '## Against the human score, on the 1-10 scale',
    '',
    'The human score is the mean of two raters, doubled.',
    '',
    '| Item | Exact | Within 1 | Mean gap | Bias | r | QWK |',
    '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ...Object.entries(t.byItem).map(([item, s]) => `| ${item} | ${pct(s.exact)} | ${pct(s.within1)} | ${num(s.mad)} | ${signed(s.bias)} | ${num(s.r)} | ${num(s.qwk)} |`),
    `| **all** | **${pct(t.all.exact)}** | **${pct(t.all.within1)}** | **${num(t.all.mad)}** | **${signed(t.all.bias)}** | **${num(t.all.r)}** | **${num(t.all.qwk)}** |`,
    '',
    'Bias above zero means the model scores higher than the people. r is the correlation, which ignores a constant offset; QWK is quadratic weighted kappa, which does not.',
    '',
    '## How the scores spread',
    '',
    `| Score | ${scores.join(' | ')} |`,
    `| --- | ${scores.map(() => '---:').join(' | ')} |`,
    `| Model | ${scores.map((s) => cmp.spread.model[s] ?? 0).join(' | ')} |`,
    `| Human | ${scores.map((s) => cmp.spread.human[s] ?? 0).join(' | ')} |`,
    '',
    `## With the average gap taken off`,
    '',
    `Adding ${cmp.shifted.by} to every score (capped at ${TEN.max}): exact ${pct(cmp.shifted.all.exact)}, within 1 ${pct(cmp.shifted.all.within1)}, mean gap ${num(cmp.shifted.all.mad)}, QWK ${num(cmp.shifted.all.qwk)}. The shift is worked out on these same essays, so this is the best case, not a result.`,
    '',
    '## Reasons that mention spelling, capitals or punctuation',
    '',
    '| Item | Reasons |',
    '| --- | ---: |',
    ...Object.entries(cmp.spelling).map(([item, s]) => `| ${item} | ${s.mention} of ${s.of} |`),
    '',
    '## Against each rater, on the raters\' own 1-5 scale',
    '',
    'The model\'s 1-10 score is put back into its band (1-2 is level 1, 9-10 is level 5) and compared with each rater on their own. The last row is the two raters compared with each other, which is the bar to clear.',
    '',
    '| Comparison | Exact | Within 1 | r | QWK |',
    '| --- | ---: | ---: | ---: | ---: |',
    ...[['Model vs rater 1', l.modelVsRater1], ['Model vs rater 2', l.modelVsRater2], ['Rater 1 vs rater 2', l.rater1VsRater2]]
      .map(([name, s]) => `| ${name} | ${pct(s.exact)} | ${pct(s.within1)} | ${num(s.r)} | ${num(s.qwk)} |`),
    '',
    '| Item | Model vs a rater, exact | Raters vs each other, exact | Model QWK | Raters QWK |',
    '| --- | ---: | ---: | ---: | ---: |',
    ...Object.entries(l.byItem).map(([item, s]) => `| ${item} | ${pct(s.model.exact)} | ${pct(s.raters.exact)} | ${num(s.model.qwk)} | ${num(s.raters.qwk)} |`),
    '',
    '## Where the model and the people disagree most',
    '',
    '| Essay | Item | Model | Human | Gap | The model\'s reason |',
    '| --- | --- | ---: | ---: | ---: | --- |',
    ...t.divergences.map((d) => `| ${d.id} | ${d.item} | ${d.a} | ${d.b} | ${d.gap > 0 ? '+' : ''}${d.gap} | ${(d.reason ?? '').replace(/\|/g, '\\|')} |`),
    '',
  ];
  return lines.join('\n');
}
