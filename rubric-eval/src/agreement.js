// How closely one set of scores follows another. Every function takes pairs
// of { a, b } on the same whole-number scale; a is usually the model, b the people.

export function stats(pairs, scale) {
  const n = pairs.length;
  if (n === 0) return { n: 0, exact: null, within1: null, mad: null, bias: null, r: null, qwk: null };
  const diffs = pairs.map((p) => p.a - p.b);
  return {
    n,
    exact: diffs.filter((d) => d === 0).length / n,
    within1: diffs.filter((d) => Math.abs(d) <= 1).length / n,
    mad: diffs.reduce((s, d) => s + Math.abs(d), 0) / n,
    bias: diffs.reduce((s, d) => s + d, 0) / n,
    r: pearson(pairs),
    qwk: qwk(pairs, scale),
  };
}

// Pearson correlation: whether the two sets rank and space the essays alike,
// ignoring any constant offset between them.
export function pearson(pairs) {
  const n = pairs.length;
  const ma = pairs.reduce((s, p) => s + p.a, 0) / n;
  const mb = pairs.reduce((s, p) => s + p.b, 0) / n;
  let cov = 0;
  let va = 0;
  let vb = 0;
  for (const { a, b } of pairs) {
    cov += (a - ma) * (b - mb);
    va += (a - ma) ** 2;
    vb += (b - mb) ** 2;
  }
  return va === 0 || vb === 0 ? null : cov / Math.sqrt(va * vb);
}

// Quadratic weighted kappa: 1 is perfect agreement, 0 is what chance gives,
// and a miss by two points costs four times a miss by one.
export function qwk(pairs, { min, max }) {
  const k = max - min + 1;
  const observed = Array.from({ length: k }, () => new Array(k).fill(0));
  const rowTotals = new Array(k).fill(0);
  const colTotals = new Array(k).fill(0);
  for (const { a, b } of pairs) {
    observed[a - min][b - min]++;
    rowTotals[a - min]++;
    colTotals[b - min]++;
  }
  const n = pairs.length;
  let num = 0;
  let den = 0;
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      const w = (i - j) ** 2 / (k - 1) ** 2;
      num += w * observed[i][j];
      den += (w * rowTotals[i] * colTotals[j]) / n;
    }
  }
  return den === 0 ? null : 1 - num / den;
}

export function byGroup(pairs, key, scale) {
  const groups = Map.groupBy(pairs, (p) => p[key]);
  return Object.fromEntries([...groups].map(([name, ps]) => [name, stats(ps, scale)]));
}

// The pairs furthest apart, largest gap first; ties keep their input order.
export function divergences(pairs, count = 5) {
  return pairs
    .map((p, i) => ({ ...p, gap: p.a - p.b, i }))
    .sort((x, y) => Math.abs(y.gap) - Math.abs(x.gap) || x.i - y.i)
    .slice(0, count)
    .map(({ i, ...p }) => p);
}

// A 1-10 score back onto a 1-5 rater's scale: 1-2 is level 1, 9-10 is level 5.
export const toLevel = (score) => Math.ceil(score / 2);
