import assert from 'node:assert/strict';
import { test } from 'node:test';
import { divergences, pearson, qwk, stats, toLevel } from '../src/agreement.js';

const pairs = (list) => list.map(([a, b]) => ({ a, b }));
const s3 = { min: 1, max: 3 };

test('perfect agreement gives kappa 1, exact reversal gives -1', () => {
  assert.equal(qwk(pairs([[1, 1], [2, 2], [3, 3]]), s3), 1);
  assert.ok(Math.abs(qwk(pairs([[1, 3], [2, 2], [3, 1]]), s3) + 1) < 1e-12);
});

test('one near miss out of four gives kappa 0.8, worked by hand', () => {
  // observed weight 0.25; expected weight 5/4; 1 - 0.25 / 1.25
  assert.ok(Math.abs(qwk(pairs([[1, 1], [1, 2], [2, 2], [3, 3]]), s3) - 0.8) < 1e-12);
});

test('kappa is undefined when neither side varies', () => {
  assert.equal(qwk(pairs([[2, 2], [2, 2]]), s3), null);
});

test('exact, within-one, mean gap and bias are worked out by hand', () => {
  // gaps: 0, +1, -2, +3
  const s = stats(pairs([[5, 5], [6, 5], [3, 5], [8, 5]]), { min: 1, max: 10 });
  assert.equal(s.n, 4);
  assert.equal(s.exact, 0.25);
  assert.equal(s.within1, 0.5);
  assert.equal(s.mad, 1.5);
  assert.equal(s.bias, 0.5);
});

test('no pairs gives empty figures, not NaN', () => {
  assert.deepEqual(stats([], s3), { n: 0, exact: null, within1: null, mad: null, bias: null, r: null, qwk: null });
});

test('correlation ignores a constant offset; kappa does not', () => {
  const offset = pairs([[1, 3], [2, 4], [3, 5], [4, 6]]);
  assert.ok(Math.abs(pearson(offset) - 1) < 1e-12);
  assert.ok(qwk(offset, { min: 1, max: 6 }) < 1);
  assert.ok(Math.abs(pearson(pairs([[1, 3], [2, 2], [3, 1]])) + 1) < 1e-12);
  assert.equal(pearson(pairs([[2, 1], [2, 3]])), null);
});

test('divergences are the widest gaps either way, widest first', () => {
  const d = divergences([{ a: 5, b: 5, id: 'a' }, { a: 2, b: 8, id: 'b' }, { a: 9, b: 4, id: 'c' }, { a: 4, b: 9, id: 'd' }], 3);
  assert.deepEqual(d.map((x) => [x.id, x.gap]), [['b', -6], ['c', 5], ['d', -5]]);
});

test('a 1-10 score maps back to the five rater levels', () => {
  assert.deepEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(toLevel), [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]);
});
