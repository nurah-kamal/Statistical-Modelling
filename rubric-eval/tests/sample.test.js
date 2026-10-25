import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

const sample = JSON.parse(readFileSync(new URL('../data/sample.json', import.meta.url), 'utf8'));
const ITEMS = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions'];

test('30 different essays', () => {
  assert.equal(sample.essays.length, 30);
  assert.equal(new Set(sample.essays.map((e) => e.id)).size, 30);
});

test('only the essay, its task and its scores are kept', () => {
  for (const e of sample.essays) assert.deepEqual(Object.keys(e), ['id', 'task', 'text', 'human', 'raters']);
});

test('no essay holds a placeholder where a name was removed', () => {
  for (const e of sample.essays) assert.doesNotMatch(e.text, /Generic_|_NAME/i, e.id);
});

test('scores are on the corpus scale and the published score sits between the two raters', () => {
  for (const e of sample.essays) {
    for (const item of ITEMS) {
      const [r1, r2] = e.raters.map((r) => r[item]);
      const h = e.human[item];
      assert.ok([r1, r2].every((r) => Number.isInteger(r) && r >= 1 && r <= 5));
      assert.ok(h >= 1 && h <= 5 && h * 2 === Math.round(h * 2));
      assert.ok(h >= Math.min(r1, r2) && h <= Math.max(r1, r2), `${e.id} ${item}`);
      if (Math.abs(r1 - r2) <= 1) assert.equal(h, (r1 + r2) / 2, `${e.id} ${item}`);
    }
  }
});
