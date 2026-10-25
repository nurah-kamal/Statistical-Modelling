import assert from 'node:assert/strict';
import { test } from 'node:test';
import { loadRubric, validateRubric } from '../src/rubric.js';

const item = (bands) => ({ id: 'x', name: 'X', description: 'd', bands });
const rubric = (bands) => ({ scale: { min: 1, max: 4 }, items: [item(bands)] });

test('the shipped rubric loads, with six items each covering 1-10', () => {
  const r = loadRubric(new URL('../rubrics/ellipse.yaml', import.meta.url));
  assert.deepEqual(r.items.map((i) => i.id), ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']);
  for (const i of r.items) assert.equal(Object.keys(i.bands).length, 5);
});

test('a rubric whose bands cover the scale once is accepted', () => {
  assert.doesNotThrow(() => validateRubric(rubric({ '1-2': 'a', 3: 'b', 4: 'c' })));
});

test('a gap in the bands is refused', () => {
  assert.throws(() => validateRubric(rubric({ '1-2': 'a', 4: 'c' })), /no band covers score 3/);
});

test('overlapping bands are refused', () => {
  assert.throws(() => validateRubric(rubric({ '1-3': 'a', '3-4': 'b' })), /score 3 is in two bands/);
});

test('a band outside the scale is refused', () => {
  assert.throws(() => validateRubric(rubric({ '1-4': 'a', 5: 'b' })), /outside the scale/);
});

test('duplicate item ids are refused', () => {
  const r = rubric({ '1-4': 'a' });
  r.items.push(r.items[0]);
  assert.throws(() => validateRubric(r), /id used twice/);
});
