import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parse } from 'yaml';
import { validateRubric } from '../../rubric-eval/src/rubric.js';
import {
  blankForm,
  fromRubric,
  idFromName,
  moveItem,
  parseBand,
  parseText,
  splitScale,
  toJson,
  toRubric,
  validate,
} from '../src/rubric.js';

const ellipsePath = new URL('../../rubric-eval/rubrics/ellipse.yaml', import.meta.url);
const ellipseText = readFileSync(ellipsePath, 'utf8');
const ellipse = parse(ellipseText);

const fields = (form) => validate(form).map((p) => p.field.replace(/(item|band)\d+/g, '$1'));

test("rubric-eval's real rubric round-trips through the form unchanged", () => {
  const form = fromRubric(parseText(ellipseText));
  assert.deepEqual(validate(form), []);
  const exported = toJson(form);
  assert.deepStrictEqual(JSON.parse(exported), ellipse);
  // Same keys in the same order, so the file reads the same as well as parsing the same.
  assert.equal(exported, JSON.stringify(ellipse, null, 2) + '\n');
});

test('the exported JSON loads in rubric-eval as it stands', () => {
  const exported = toJson(fromRubric(ellipse));
  assert.doesNotThrow(() => validateRubric(parse(exported)));
});

test('importing the export rebuilds the same form', () => {
  const once = fromRubric(ellipse);
  const twice = fromRubric(parseText(toJson(once)));
  const shape = (f) => ({ ...toRubric(f) });
  assert.deepStrictEqual(shape(twice), shape(once));
  assert.equal(twice.items.length, 6);
  assert.deepEqual(
    twice.bands.map((b) => b.text),
    ['1-2', '3-4', '5-6', '7-8', '9-10'],
  );
});

test('an item with no guidance in a band is a problem', () => {
  const form = fromRubric(ellipse);
  form.items[2].guidance[form.bands[1].key] = '   ';
  const problems = validate(form);
  assert.equal(problems.length, 1);
  assert.equal(problems[0].message, 'Vocabulary has no guidance for 3-4.');
});

test('a scale with gaps or overlaps is a problem', () => {
  const form = fromRubric(ellipse);
  form.bands[1].text = '4';
  form.bands[3].text = '6-8';
  const messages = validate(form).map((p) => p.message);
  assert.ok(messages.includes('Score 6 is in two bands (5-6 and 6-8).'));
  assert.ok(messages.includes('No band covers score 3.'));
});

test('bands outside the scale, and a backwards scale, are problems', () => {
  const form = fromRubric(ellipse);
  form.max = '8';
  assert.ok(validate(form).some((p) => p.message === 'Score 9 is outside the scale 1-8.'));
  form.max = '1';
  assert.ok(validate(form).some((p) => p.field === 'scale'));
});

test('duplicate names are a problem, whatever the case', () => {
  const form = fromRubric(ellipse);
  form.items[4].name = ' SYNTAX ';
  assert.deepEqual(
    validate(form).map((p) => p.message),
    ['"SYNTAX" is also the name of item 2.'],
  );
});

test('ids must suit rubric-eval and be unique', () => {
  const form = fromRubric(ellipse);
  form.items[1].id = 'Syntax!';
  form.items[3].id = 'cohesion';
  assert.deepEqual(fields(form), ['item:item:id', 'item:item:id']);
});

test('the checks here agree with rubric-eval on every broken rubric', () => {
  const breaks = [
    (f) => (f.bands[1].text = '4'),
    (f) => (f.bands[2].text = '5-7'),
    (f) => (f.bands[4].text = '9-11'),
    (f) => (f.bands[0].text = 'low'),
    (f) => (f.max = '1'),
    (f) => (f.items[0].id = '1st'),
    (f) => (f.items[5].id = 'grammar'),
    (f) => (f.items[2].description = ''),
    (f) => (f.items = []),
  ];
  for (const breakIt of breaks) {
    const form = fromRubric(ellipse);
    breakIt(form);
    assert.ok(validate(form).length > 0, String(breakIt));
    assert.throws(() => validateRubric(toRubric(form)), String(breakIt));
  }
});

test('a new item gets its id from its name until the id is edited', () => {
  assert.equal(idFromName('Word choice'), 'word_choice');
  assert.equal(idFromName('Éclat & flow'), 'eclat_flow');
  assert.equal(idFromName('2nd draft'), 'item_2nd_draft');
  assert.equal(idFromName('!!!'), '');
});

test('bands read as a score or a range, and a scale splits evenly', () => {
  assert.deepEqual(parseBand('3 – 4'), { lo: 3, hi: 4 });
  assert.deepEqual(parseBand('7'), { lo: 7, hi: 7 });
  assert.equal(parseBand('4-3'), null);
  assert.equal(parseBand('a'), null);
  assert.deepEqual(splitScale(1, 10, 5), [
    { lo: 1, hi: 2 },
    { lo: 3, hi: 4 },
    { lo: 5, hi: 6 },
    { lo: 7, hi: 8 },
    { lo: 9, hi: 10 },
  ]);
  assert.deepEqual(
    splitScale(0, 6, 3).map((b) => `${b.lo}-${b.hi}`),
    ['0-1', '2-3', '4-6'],
  );
});

test('items move up and down, and stay put at the ends', () => {
  const form = fromRubric(ellipse);
  const ids = (f) => f.items.map((i) => i.id);
  assert.deepEqual(ids(moveItem(form, 0, 2)).slice(0, 3), ['syntax', 'vocabulary', 'cohesion']);
  assert.equal(moveItem(form, 0, -1), form);
  assert.equal(moveItem(form, 5, 6), form);
});

test('a rubric whose items use different bands is refused on import, with the reason', () => {
  const odd = structuredClone(ellipse);
  odd.items[1].bands = { '1-5': 'a', '6-10': 'b' };
  assert.throws(() => fromRubric(odd), /Every item needs the same bands\. "Syntax" has 1-5, 6-10/);
  assert.throws(() => parseText('- just\n- a list\n'), /does not hold a rubric/);
});

test('a blank form lists what it still needs', () => {
  assert.deepEqual(fields(blankForm()), [
    'name',
    'item:item:name',
    'item:item:id',
    'item:item:description',
    'item:item:band:band',
    'item:item:band:band',
    'item:item:band:band',
    'item:item:band:band',
  ]);
});
