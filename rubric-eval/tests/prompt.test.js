import assert from 'node:assert/strict';
import { test } from 'node:test';
import { buildPrompt, countSentences, parseReply } from '../src/prompt.js';
import { loadRubric } from '../src/rubric.js';

const rubric = loadRubric(new URL('../rubrics/ellipse.yaml', import.meta.url));
const scale = rubric.scale;

test('the prompt names one item, lists its bands and fences the essay', () => {
  const p = buildPrompt(rubric, rubric.items[1], { task: 'Being busy', text: 'I am busy.' });
  assert.match(p, /rubric item only: Syntax\./);
  assert.match(p, /9-10: Flexible, effective use/);
  assert.match(p, /The task was: "Being busy"/);
  assert.match(p, /"""\nI am busy\.\n"""$/);
  assert.doesNotMatch(p, /Cohesion/);
  assert.doesNotMatch(p, /\n\n\n/);
});

test('a well-formed reply is accepted', () => {
  assert.deepEqual(parseReply('{"score": 7, "reason": "Mostly  correct verbs."}', scale), { score: 7, reason: 'Mostly correct verbs.', sentences: 1 });
});

test('a reply in a code fence is still read', () => {
  assert.equal(parseReply('```json\n{"score": 3, "reason": "Many errors."}\n```', scale).score, 3);
});

test('scores off the scale or not whole are refused', () => {
  assert.throws(() => parseReply('{"score": 11, "reason": "x"}', scale), /not a whole number from 1 to 10/);
  assert.throws(() => parseReply('{"score": 0, "reason": "x"}', scale), /not a whole number/);
  assert.throws(() => parseReply('{"score": 6.5, "reason": "x"}', scale), /not a whole number/);
  assert.throws(() => parseReply('{"score": "7", "reason": "x"}', scale), /not a whole number/);
});

test('a missing reason or non-JSON reply is refused', () => {
  assert.throws(() => parseReply('{"score": 5}', scale), /reason is missing/);
  assert.throws(() => parseReply('Score: 5', scale), /not JSON/);
});

test('sentences are counted outside quotations only', () => {
  assert.equal(countSentences('Errors such as "i dont know. its bad." recur throughout.'), 1);
  assert.equal(countSentences('Verbs are weak. Tense shifts often.'), 2);
});
