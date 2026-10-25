import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, test } from 'node:test';
import { recordedClient, recording, stubClient } from '../src/clients.js';
import { compare } from '../src/report.js';
import { loadRubric } from '../src/rubric.js';
import { scoreEssays } from '../src/score.js';

const rubric = loadRubric(new URL('../rubrics/ellipse.yaml', import.meta.url));
const { essays } = JSON.parse(readFileSync(new URL('../data/sample.json', import.meta.url), 'utf8'));
const oneItem = { ...rubric, items: rubric.items.slice(0, 1) };
const dir = mkdtempSync(join(tmpdir(), 'rubric-eval-'));
after(() => rmSync(dir, { recursive: true, force: true }));

test('every essay gets one score per item, with a reason', async () => {
  const results = await scoreEssays({ rubric, essays: essays.slice(0, 3), client: stubClient() });
  assert.equal(results.length, 18);
  for (const r of results) {
    assert.ok(Number.isInteger(r.score) && r.score >= 1 && r.score <= 10);
    assert.equal(r.sentences, 1);
  }
});

test('a recording replays to the same scores, and a changed prompt is caught', async () => {
  const path = join(dir, 'rec.jsonl');
  const live = await scoreEssays({ rubric, essays: essays.slice(0, 2), client: recording(stubClient(), path) });
  const replay = await scoreEssays({ rubric, essays: essays.slice(0, 2), client: recordedClient(path) });
  assert.deepEqual(replay, live);
  assert.ok(!readFileSync(path, 'utf8').includes(essays[0].text.slice(0, 40)), 'prompts are not stored');

  const edited = [{ ...essays[0], text: essays[0].text + ' Extra.' }];
  const [first] = await scoreEssays({ rubric, essays: edited, client: recordedClient(path) });
  assert.equal(first.score, null);
  assert.match(first.error, /no recorded reply/);
});

test('resuming reuses saved replies and only asks for the rest', async () => {
  const path = join(dir, 'resume.jsonl');
  let calls = 0;
  const counting = { name: 'stub', complete: async (req) => { calls++; return stubClient().complete(req); } };
  await scoreEssays({ rubric, essays: essays.slice(0, 1), client: recording(counting, path) });
  assert.equal(calls, 6);
  const results = await scoreEssays({ rubric, essays: essays.slice(0, 2), client: recording(counting, path, { resume: true }) });
  assert.equal(calls, 12);
  assert.equal(results.filter((r) => r.score != null).length, 12);
});

test('a bad reply is asked for again once, then recorded as an error', async () => {
  let calls = 0;
  const flaky = { name: 'flaky', complete: async () => ({ text: ++calls === 1 ? 'nonsense' : '{"score": 6, "reason": "Fine."}', model: 'm' }) };
  const [r] = await scoreEssays({ rubric: oneItem, essays: essays.slice(0, 1), client: flaky });
  assert.equal(r.score, 6);
  assert.equal(calls, 2);

  const broken = { name: 'broken', complete: async () => ({ text: '{"score": 42, "reason": "x"}', model: 'm' }) };
  const [e] = await scoreEssays({ rubric: oneItem, essays: essays.slice(0, 1), client: broken });
  assert.equal(e.score, null);
  assert.match(e.error, /42/);
});

test('the comparison doubles the human mean and leaves failed calls out', () => {
  const e = essays[0];
  const results = [
    { id: e.id, item: 'grammar', score: e.human.grammar * 2, reason: 'r.', sentences: 1 },
    { id: e.id, item: 'syntax', score: null, error: 'x' },
  ];
  const cmp = compare(results, [e]);
  assert.equal(cmp.errors, 1);
  assert.equal(cmp.ten.all.n, 1);
  assert.equal(cmp.ten.all.exact, 1);
});
