import assert from 'node:assert/strict';
import { test } from 'node:test';
import { geminiClient } from '../src/gemini.js';

const KEY = 'test-key-not-real';
const ok = (body) => ({ ok: true, status: 200, json: async () => body });
const fail = (status, body) => ({ ok: false, status, json: async () => body });
const noWait = { minIntervalMs: 0, sleep: async () => {} };

test('the key goes in a header, never the URL, and thought parts are skipped', async () => {
  const seen = [];
  const client = geminiClient({
    apiKey: KEY, ...noWait,
    fetchImpl: async (url, init) => {
      seen.push({ url, init });
      return ok({ modelVersion: 'gemini-x', candidates: [{ content: { parts: [{ thought: true, text: 'thinking' }, { text: '{"score":4,"reason":"r."}' }] } }] });
    },
  });
  const reply = await client.complete({ prompt: 'p', schema: {} });
  assert.deepEqual(reply, { text: '{"score":4,"reason":"r."}', model: 'gemini-x' });
  assert.ok(!seen[0].url.includes(KEY));
  assert.equal(seen[0].init.headers['x-goog-api-key'], KEY);
  const body = JSON.parse(seen[0].init.body);
  assert.equal(body.generationConfig.temperature, 0);
  assert.equal(body.generationConfig.responseMimeType, 'application/json');
});

test('a 429 is retried after the delay the API asks for', async () => {
  const waits = [];
  let calls = 0;
  const client = geminiClient({
    apiKey: KEY, minIntervalMs: 0, sleep: async (ms) => waits.push(ms),
    fetchImpl: async () => (++calls === 1
      ? fail(429, { error: { status: 'RESOURCE_EXHAUSTED', details: [{ '@type': 'type.googleapis.com/google.rpc.RetryInfo', retryDelay: '12s' }] } })
      : ok({ candidates: [{ content: { parts: [{ text: '{}' }] } }] })),
  });
  await client.complete({ prompt: 'p', schema: {} });
  assert.equal(calls, 2);
  assert.deepEqual(waits, [12500]);
});

test('a used-up daily quota stops the run instead of waiting hours', async () => {
  const client = geminiClient({
    apiKey: KEY, ...noWait,
    fetchImpl: async () => fail(429, { error: { status: 'RESOURCE_EXHAUSTED', details: [{ '@type': 'type.googleapis.com/google.rpc.RetryInfo', retryDelay: '42986s' }] } }),
  });
  await assert.rejects(client.complete({ prompt: 'p', schema: {} }), (e) => {
    assert.equal(e.stop, true);
    assert.match(e.message, /used up; it resets in about 12 hours/);
    return true;
  });
});

test('an error names the status only, not the message the API echoes back', async () => {
  const client = geminiClient({
    apiKey: KEY, ...noWait,
    fetchImpl: async () => fail(400, { error: { status: 'INVALID_ARGUMENT', message: `bad key ${KEY} and the essay text` } }),
  });
  await assert.rejects(client.complete({ prompt: 'p', schema: {} }), (e) => {
    assert.equal(e.message, 'Gemini request failed: 400 INVALID_ARGUMENT');
    return true;
  });
});

test('retries stop after the limit', async () => {
  let calls = 0;
  const client = geminiClient({ apiKey: KEY, ...noWait, maxRetries: 2, fetchImpl: async () => { calls++; return fail(503, {}); } });
  await assert.rejects(client.complete({ prompt: 'p', schema: {} }), /503/);
  assert.equal(calls, 3);
});

test('a request that hangs is retried, then reported as timed out', async () => {
  let calls = 0;
  const client = geminiClient({
    apiKey: KEY, ...noWait, maxRetries: 1,
    fetchImpl: async () => { calls++; throw Object.assign(new Error('x'), { name: 'TimeoutError' }); },
  });
  await assert.rejects(client.complete({ prompt: 'p', schema: {} }), /timed out/);
  assert.equal(calls, 2);
});

test('no key, no client', () => {
  assert.throws(() => geminiClient({ apiKey: '' }), /GEMINI_API_KEY is not set/);
});
