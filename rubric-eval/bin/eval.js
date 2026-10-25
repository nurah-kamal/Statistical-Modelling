// npm run eval -- [--client stub|recorded|gemini] [--limit N] [--record [--resume]] [--label NAME]
//
// stub      no model; surface counts only (the default, needs nothing)
// recorded  replays recordings/gemini.jsonl, so the published run can be checked without a key
// gemini    calls the API with GEMINI_API_KEY from .env; --record saves the replies,
//           and --resume keeps the ones already saved, so a stopped run carries on

import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { recordedClient, recording, stubClient } from '../src/clients.js';
import { DEFAULT_MODEL, geminiClient } from '../src/gemini.js';
import { compare, renderReport } from '../src/report.js';
import { loadRubric } from '../src/rubric.js';
import { scoreEssays } from '../src/score.js';

const root = new URL('../', import.meta.url);
const at = (p) => new URL(p, root);

const { values: opts } = parseArgs({
  options: {
    client: { type: 'string', default: 'stub' },
    rubric: { type: 'string', default: 'rubrics/ellipse.yaml' },
    data: { type: 'string', default: 'data/sample.json' },
    limit: { type: 'string' },
    record: { type: 'boolean', default: false },
    resume: { type: 'boolean', default: false },
    label: { type: 'string' },
    model: { type: 'string' },
  },
});

const rubric = loadRubric(at(opts.rubric));
let { essays } = JSON.parse(readFileSync(at(opts.data), 'utf8'));
if (opts.limit) essays = essays.slice(0, Number(opts.limit));

if (existsSync(at('.env'))) process.loadEnvFile(at('.env'));
const model = opts.model || process.env.GEMINI_MODEL || DEFAULT_MODEL;
const recordingPath = at(`recordings/${model}.jsonl`);
let client;
if (opts.client === 'stub') client = stubClient();
else if (opts.client === 'recorded') client = recordedClient(recordingPath);
else if (opts.client === 'gemini') {
  client = geminiClient({
    apiKey: process.env.GEMINI_API_KEY,
    model,
    minIntervalMs: Number(process.env.GEMINI_MIN_INTERVAL_MS || 6500),
    onRetry: ({ status, waitMs }) => console.error(`\n  ${status}, waiting ${Math.round(waitMs / 1000)}s`),
  });
  if (opts.record) {
    mkdirSync(at('recordings'), { recursive: true });
    if (!opts.resume) rmSync(recordingPath, { force: true });
    client = recording(client, recordingPath, { resume: opts.resume });
  }
} else {
  console.error(`Unknown client "${opts.client}". Use stub, recorded or gemini.`);
  process.exit(1);
}

const total = essays.length * rubric.items.length;
let done = 0;
const results = await scoreEssays({
  rubric,
  essays,
  client,
  onResult: (r) => {
    done++;
    console.log(`${done}/${total}  ${r.id} ${r.item}: ${r.score ?? r.error}`);
  },
}).catch((e) => {
  if (!e.stop) throw e;
  console.error(`Stopped after ${done} of ${total}: ${e.message}`);
  process.exit(2);
});

const label = opts.label ?? opts.client;
const usedModel = results.find((r) => r.model)?.model ?? client.model;
const cmp = compare(results, essays);
mkdirSync(at('results'), { recursive: true });
writeFileSync(at(`results/${label}.json`), JSON.stringify({ model: usedModel, rubric: opts.rubric, data: opts.data, results }, null, 2) + '\n');
writeFileSync(at(`results/${label}.md`), renderReport({ label, model: usedModel, rubric, essays, cmp }));

const a = cmp.ten.all;
if (a.n === 0) {
  console.error(`No scores: all ${cmp.calls} calls failed.`);
  process.exitCode = 1;
} else {
  console.log(`${usedModel}: exact ${(a.exact * 100).toFixed(0)}%, within 1 ${(a.within1 * 100).toFixed(0)}%, mean gap ${a.mad.toFixed(2)}, QWK ${a.qwk?.toFixed(2) ?? 'n/a'}`);
}
console.log(`Written: results/${label}.md and results/${label}.json`);
