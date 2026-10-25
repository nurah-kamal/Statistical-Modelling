// Builds data/sample.json from the ELLIPSE corpus. The corpus is not committed;
// download the two files listed in data/README.md and pass their paths:
//   node scripts/make-sample.js ELLIPSE_Final_github_train.csv ellipsis_raw_rater_scores_anon_all_essay.csv

import { readFileSync, writeFileSync } from 'node:fs';
import { parseCsv } from '../src/csv.js';

const ITEMS = ['Cohesion', 'Syntax', 'Vocabulary', 'Phraseology', 'Grammar', 'Conventions'];
const PER_BAND = 6;
const SEED = 20261025;

// Placeholders the corpus uses where a name or place was removed. An essay
// that needed one is left out, in case a name slipped past the clean-up.
const PLACEHOLDER = /Generic_|_NAME|PROEPR/i;

const [trainPath, rawPath] = process.argv.slice(2);
if (!trainPath || !rawPath) {
  console.error('usage: node scripts/make-sample.js <train.csv> <raw rater scores.csv>');
  process.exit(1);
}

const train = parseCsv(readFileSync(trainPath, 'utf8'));
const raters = new Map(parseCsv(readFileSync(rawPath, 'utf8')).map((r) => [r.text_id_kaggle, r]));

function mulberry32(a) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = mulberry32(SEED);

function band(overall) {
  const o = Number(overall);
  if (o <= 2) return 'low';
  if (o >= 4) return 'high';
  return String(o);
}

const eligible = train.filter((r) => {
  const words = Number(r.num_words);
  const raw = raters.get(r.text_id_kaggle);
  return words >= 150 && words <= 450
    && !PLACEHOLDER.test(r.full_text)
    && raw && raw.Identifying_Info_1 === '0' && raw.Identifying_Info_2 === '0';
});

const byBand = Map.groupBy(eligible, (r) => band(r.Overall));
const picked = [];
for (const name of ['low', '2.5', '3', '3.5', 'high']) {
  const pool = [...byBand.get(name)];
  for (let i = 0; i < PER_BAND; i++) picked.push(pool.splice(Math.floor(rand() * pool.length), 1)[0]);
}

const sample = picked.map((r) => {
  const raw = raters.get(r.text_id_kaggle);
  const scores = (suffix) => Object.fromEntries(ITEMS.map((k) => [k.toLowerCase(), Number(raw[`${k}_${suffix}`])]));
  return {
    id: r.text_id_kaggle,
    task: r.prompt,
    text: r.full_text.replace(/\r\n/g, '\n').trim(),
    human: Object.fromEntries(ITEMS.map((k) => [k.toLowerCase(), Number(r[k])])),
    raters: [scores(1), scores(2)],
  };
});

writeFileSync(new URL('../data/sample.json', import.meta.url), JSON.stringify({
  source: 'ELLIPSE corpus, ELLIPSE_Final_github_train.csv and ellipsis_raw_rater_scores_anon_all_essay.csv',
  licence: 'CC BY-NC-SA 4.0',
  seed: SEED,
  eligible: eligible.length,
  essays: sample,
}, null, 2) + '\n');
console.log(`${eligible.length} eligible essays, ${sample.length} sampled.`);
