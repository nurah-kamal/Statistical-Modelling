import { createHash } from 'node:crypto';
import { appendFileSync, existsSync, readFileSync } from 'node:fs';

const hashOf = (prompt) => createHash('sha256').update(prompt).digest('hex');

// Saves each reply under a hash of its prompt, so a run can be replayed later
// without a key. The prompt itself is not stored. With resume, a reply already
// saved is used once instead of asking again; a retry of it still goes live.
export function recording(client, path, { resume = false } = {}) {
  const saved = resume && existsSync(path) ? readReplies(path) : new Map();
  const used = new Set();
  return {
    name: client.name,
    model: client.model,
    async complete(req) {
      const hash = hashOf(req.prompt);
      if (saved.has(hash) && !used.has(hash)) {
        used.add(hash);
        return { text: saved.get(hash).text, model: saved.get(hash).model };
      }
      used.add(hash);
      const reply = await client.complete(req);
      appendFileSync(path, JSON.stringify({ hash, model: reply.model, text: reply.text }) + '\n');
      return reply;
    },
  };
}

function readReplies(path) {
  const replies = new Map();
  for (const line of readFileSync(path, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    const r = JSON.parse(line);
    replies.set(r.hash, r);
  }
  return replies;
}

export function recordedClient(path) {
  if (!existsSync(path)) throw new Error(`No recording at ${path}`);
  const replies = readReplies(path);
  const model = [...replies.values()][0]?.model;
  return {
    name: 'recorded',
    model,
    async complete({ prompt }) {
      const r = replies.get(hashOf(prompt));
      if (!r) throw new Error('no recorded reply for this prompt (the rubric, essay or prompt wording changed since recording)');
      return { text: r.text, model: r.model };
    },
  };
}

// No model at all: scores from surface counts, so the pipeline can run with no
// key and there is a crude baseline to beat.
export function stubClient() {
  return {
    name: 'stub',
    model: 'surface-counts',
    async complete({ prompt }) {
      const item = /rubric item only: (\w+)/.exec(prompt)[1].toLowerCase();
      const essay = prompt.slice(prompt.indexOf('"""\n') + 4, prompt.lastIndexOf('\n"""'));
      const { score, reason } = surfaceScore(item, essay);
      return { text: JSON.stringify({ score, reason }), model: 'surface-counts' };
    },
  };
}

export function surfaceScore(item, text) {
  const words = text.match(/[A-Za-z']+/g) ?? [];
  const sentences = text.split(/[.!?]+\s+/).filter((s) => s.trim());
  const perSentence = words.length / Math.max(sentences.length, 1);
  const first200 = words.slice(0, 200).map((w) => w.toLowerCase());
  const variety = new Set(first200).size / Math.max(first200.length, 1);
  const longWords = words.filter((w) => w.length >= 8).length / Math.max(words.length, 1);
  const slips = (text.match(/\b(i|dont|cant|wont|im|youre|thats|doesnt|isnt)\b/g) ?? []).length
    + sentences.filter((s) => /^[a-z]/.test(s.trim())).length;
  const slipRate = (slips / Math.max(words.length, 1)) * 100;
  const linkers = (text.match(/\b(however|therefore|for example|first|second|finally|in conclusion|also|because|although)\b/gi) ?? []).length;
  const paragraphs = text.split(/\n\s*\n/).filter((p) => p.trim()).length;

  const clamp = (x) => Math.max(1, Math.min(10, Math.round(x)));
  const table = {
    cohesion: [clamp(3 + paragraphs * 0.6 + linkers * 0.25), `${paragraphs} paragraphs and ${linkers} linking words.`],
    syntax: [clamp(2 + perSentence / 4), `${perSentence.toFixed(1)} words per sentence.`],
    vocabulary: [clamp(variety * 8 + longWords * 20), `${Math.round(variety * 100)}% different words in the first 200 and ${Math.round(longWords * 100)}% long words.`],
    phraseology: [clamp(variety * 9), `${Math.round(variety * 100)}% different words in the first 200.`],
    grammar: [clamp(8 - slipRate * 1.5), `${slipRate.toFixed(1)} surface slips per 100 words.`],
    conventions: [clamp(9 - slipRate * 2), `${slipRate.toFixed(1)} spelling, apostrophe and capital slips per 100 words.`],
  };
  const [score, reason] = table[item] ?? [5, 'No surface count for this item.'];
  return { score, reason };
}
