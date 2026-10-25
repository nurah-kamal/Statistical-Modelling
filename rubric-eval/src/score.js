import { buildPrompt, parseReply, replySchema } from './prompt.js';

// Scores every essay on every item, one request each. A reply that breaks the
// shape is asked for once more, then recorded as an error rather than guessed.
export async function scoreEssays({ rubric, essays, client, onResult = () => {} }) {
  const results = [];
  for (const essay of essays) {
    for (const item of rubric.items) {
      const prompt = buildPrompt(rubric, item, essay);
      let result;
      for (let attempt = 1; attempt <= 2 && !result?.score; attempt++) {
        try {
          const reply = await client.complete({ prompt, schema: replySchema });
          result = { id: essay.id, item: item.id, ...parseReply(reply.text, rubric.scale), model: reply.model };
        } catch (e) {
          if (e.stop) throw e;
          result = { id: essay.id, item: item.id, score: null, error: e.message };
          if (client.name === 'recorded') break;
        }
      }
      results.push(result);
      onResult(result);
    }
  }
  return results;
}
