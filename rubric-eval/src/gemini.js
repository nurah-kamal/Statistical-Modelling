const BASE = 'https://generativelanguage.googleapis.com/v1beta/models';
export const DEFAULT_MODEL = 'gemini-3.6-flash';

// The free tier allows a handful of requests a minute, so calls are spaced out
// and a 429 or 503 is retried after the delay the API asks for.
export function geminiClient({
  apiKey,
  model = DEFAULT_MODEL,
  minIntervalMs = 6500,
  maxRetries = 5,
  timeoutMs = 90_000,
  onRetry = () => {},
  maxWaitMs = 10 * 60_000,
  fetchImpl = fetch,
  sleep = (ms) => new Promise((r) => setTimeout(r, ms)),
} = {}) {
  if (!apiKey) throw new Error('GEMINI_API_KEY is not set. Put it in .env (see .env.example).');
  let last = 0;

  async function complete({ prompt, schema }) {
    const body = {
      contents: [{ role: 'user', parts: [{ text: prompt }] }],
      generationConfig: { temperature: 0, responseMimeType: 'application/json', responseSchema: schema },
    };
    for (let attempt = 0; ; attempt++) {
      const wait = last + minIntervalMs - Date.now();
      if (wait > 0) await sleep(wait);
      last = Date.now();

      let res;
      try {
        res = await fetchImpl(`${BASE}/${encodeURIComponent(model)}:generateContent`, {
          method: 'POST',
          headers: { 'content-type': 'application/json', 'x-goog-api-key': apiKey },
          body: JSON.stringify(body),
          signal: AbortSignal.timeout(timeoutMs),
        });
      } catch (e) {
        if (attempt < maxRetries) {
          onRetry({ status: e.name === 'TimeoutError' ? 'timed out' : 'network error', waitMs: 2 ** attempt * 5000 });
          await sleep(2 ** attempt * 5000);
          continue;
        }
        throw new Error(`Gemini request failed: ${e.name === 'TimeoutError' ? 'timed out' : 'network error'}`);
      }
      const json = await res.json().catch(() => ({}));

      if (res.ok) {
        const parts = json.candidates?.[0]?.content?.parts ?? [];
        const text = parts.filter((p) => !p.thought).map((p) => p.text ?? '').join('');
        if (!text) throw new Error(`empty reply (finish reason ${json.candidates?.[0]?.finishReason ?? 'unknown'})`);
        return { text, model: json.modelVersion ?? model };
      }
      // Only the status goes into the error: the API's message can echo the request.
      const status = `${res.status} ${json.error?.status ?? ''}`.trim();
      if ((res.status === 429 || res.status >= 500) && attempt < maxRetries) {
        const ms = retryDelay(json) ?? 2 ** attempt * 5000;
        if (ms > maxWaitMs) {
          const err = new Error(`Gemini quota for ${model} is used up; it resets in about ${Math.ceil(ms / 3_600_000)} hours. Run again with --resume then.`);
          err.stop = true;
          throw err;
        }
        onRetry({ status, waitMs: ms });
        await sleep(ms);
        continue;
      }
      throw new Error(`Gemini request failed: ${status}`);
    }
  }

  return { name: 'gemini', model, complete };
}

function retryDelay(json) {
  const info = json.error?.details?.find((d) => d['@type']?.endsWith('RetryInfo'));
  const m = /^(\d+(?:\.\d+)?)s$/.exec(info?.retryDelay ?? '');
  return m ? Math.ceil(Number(m[1]) * 1000) + 500 : null;
}
