// The rubric as the form holds it, and the conversions to and from the file
// rubric-eval reads. No React here, so the tests run in plain Node.
import { parse } from 'yaml';

let counter = 0;
const uid = (prefix) => `${prefix}${++counter}`;

export const ID_PATTERN = /^[a-z][a-z0-9_]*$/;

// An id rubric-eval accepts, made from a name: "Word choice" -> "word_choice".
export function idFromName(name) {
  const id = name
    .normalize('NFKD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
  return /^[a-z]/.test(id) ? id : id ? `item_${id}` : '';
}

// "3-4", "3–4" or "3" -> { lo, hi }, or null if it isn't a band.
export function parseBand(text) {
  const m = /^\s*(\d+)\s*(?:[-–]\s*(\d+))?\s*$/.exec(String(text));
  if (!m) return null;
  const lo = Number(m[1]);
  const hi = Number(m[2] ?? m[1]);
  return lo <= hi ? { lo, hi } : null;
}

export function bandKey({ lo, hi }) {
  return lo === hi ? String(lo) : `${lo}-${hi}`;
}

// Bands of equal width from min to max; the last one takes any remainder.
export function splitScale(min, max, count) {
  const size = max - min + 1;
  const n = Math.max(1, Math.min(count, size));
  const width = Math.floor(size / n);
  const bands = [];
  for (let i = 0, lo = min; i < n; i++) {
    const hi = i === n - 1 ? max : lo + width - 1;
    bands.push({ lo, hi });
    lo = hi + 1;
  }
  return bands;
}

export function newItem(bands, fields = {}) {
  return {
    key: uid('item'),
    id: fields.id ?? '',
    idEdited: fields.idEdited ?? false,
    name: fields.name ?? '',
    description: fields.description ?? '',
    guidance: Object.fromEntries(bands.map((b) => [b.key, fields.guidance?.[b.key] ?? ''])),
  };
}

export function newBand(text) {
  return { key: uid('band'), text };
}

export function blankForm() {
  const bands = splitScale(1, 4, 4).map((b) => newBand(bandKey(b)));
  return { name: '', context: '', min: '1', max: '4', bands, items: [newItem(bands)] };
}

// Reads a rubric file's text, JSON or YAML (JSON is valid YAML).
export function parseText(text) {
  const data = parse(text);
  if (!data || typeof data !== 'object' || Array.isArray(data)) {
    throw new Error('The file does not hold a rubric: expected name, scale and items.');
  }
  return data;
}

// A rubric object -> the form. The form sets bands once for the whole rubric,
// so every item must use the same bands.
export function fromRubric(rubric) {
  const items = Array.isArray(rubric.items) ? rubric.items : [];
  const first = Object.keys(items[0]?.bands ?? {});
  for (const item of items) {
    const keys = Object.keys(item?.bands ?? {});
    const same = keys.length === first.length && keys.every((k) => first.includes(k));
    if (!same) {
      throw new Error(
        `Every item needs the same bands. "${item?.name ?? item?.id}" has ${keys.join(', ') || 'none'}; the first item has ${first.join(', ')}.`,
      );
    }
  }
  const ordered = [...first].sort((a, b) => (parseBand(a)?.lo ?? 0) - (parseBand(b)?.lo ?? 0));
  const bands = ordered.map((text) => newBand(text));
  const byText = Object.fromEntries(bands.map((b) => [b.text, b.key]));
  return {
    name: String(rubric.name ?? ''),
    context: rubric.context == null ? '' : String(rubric.context),
    min: String(rubric.scale?.min ?? ''),
    max: String(rubric.scale?.max ?? ''),
    bands,
    items: items.map((item) =>
      newItem(bands, {
        id: String(item.id ?? ''),
        idEdited: true,
        name: String(item.name ?? ''),
        description: String(item.description ?? ''),
        guidance: Object.fromEntries(
          Object.entries(item.bands ?? {}).map(([k, v]) => [byText[k], v == null ? '' : String(v)]),
        ),
      }),
    ),
  };
}

// The form -> the rubric object rubric-eval reads, in its key order.
export function toRubric(form) {
  const rubric = { name: form.name.trim() };
  if (form.context.trim()) rubric.context = form.context.trim();
  rubric.scale = { min: Number(form.min), max: Number(form.max) };
  rubric.items = form.items.map((item) => ({
    id: item.id,
    name: item.name.trim(),
    description: item.description.trim(),
    bands: Object.fromEntries(
      form.bands.map((b) => {
        const parsed = parseBand(b.text);
        return [parsed ? bandKey(parsed) : b.text, item.guidance[b.key].trim()];
      }),
    ),
  }));
  return rubric;
}

export function toJson(form) {
  return JSON.stringify(toRubric(form), null, 2) + '\n';
}

// Everything that would stop rubric-eval loading the export, plus what makes a
// rubric unusable to a marker. Each problem names the field it belongs to.
export function validate(form) {
  const problems = [];
  const add = (field, message) => problems.push({ field, message });

  if (!form.name.trim()) add('name', 'Give the rubric a name.');

  const min = Number(form.min);
  const max = Number(form.max);
  const scaleOk = form.min.trim() !== '' && form.max.trim() !== '' && Number.isInteger(min) && Number.isInteger(max) && min < max;
  if (!scaleOk) add('scale', 'The scale needs whole numbers, with the lowest below the highest.');

  const covered = new Map();
  for (const band of form.bands) {
    const parsed = parseBand(band.text);
    if (!parsed) {
      add(`band:${band.key}`, `"${band.text}" isn't a band. Write a score like 3 or a range like 3-4.`);
      continue;
    }
    for (let s = parsed.lo; s <= parsed.hi; s++) {
      if (covered.has(s)) add(`band:${band.key}`, `Score ${s} is in two bands (${covered.get(s)} and ${band.text}).`);
      else covered.set(s, band.text);
      if (scaleOk && (s < min || s > max)) add(`band:${band.key}`, `Score ${s} is outside the scale ${min}-${max}.`);
    }
  }
  if (form.bands.length === 0) add('bands', 'Add at least one band.');
  if (scaleOk) {
    const gaps = [];
    for (let s = min; s <= max; s++) if (!covered.has(s)) gaps.push(s);
    if (gaps.length) add('bands', `No band covers ${gaps.length === 1 ? 'score' : 'scores'} ${listScores(gaps)}.`);
  }

  if (form.items.length === 0) add('items', 'Add at least one item.');
  const names = new Map();
  const ids = new Map();
  form.items.forEach((item, i) => {
    const label = item.name.trim() || `Item ${i + 1}`;
    const name = item.name.trim().toLowerCase();
    if (!name) add(`item:${item.key}:name`, `Item ${i + 1} needs a name.`);
    else if (names.has(name)) add(`item:${item.key}:name`, `"${item.name.trim()}" is also the name of item ${names.get(name)}.`);
    else names.set(name, i + 1);

    if (!ID_PATTERN.test(item.id)) {
      add(`item:${item.key}:id`, `${label}: the id must start with a letter and use only a-z, 0-9 and _.`);
    } else if (ids.has(item.id)) {
      add(`item:${item.key}:id`, `${label}: the id "${item.id}" is also used by item ${ids.get(item.id)}.`);
    } else {
      ids.set(item.id, i + 1);
    }

    if (!item.description.trim()) add(`item:${item.key}:description`, `${label} needs a description.`);
    for (const band of form.bands) {
      if (!item.guidance[band.key]?.trim()) {
        add(`item:${item.key}:band:${band.key}`, `${label} has no guidance for ${band.text || 'a band'}.`);
      }
    }
  });
  return problems;
}

function listScores(scores) {
  const runs = [];
  for (const s of scores) {
    const last = runs.at(-1);
    if (last && s === last[1] + 1) last[1] = s;
    else runs.push([s, s]);
  }
  return runs.map(([a, b]) => (a === b ? `${a}` : `${a}-${b}`)).join(', ');
}

export function moveItem(form, from, to) {
  if (to < 0 || to >= form.items.length || from === to) return form;
  const items = [...form.items];
  const [moved] = items.splice(from, 1);
  items.splice(to, 0, moved);
  return { ...form, items };
}
