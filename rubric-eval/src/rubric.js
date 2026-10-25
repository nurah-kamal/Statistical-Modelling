import { readFileSync } from 'node:fs';
import { parse } from 'yaml';

// Loads a rubric and refuses one that would let a score fall outside every band.
export function loadRubric(path) {
  return validateRubric(parse(readFileSync(path, 'utf8')));
}

export function validateRubric(r) {
  const problems = [];
  const { min, max } = r?.scale ?? {};
  if (!Number.isInteger(min) || !Number.isInteger(max) || min >= max) problems.push('scale needs whole-number min and max, min below max');
  if (!Array.isArray(r?.items) || r.items.length === 0) problems.push('rubric has no items');

  const ids = new Set();
  for (const [i, item] of (r?.items ?? []).entries()) {
    const where = item?.id ?? `item ${i + 1}`;
    if (!item?.id || !/^[a-z][a-z0-9_]*$/.test(item.id)) problems.push(`${where}: id must be lowercase letters, digits or _`);
    if (ids.has(item?.id)) problems.push(`${where}: id used twice`);
    ids.add(item?.id);
    if (!item?.name || !item?.description) problems.push(`${where}: needs a name and a description`);

    const covered = new Set();
    for (const key of Object.keys(item?.bands ?? {})) {
      const m = /^(\d+)(?:-(\d+))?$/.exec(key);
      if (!m) { problems.push(`${where}: band "${key}" is not a number or a range like 3-4`); continue; }
      const lo = Number(m[1]);
      const hi = Number(m[2] ?? m[1]);
      for (let s = lo; s <= hi; s++) {
        if (covered.has(s)) problems.push(`${where}: score ${s} is in two bands`);
        covered.add(s);
      }
    }
    if (Number.isInteger(min) && Number.isInteger(max)) {
      for (let s = min; s <= max; s++) if (!covered.has(s)) problems.push(`${where}: no band covers score ${s}`);
      for (const s of covered) if (s < min || s > max) problems.push(`${where}: band score ${s} is outside the scale`);
    }
  }

  if (problems.length) throw new Error(`Rubric is not usable:\n- ${problems.join('\n- ')}`);
  return r;
}
