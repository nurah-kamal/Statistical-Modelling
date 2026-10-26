import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import ellipseText from '../../rubric-eval/rubrics/ellipse.yaml?raw';
import {
  bandKey,
  blankForm,
  fromRubric,
  idFromName,
  moveItem,
  newBand,
  newItem,
  parseBand,
  parseText,
  splitScale,
  toJson,
  validate,
} from './rubric.js';
import { sampleForm } from './sample.js';

const fieldId = (field) => `f-${field.replace(/:/g, '-')}`;

export default function App() {
  const [form, setForm] = useState(sampleForm);
  const [moved, setMoved] = useState(null);
  const [notice, setNotice] = useState(null);
  const [showProblems, setShowProblems] = useState(false);
  const [announcement, setAnnouncement] = useState('');
  const [previous, setPrevious] = useState(null);
  const problems = useMemo(() => validate(form), [form]);
  const byField = useMemo(() => {
    const map = new Map();
    for (const p of problems) if (!map.has(p.field)) map.set(p.field, p.message);
    return map;
  }, [problems]);
  const listRef = useRef(null);
  const positions = useRef(null);

  // Any edit clears the highlight left by the last move.
  const edit = (change) => {
    setMoved(null);
    setNotice(null);
    setForm((f) => change(f));
  };

  const setItem = (key, change) =>
    edit((f) => ({ ...f, items: f.items.map((it) => (it.key === key ? { ...it, ...change(it) } : it)) }));

  // Records where every item is, so the move can slide them from there.
  const remember = () => {
    positions.current = new Map(
      [...listRef.current.children].map((el) => [el.dataset.key, el.getBoundingClientRect().top]),
    );
  };

  const move = (from, to) => {
    if (to < 0 || to >= form.items.length || from === to) return;
    remember();
    const item = form.items[from];
    setForm((f) => moveItem(f, from, to));
    setMoved(item.key);
    setAnnouncement(`${item.name || `Item ${from + 1}`} moved to position ${to + 1} of ${form.items.length}.`);
  };

  useLayoutEffect(() => {
    const before = positions.current;
    positions.current = null;
    if (!before || !listRef.current) return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    for (const el of listRef.current.children) {
      const from = before.get(el.dataset.key);
      if (from === undefined) continue;
      const dy = from - el.getBoundingClientRect().top;
      if (!dy) continue;
      el.animate([{ transform: `translateY(${dy}px)` }, { transform: 'translateY(0)' }], {
        duration: 420,
        easing: 'cubic-bezier(0.16, 1, 0.3, 1)',
      });
    }
  }, [form.items]);

  const focusField = (field) => {
    setShowProblems(false);
    const el = document.getElementById(fieldId(field)) ?? document.getElementById(fieldId(field.split(':')[0]));
    if (el) {
      el.focus();
      el.scrollIntoView({ block: 'center', behavior: 'smooth' });
    }
  };

  const exportFile = () => {
    if (problems.length) {
      setShowProblems(true);
      setNotice({ kind: 'problem', text: `Fix ${problems.length === 1 ? 'the problem' : `the ${problems.length} problems`} first, so rubric-eval can read the file.` });
      return;
    }
    const blob = new Blob([toJson(form)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${idFromName(form.name) || 'rubric'}.json`;
    a.click();
    URL.revokeObjectURL(url);
    setNotice({ kind: 'ok', text: `Saved ${a.download}. In rubric-eval: npm run eval -- --rubric <path to ${a.download}>` });
  };

  // Swaps in a whole new form, keeping the old one for Undo.
  const replace = (next, text) => {
    setPrevious(form);
    setForm(next);
    setMoved(null);
    setShowProblems(false);
    setNotice({ kind: 'ok', text, undo: true });
  };

  const undo = () => {
    if (previous) setForm(previous);
    setPrevious(null);
    setNotice({ kind: 'ok', text: 'Put back the sheet you had.' });
  };

  useEffect(() => {
    if (!notice?.undo) return;
    const timer = setTimeout(() => setNotice((n) => (n?.undo ? { ...n, undo: false } : n)), 10000);
    return () => clearTimeout(timer);
  }, [notice]);

  const load = (text, label) => {
    try {
      replace(fromRubric(parseText(text)), `Opened ${label}.`);
    } catch (err) {
      setNotice({ kind: 'problem', text: `Couldn't open ${label}. ${err.message}` });
    }
  };

  const importFile = async (event) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (file) load(await file.text(), file.name);
  };

  return (
    <>
      <main className="desk">
        <article className="sheet" aria-label="Rubric sheet">
          <header className="sheet-head">
            <Field label="Rubric name" field="name" problem={byField.get('name')}>
              {/* Wraps like the other fields, so a long name stays readable; a name has no line breaks. */}
              <AutoText
                id={fieldId('name')}
                className="write title"
                value={form.name}
                placeholder="What is being scored"
                onKeyDown={(e) => e.key === 'Enter' && e.preventDefault()}
                onChange={(value) => edit((f) => ({ ...f, name: value.replace(/\s*\n\s*/g, ' ') }))}
              />
            </Field>
            <Field label="Context for whoever scores (optional)" field="context">
              <AutoText
                id={fieldId('context')}
                className="write"
                value={form.context}
                placeholder="What the work is and what to judge"
                onChange={(value) => edit((f) => ({ ...f, context: value }))}
              />
            </Field>
          </header>

          <Scale form={form} edit={edit} byField={byField} />

          <section aria-labelledby="items-title">
            <h2 id="items-title" className="printed section-title">
              Items <span className="count">{form.items.length}</span>
            </h2>
            {byField.has('items') && <p className="problem-text">{byField.get('items')}</p>}
            <ol className="items" ref={listRef} id="f-items" tabIndex={-1}>
              {form.items.map((item, index) => (
                <Item
                  key={item.key}
                  item={item}
                  index={index}
                  count={form.items.length}
                  bands={form.bands}
                  byField={byField}
                  moved={moved === item.key}
                  setItem={setItem}
                  move={move}
                  remove={() => edit((f) => ({ ...f, items: f.items.filter((it) => it.key !== item.key) }))}
                  listRef={listRef}
                />
              ))}
            </ol>
            <button
              type="button"
              className="add-item"
              onClick={() => {
                edit((f) => ({ ...f, items: [...f.items, newItem(f.bands)] }));
                requestAnimationFrame(() => {
                  const last = listRef.current?.lastElementChild;
                  last?.querySelector('input')?.focus();
                });
              }}
            >
              <PlusIcon /> Add an item
            </button>
          </section>

          <footer className="sheet-foot">
            <p>
              Start from{' '}
              <button type="button" className="link" onClick={() => load(ellipseText, 'the ELLIPSE rubric')}>
                rubric-eval's ELLIPSE rubric
              </button>
              ,{' '}
              <button type="button" className="link" onClick={() => replace(sampleForm(), 'Opened the service desk example.')}>
                the service desk example
              </button>{' '}
              or{' '}
              <button type="button" className="link" onClick={() => replace(blankForm(), 'Started a blank sheet.')}>
                a blank sheet
              </button>
              . Export JSON saves the file rubric-eval reads. Nothing leaves this page: the file is made in your browser.
            </p>
            <p>
              The ELLIPSE rubric is adapted from the ELLIPSE corpus, CC BY-NC-SA 4.0. The service desk example is invented. Code:{' '}
              <a href="https://github.com/nurah-kamal/statistical-modelling/tree/main/rubric-builder">rubric-builder</a> and{' '}
              <a href="https://github.com/nurah-kamal/statistical-modelling/tree/main/rubric-eval">rubric-eval</a>.
            </p>
          </footer>
        </article>
      </main>

      <Readout
        form={form}
        problems={problems}
        show={showProblems}
        setShow={setShowProblems}
        focusField={focusField}
        notice={notice}
        onExport={exportFile}
        onImport={importFile}
        onUndo={undo}
      />
      <p className="visually-hidden" aria-live="polite">
        {announcement}
      </p>
    </>
  );
}

function Field({ label, field, problem, children, className = '' }) {
  return (
    <div className={`field ${className}`} data-invalid={problem ? '' : undefined}>
      <label className="printed" htmlFor={fieldId(field)}>
        {label}
      </label>
      {children}
      {problem && (
        <p className="problem-text" id={`${fieldId(field)}-problem`}>
          {problem}
        </p>
      )}
    </div>
  );
}

// A textarea that grows with what is written in it.
function AutoText({ value, onChange, className, ...rest }) {
  const ref = useRef(null);
  const fit = () => {
    const el = ref.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${el.scrollHeight}px`;
  };
  useLayoutEffect(fit, [value]);
  // Text rewraps when the width changes or the web font arrives, so measure again
  // then. Only a change of width counts, so setting the height can't loop.
  useEffect(() => {
    let width = ref.current.clientWidth;
    const observer = new ResizeObserver(() => {
      if (ref.current && ref.current.clientWidth !== width) {
        width = ref.current.clientWidth;
        fit();
      }
    });
    observer.observe(ref.current);
    document.fonts?.ready.then(fit);
    return () => observer.disconnect();
  }, []);
  return <textarea ref={ref} rows={1} className={className} value={value} onChange={(e) => onChange(e.target.value)} {...rest} />;
}

function Scale({ form, edit, byField }) {
  const [split, setSplit] = useState(String(form.bands.length || 4));
  useEffect(() => setSplit(String(form.bands.length || 4)), [form.bands.length]);
  const min = Number(form.min);
  const max = Number(form.max);
  const track = Number.isInteger(min) && Number.isInteger(max) && min < max && max - min <= 40;

  // Where each band sits on the track; a band that can't be placed is listed after it.
  const placed = [];
  const loose = [];
  const owners = new Map();
  for (const band of form.bands) {
    const p = parseBand(band.text);
    if (track && p && p.lo >= min && p.hi <= max) {
      placed.push({ band, p });
      for (let s = p.lo; s <= p.hi; s++) owners.set(s, (owners.get(s) ?? 0) + 1);
    } else {
      loose.push(band);
    }
  }
  const scores = [];
  if (track) for (let s = min; s <= max; s++) scores.push(s);

  const setBands = (bands) =>
    edit((f) => ({
      ...f,
      bands,
      items: f.items.map((it) => ({
        ...it,
        guidance: Object.fromEntries(bands.map((b) => [b.key, it.guidance[b.key] ?? ''])),
      })),
    }));

  const splitEvenly = () => {
    const n = Number(split);
    if (!track || !Number.isInteger(n) || n < 1) return;
    const next = splitScale(min, max, n).map((b) => bandKey(b));
    // Each band keeps its guidance where a band survives in the same position.
    setBands(next.map((text, i) => (form.bands[i] ? { ...form.bands[i], text } : newBand(text))));
  };

  const bandOval = (band, i, style) => {
    const id = fieldId(`band:${band.key}`);
    const problem = byField.get(`band:${band.key}`);
    return (
      <div key={band.key} className="band-oval" style={style} data-invalid={problem ? '' : undefined}>
        <label className="visually-hidden" htmlFor={id}>
          Band {i + 1}, a score or a range
        </label>
        <input
          id={id}
          className="band-input"
          value={band.text}
          size={Math.max(2, band.text.length)}
          aria-invalid={problem ? true : undefined}
          aria-describedby={problem ? `${id}-problem` : undefined}
          onChange={(e) => setBands(form.bands.map((b) => (b.key === band.key ? { ...b, text: e.target.value } : b)))}
        />
        <button
          type="button"
          className="band-remove"
          aria-label={`Remove band ${band.text}`}
          onClick={() => setBands(form.bands.filter((b) => b.key !== band.key))}
        >
          <CrossIcon />
        </button>
        {problem && (
          <span className="visually-hidden" id={`${id}-problem`}>
            {problem}
          </span>
        )}
      </div>
    );
  };

  return (
    <section className="scale" aria-labelledby="scale-title">
      <h2 id="scale-title" className="printed section-title">
        Scale, the same for every item
      </h2>
      <div className="scale-row">
        <div className="field small" data-invalid={byField.has('scale') ? '' : undefined}>
          <label className="printed" htmlFor={fieldId('scale')}>
            Lowest
          </label>
          <input
            id={fieldId('scale')}
            className="write number"
            inputMode="numeric"
            value={form.min}
            onChange={(e) => edit((f) => ({ ...f, min: e.target.value }))}
          />
        </div>
        <div className="field small" data-invalid={byField.has('scale') ? '' : undefined}>
          <label className="printed" htmlFor="f-scale-max">
            Highest
          </label>
          <input
            id="f-scale-max"
            className="write number"
            inputMode="numeric"
            value={form.max}
            onChange={(e) => edit((f) => ({ ...f, max: e.target.value }))}
          />
        </div>
        <div className="field small split">
          <label className="printed" htmlFor="f-split">
            Split into
          </label>
          <div className="split-row">
            <input
              id="f-split"
              className="write number"
              inputMode="numeric"
              value={split}
              onChange={(e) => setSplit(e.target.value)}
            />
            <span className="unit">bands</span>
            <button type="button" className="quiet" onClick={splitEvenly} disabled={!track}>
              Split evenly
            </button>
          </div>
        </div>
      </div>
      {byField.has('scale') && <p className="problem-text">{byField.get('scale')}</p>}

      <div className="bands" role="group" aria-labelledby="bands-title" id="f-bands" tabIndex={-1}>
        <h3 id="bands-title" className="printed">
          Bands
        </h3>
        <div className="track-row">
        {track && (
          <div className="track" style={{ '--scores': scores.length }}>
            {scores.map((s, i) => {
              const n = owners.get(s) ?? 0;
              return (
                <span
                  key={s}
                  className="score"
                  style={{ gridColumn: i + 1 }}
                  data-state={n === 0 ? 'gap' : n > 1 ? 'overlap' : undefined}
                  aria-hidden="true"
                >
                  {s}
                </span>
              );
            })}
            {scores.map((s, i) =>
              owners.has(s) ? null : (
                <span key={`gap-${s}`} className="gap-mark" style={{ gridColumn: i + 1, gridRow: 2 }} aria-hidden="true" />
              ),
            )}
            {placed.map(({ band, p }) =>
              bandOval(band, form.bands.indexOf(band), { gridColumn: `${p.lo - min + 1} / ${p.hi - min + 2}` }),
            )}
          </div>
        )}
          {loose.map((band) => bandOval(band, form.bands.indexOf(band)))}
          <button
            type="button"
            className="quiet add-band"
            onClick={() => {
              const last = parseBand(form.bands.at(-1)?.text ?? '');
              const next = last ? String(last.hi + 1) : form.min;
              setBands([...form.bands, newBand(next)]);
            }}
          >
            <PlusIcon /> Add a band
          </button>
        </div>
        {[...byField.entries()]
          .filter(([field]) => field === 'bands' || field.startsWith('band:'))
          .map(([field, message]) => (
            <p key={field} className="problem-text">
              {message}
            </p>
          ))}
      </div>
    </section>
  );
}

function Item({ item, index, count, bands, byField, moved, setItem, move, remove, listRef }) {
  const ref = useRef(null);
  const drag = useRef(null);
  const [dragging, setDragging] = useState(false);
  const name = item.name.trim() || `Item ${index + 1}`;
  const problem = (part) => byField.get(`item:${item.key}:${part}`);

  // Pointer dragging by the timing mark works for mouse, pen and touch alike.
  const onPointerDown = (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      // No live pointer to capture (a synthetic event); the drag still works.
    }
    const siblings = [...listRef.current.children];
    drag.current = {
      startY: e.clientY,
      mids: siblings.map((el) => {
        const r = el.getBoundingClientRect();
        return r.top + r.height / 2;
      }),
      target: index,
    };
    setDragging(true);
  };
  const onPointerMove = (e) => {
    const d = drag.current;
    if (!d) return;
    const dy = e.clientY - d.startY;
    ref.current.style.transform = `translateY(${dy}px)`;
    const y = d.mids[index] + dy;
    let target = d.mids.filter((m) => m < y).length;
    if (target > index) target -= 1;
    d.target = Math.max(0, Math.min(count - 1, target));
  };
  const onPointerUp = () => {
    const d = drag.current;
    if (!d) return;
    drag.current = null;
    ref.current.style.transform = '';
    setDragging(false);
    move(index, d.target);
  };
  const onKeyDown = (e) => {
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      move(index, index - 1);
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      move(index, index + 1);
    }
  };

  return (
    <li ref={ref} className="item" data-key={item.key} data-moved={moved ? '' : undefined} data-dragging={dragging ? '' : undefined}>
      <button
        type="button"
        className="timing-mark"
        aria-label={`Move ${name}, item ${index + 1} of ${count}. Drag, or use the up and down arrow keys.`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onKeyDown={onKeyDown}
      />
      <div className="item-body">
        <div className="item-head">
          <span className="printed item-number">Item {index + 1}</span>
          <div className="item-actions">
            <button type="button" className="quiet" onClick={() => move(index, index - 1)} disabled={index === 0} aria-label={`Move ${name} up`}>
              <ArrowIcon up />
            </button>
            <button type="button" className="quiet" onClick={() => move(index, index + 1)} disabled={index === count - 1} aria-label={`Move ${name} down`}>
              <ArrowIcon />
            </button>
            <button type="button" className="quiet danger" onClick={remove}>
              Remove
            </button>
          </div>
        </div>
        <div className="item-fields">
          <Field label="Name" field={`item:${item.key}:name`} problem={problem('name')} className="grow">
            <input
              id={fieldId(`item:${item.key}:name`)}
              className="write item-name"
              value={item.name}
              placeholder="What this item judges"
              aria-invalid={problem('name') ? true : undefined}
              aria-describedby={problem('name') ? `${fieldId(`item:${item.key}:name`)}-problem` : undefined}
              onChange={(e) =>
                setItem(item.key, (it) => ({
                  name: e.target.value,
                  id: it.idEdited ? it.id : idFromName(e.target.value),
                }))
              }
            />
          </Field>
          <Field label="Id in the file" field={`item:${item.key}:id`} problem={problem('id')} className="id-field">
            <input
              id={fieldId(`item:${item.key}:id`)}
              className="write mono"
              value={item.id}
              spellCheck={false}
              aria-invalid={problem('id') ? true : undefined}
              aria-describedby={problem('id') ? `${fieldId(`item:${item.key}:id`)}-problem` : undefined}
              onChange={(e) => setItem(item.key, () => ({ id: e.target.value, idEdited: true }))}
            />
          </Field>
        </div>
        <Field label="Description" field={`item:${item.key}:description`} problem={problem('description')}>
          <AutoText
            id={fieldId(`item:${item.key}:description`)}
            className="write"
            value={item.description}
            placeholder="What a scorer should look at"
            aria-invalid={problem('description') ? true : undefined}
            onChange={(value) => setItem(item.key, () => ({ description: value }))}
          />
        </Field>
        <fieldset className="guidance">
          <legend className="printed">Guidance per band</legend>
          {bands.map((band) => {
            const id = fieldId(`item:${item.key}:band:${band.key}`);
            const filled = Boolean(item.guidance[band.key]?.trim());
            return (
              <div key={band.key} className="band-row">
                <label htmlFor={id} className="bubble" data-filled={filled ? '' : undefined}>
                  <span>{band.text || '?'}</span>
                  <span className="visually-hidden">{filled ? ', written' : ', still empty'}</span>
                </label>
                <AutoText
                  id={id}
                  className="write guidance-text"
                  value={item.guidance[band.key] ?? ''}
                  placeholder="What work in this band looks like"
                  aria-invalid={filled ? undefined : true}
                  onChange={(value) => setItem(item.key, (it) => ({ guidance: { ...it.guidance, [band.key]: value } }))}
                />
              </div>
            );
          })}
        </fieldset>
      </div>
    </li>
  );
}

function Readout({ form, problems, show, setShow, focusField, notice, onExport, onImport, onUndo }) {
  const n = problems.length;
  const first = useRef(n);
  const tick = n !== first.current ? ' tick-in' : '';
  if (tick) first.current = -1;
  const scale = `${form.min || '?'}-${form.max || '?'}`;
  return (
    <aside className="readout" aria-label="Sheet read-out">
      {show && n > 0 && (
        <div className="problem-list" id="problem-list">
          <p className="printed">What to fix</p>
          <ul>
            {problems.map((p, i) => (
              <li key={`${p.field}-${i}`}>
                <button type="button" className="link" onClick={() => focusField(p.field)}>
                  {p.message}
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
      <div className="readout-bar">
        <p className="read" aria-live="polite">
          <span>{form.items.length} {form.items.length === 1 ? 'item' : 'items'} read</span>
          <span>
            scale {scale}, {form.bands.length} {form.bands.length === 1 ? 'band' : 'bands'}
          </span>
          {n === 0 ? (
            <span className={`ok${tick}`} key="ok">no problems</span>
          ) : (
            <button
              key={n}
              type="button"
              className={`problems-toggle${tick}`}
              aria-expanded={show}
              aria-controls="problem-list"
              onClick={() => setShow(!show)}
            >
              {n} {n === 1 ? 'problem' : 'problems'}
            </button>
          )}
        </p>
        <div className="readout-actions">
          <label className="button secondary">
            Import
            <input type="file" accept=".json,.yaml,.yml,application/json" onChange={onImport} className="visually-hidden" />
          </label>
          <button type="button" className="button primary" onClick={onExport}>
            Export JSON
          </button>
        </div>
      </div>
      {notice && (
        <p className="notice" data-kind={notice.kind} role="status">
          {notice.text}
          {notice.undo && (
            <button type="button" className="link undo" onClick={onUndo}>
              Undo
            </button>
          )}
        </p>
      )}
    </aside>
  );
}

function PlusIcon() {
  return (
    <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" focusable="false">
      <path d="M8 3v10M3 8h10" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" fill="none" />
    </svg>
  );
}

function CrossIcon() {
  return (
    <svg viewBox="0 0 16 16" width="12" height="12" aria-hidden="true" focusable="false">
      <path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" fill="none" />
    </svg>
  );
}

function ArrowIcon({ up }) {
  return (
    <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" focusable="false">
      <path d={up ? 'M8 13V3M4 7l4-4 4 4' : 'M8 3v10M4 9l4 4 4-4'} stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" fill="none" />
    </svg>
  );
}
