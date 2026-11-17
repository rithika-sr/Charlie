// Origin/destination form shared by Overview and Trip Planner.
// Selected stations keep their id and coordinates; typing free text clears them.

import { store } from '../core/store.js';
import { rank } from '../core/search.js';
import { escapeHtml as e } from '../core/format.js';
import { combobox } from './combobox.js';
import { lineBadge } from './components.js';
import { icon } from './icons.js';
import { requestLocation } from './location.js';
import { toast } from './toast.js';

let formId = 0;

export const WHEN_MODES = [
  { id: 'now', name: 'Leave now' },
  { id: 'depart', name: 'Depart at' },
  { id: 'arrive', name: 'Arrive by' },
];
const QUICK_MINUTES = [10, 20, 30, 60];
const BOSTON_TZ = 'America/New_York';

/** Wall-clock date ('YYYY-MM-DD') and time ('HH:MM') in Boston, `plusMin` minutes from `from`. */
export function bostonParts(plusMin = 0, from = new Date()) {
  const d = new Date(from.getTime() + plusMin * 60000);
  const p = Object.fromEntries(new Intl.DateTimeFormat('en-CA', {
    timeZone: BOSTON_TZ, year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hourCycle: 'h23',
  }).formatToParts(d).map(x => [x.type, x.value]));
  return { date: `${p.year}-${p.month}-${p.day}`, time: `${p.hour}:${p.minute}` };
}

/** Round a Boston time up to the next 5 minutes, `plusMin` from now. */
export function suggestedParts(plusMin = 10, from = new Date()) {
  const ms = from.getTime() + plusMin * 60000;
  return bostonParts(0, new Date(Math.ceil(ms / 300000) * 300000));
}

/** "Leaving 6:10 PM" / "Arrive by 9:00 AM tomorrow" from a mode and Boston wall-clock parts. */
export function whenText(when, from = new Date()) {
  if (!when || when.mode === 'now') return 'Leaving now';
  const [date, time] = when.time.split('T');
  const [h, m] = time.split(':').map(Number);
  const clock = `${h % 12 || 12}:${String(m).padStart(2, '0')} ${h < 12 ? 'AM' : 'PM'}`;
  const today = bostonParts(0, from).date;
  const tomorrow = bostonParts(24 * 60, from).date;
  const day = date === today ? 'today' : date === tomorrow ? 'tomorrow'
    : new Date(`${date}T12:00:00`).toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
  return when.mode === 'arrive' ? `Arrive by ${clock} ${day}` : `Leave ${day} at ${clock}`;
}

function whenHtml(id) {
  return `<fieldset class="trip-when" data-when>
      <legend class="sr-only">When are you travelling?</legend>
      <div class="segmented trip-when__modes" role="group" aria-label="When">
        ${WHEN_MODES.map(m => `<button type="button" class="segmented__btn" data-when-mode="${m.id}" aria-pressed="${m.id === 'now'}">${e(m.name)}</button>`).join('')}
      </div>
      <div class="trip-when__pick" data-when-pick hidden>
        <div class="trip-when__inputs">
          <label class="sr-only" for="${id}-date">Date</label>
          <input id="${id}-date" class="input input--sm" type="date" name="when_date" />
          <label class="sr-only" for="${id}-time">Time</label>
          <input id="${id}-time" class="input input--sm" type="time" name="when_time" step="60" />
        </div>
        <div class="chips trip-when__quick" role="group" aria-label="Quick departure times">
          ${QUICK_MINUTES.map(n => `<button type="button" class="chip chip--btn" data-when-plus="${n}">${n < 60 ? `In ${n} min` : 'In 1 hr'}</button>`).join('')}
        </div>
        <p class="field-error field-error--inline" id="${id}-when-err" data-error-for="when" role="alert" hidden></p>
      </div>
    </fieldset>`;
}

export function tripFormHtml({ large = false, submitLabel = 'Plan journey', timeOptions = false } = {}) {
  const id = `tf${++formId}`;
  return `<form class="trip-form${large ? ' trip-form--lg' : ''}" data-trip-form novalidate>
    <div class="trip-form__fields">
      <div class="trip-form__rail" aria-hidden="true"><span class="dot dot--origin"></span><span class="rail"></span><span class="dot dot--dest"></span></div>
      <div class="trip-form__inputs">
        <div class="field combo">
          <label class="sr-only" for="${id}-from">Starting point</label>
          <input id="${id}-from" class="input" name="origin" placeholder="From — station, address or place" />
          <button type="button" class="field__action" data-locate aria-label="Use my current location" title="Use my current location">${icon('locate', { size: 18 })}</button>
        </div>
        <p class="field-error field-error--inline" id="${id}-from-err" data-error-for="origin" role="alert" hidden></p>
        <div class="field combo">
          <label class="sr-only" for="${id}-to">Destination</label>
          <input id="${id}-to" class="input" name="destination" placeholder="To — where are you going?" />
        </div>
        <p class="field-error field-error--inline" id="${id}-to-err" data-error-for="destination" role="alert" hidden></p>
      </div>
      <button type="button" class="icon-btn trip-form__swap" data-swap aria-label="Swap starting point and destination" title="Swap">${icon('swap', { size: 18 })}</button>
    </div>
    ${timeOptions ? whenHtml(id) : ''}
    <p class="field-error" data-error role="alert" hidden></p>
    <button class="btn btn--primary btn--block" type="submit">${icon('route', { size: 18 })}${e(submitLabel)}</button>
  </form>`;
}

function optionHtml(item) {
  if (item.kind === 'location') return `<span class="opt-icon">${icon('locate', { size: 16 })}</span><span class="opt-name">Current location</span><span class="opt-sub">${e(item.sub || '')}</span>`;
  if (item.kind === 'recent-trip') return `<span class="opt-icon">${icon('history', { size: 16 })}</span><span class="opt-name">${e(item.trip.origin.name)} → ${e(item.trip.destination.name)}</span>`;
  return `<span class="opt-icon">${icon(item.recent ? 'history' : 'pin', { size: 16 })}</span><span class="opt-name">${e(item.name)}</span><span class="opt-lines">${(item.lines || []).map(l => lineBadge(l, { size: 'sm' })).join('')}</span>`;
}

function setValue(input, place) {
  input.value = place ? place.name : '';
  if (place && place.lat != null) {
    input.dataset.lat = place.lat; input.dataset.lon = place.lon; input.dataset.id = place.id || '';
  } else {
    delete input.dataset.lat; delete input.dataset.lon; delete input.dataset.id;
  }
}

export function readPlace(input) {
  const name = input.value.trim();
  if (!name) return null;
  const place = { name };
  if (input.dataset.lat) Object.assign(place, { lat: Number(input.dataset.lat), lon: Number(input.dataset.lon), id: input.dataset.id || null });
  return place;
}

export function bindTripForm(form, { onSubmit, onChange = () => {}, initial = {} } = {}) {
  const origin = form.querySelector('[name="origin"]');
  const dest = form.querySelector('[name="destination"]');
  const error = form.querySelector('[data-error]');
  setValue(origin, initial.origin);
  setValue(dest, initial.destination);

  const stations = () => store.get('stations');
  const recentItems = includeLocation => {
    const items = [];
    const loc = store.get('location');
    if (includeLocation && loc && loc.lat) items.push({ kind: 'location', name: loc.station ? `Near ${loc.station}` : 'Current location', sub: loc.station ? `Near ${loc.station}` : '', lat: loc.lat, lon: loc.lon });
    for (const t of store.get('recentTrips').slice(0, 3)) items.push({ kind: 'recent-trip', trip: t });
    for (const id of store.get('recentStations').slice(0, 4)) {
      const s = store.stationById(id);
      if (s) items.push({ ...s, recent: true });
    }
    return items;
  };

  const fieldOf = name => (name === 'origin' ? origin : dest);
  const fieldError = (name, message) => {
    const input = fieldOf(name);
    const msg = form.querySelector(`[data-error-for="${name}"]`);
    if (!input || !msg) return;
    msg.textContent = message;
    msg.hidden = !message;
    input.setAttribute('aria-invalid', message ? 'true' : 'false');
    if (message) input.setAttribute('aria-describedby', msg.id); else input.removeAttribute('aria-describedby');
  };
  const clearErrors = () => { fieldError('origin', ''); fieldError('destination', ''); error.hidden = true; };

  for (const input of [origin, dest]) {
    input.addEventListener('input', () => {
      delete input.dataset.lat; delete input.dataset.lon; delete input.dataset.id;
      error.hidden = true;
      fieldError(input === origin ? 'origin' : 'destination', '');
      onChange();
    });
    combobox(input, {
      source: q => rank(stations(), q, s => s.name, 8),
      onOpenEmpty: () => recentItems(input === origin),
      render: optionHtml,
      emptyText: 'No stations match — you can still search this as an address',
      onSelect: item => {
        if (item.kind === 'recent-trip') {
          setValue(origin, item.trip.origin); setValue(dest, item.trip.destination);
          onChange();
          form.requestSubmit();
          return;
        }
        setValue(input, item.kind === 'location' ? { name: 'Current location', lat: item.lat, lon: item.lon } : item);
        fieldError(input === origin ? 'origin' : 'destination', '');
        onChange();
        if (input === origin && !dest.value) dest.focus();
      },
    });
  }

  form.querySelector('[data-swap]').addEventListener('click', () => {
    const a = readPlace(origin), b = readPlace(dest);
    setValue(origin, b); setValue(dest, a);
    clearErrors();
    onChange();
  });

  form.querySelector('[data-locate]').addEventListener('click', async () => {
    const loc = await requestLocation({ quiet: true });
    if (loc) {
      setValue(origin, { name: loc.station ? `Current location (near ${loc.station})` : 'Current location', lat: loc.lat, lon: loc.lon });
      onChange();
      toast('Starting from your current location', { tone: 'success' });
    }
  });

  // "Leave now" / "Depart at" / "Arrive by", with a Boston date and time.
  const whenBox = form.querySelector('[data-when]');
  let whenMode = 'now';
  const whenDate = form.querySelector('[name="when_date"]');
  const whenTime = form.querySelector('[name="when_time"]');
  const whenError = form.querySelector('[data-error-for="when"]');
  const showWhenError = message => {
    if (!whenError) return;
    whenError.textContent = message; whenError.hidden = !message;
    for (const i of [whenDate, whenTime]) i.setAttribute('aria-invalid', message ? 'true' : 'false');
  };
  const setWhen = (mode, parts) => {
    if (!whenBox) return;
    whenMode = mode;
    whenBox.querySelectorAll('[data-when-mode]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.whenMode === mode)));
    whenBox.querySelector('[data-when-pick]').hidden = mode === 'now';
    if (mode !== 'now') {
      const p = parts || (whenDate.value && whenTime.value ? null : suggestedParts(10));
      if (p) { whenDate.value = p.date; whenTime.value = p.time; }
      whenDate.min = bostonParts().date;
    }
    showWhenError('');
  };
  const readWhen = () => (!whenBox || whenMode === 'now' ? null : { mode: whenMode, time: `${whenDate.value}T${whenTime.value}` });
  if (whenBox) {
    whenBox.querySelectorAll('[data-when-mode]').forEach(b => b.addEventListener('click', () => { setWhen(b.dataset.whenMode); onChange(); }));
    whenBox.querySelectorAll('[data-when-plus]').forEach(b => b.addEventListener('click', () => {
      // "In 10 min" etc. always means leaving then.
      setWhen('depart', bostonParts(Number(b.dataset.whenPlus)));
      onChange();
    }));
    for (const i of [whenDate, whenTime]) i.addEventListener('change', () => { showWhenError(''); onChange(); });
    if (initial.when && initial.when.mode !== 'now' && initial.when.time) {
      const [date, time] = initial.when.time.split('T');
      setWhen(initial.when.mode, { date, time: time.slice(0, 5) });
    }
  }

  form.addEventListener('submit', ev => {
    ev.preventDefault();
    const o = readPlace(origin), d = readPlace(dest);
    clearErrors();
    showWhenError('');
    if (!o || !d) {
      if (!o) fieldError('origin', 'Enter a starting point.');
      if (!d) fieldError('destination', 'Enter a destination.');
      (!o ? origin : dest).focus();
      return;
    }
    const when = readWhen();
    if (when) {
      if (!whenDate.value || !whenTime.value) { showWhenError('Pick a date and time.'); (whenDate.value ? whenTime : whenDate).focus(); return; }
      const now = bostonParts();
      if (`${whenDate.value}T${whenTime.value}` < `${now.date}T${now.time}`) {
        showWhenError('That time has already passed. Pick a later time, or choose Leave now.');
        whenTime.focus();
        return;
      }
    }
    onSubmit({ origin: o, destination: d, when });
  });

  return {
    setOrigin: p => setValue(origin, p), setDestination: p => setValue(dest, p),
    fieldError, clearErrors, setWhen, readWhen, whenError: showWhenError,
    focusField: name => {
      const input = fieldOf(name);
      if (!input) return;
      if (document.activeElement === input) { input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' })); return; }
      input.dataset.noAutoOpen = '1';
      input.focus();
    },
  };
}
