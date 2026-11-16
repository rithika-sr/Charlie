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

export function tripFormHtml({ large = false, submitLabel = 'Plan journey' } = {}) {
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

  form.addEventListener('submit', ev => {
    ev.preventDefault();
    const o = readPlace(origin), d = readPlace(dest);
    clearErrors();
    if (!o || !d) {
      if (!o) fieldError('origin', 'Enter a starting point.');
      if (!d) fieldError('destination', 'Enter a destination.');
      (!o ? origin : dest).focus();
      return;
    }
    onSubmit({ origin: o, destination: d });
  });

  return {
    setOrigin: p => setValue(origin, p), setDestination: p => setValue(dest, p),
    fieldError, clearErrors,
    focusField: name => {
      const input = fieldOf(name);
      if (!input) return;
      if (document.activeElement === input) { input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' })); return; }
      input.dataset.noAutoOpen = '1';
      input.focus();
    },
  };
}
