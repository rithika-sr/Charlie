// Trip Planner view: form + ranked journey options.

import { api } from '../core/api.js';
import { store } from '../core/store.js';
import { escapeHtml as e } from '../core/format.js';
import { alertsForRoutes, effectLabel } from '../core/alerts.js';
import { tripFormHtml, bindTripForm } from '../ui/trip-form.js';
import { tripOptionCard, emptyState, errorState, skeleton, sectionHeader } from '../ui/components.js';
import { icon } from '../ui/icons.js';
import { createRequestGate } from '../core/request-gate.js';

const gate = createRequestGate();
let lastQuery = null;
let lastRoutes = null;
let lastNotices = [];

/** Map a trip step's display name to MBTA route ids, for alert matching. */
export function routeIdsForStep(step) {
  const name = String(step.name || '');
  const m = name.match(/Green Line ([BCDE])\b/);
  if (m) return [`Green-${m[1]}`];
  if (/^Green Line/.test(name)) return ['Green-B', 'Green-C', 'Green-D', 'Green-E'];
  for (const line of ['Red', 'Orange', 'Blue']) if (name.startsWith(`${line} Line`)) return [line];
  if (/Mattapan/i.test(name)) return ['Mattapan'];
  const bus = name.match(/^Bus (\w+)/);
  if (bus) return [bus[1]];
  return [];
}

export const SORTS = [
  { id: 'arrival', name: 'Earliest arrival' },
  { id: 'duration', name: 'Shortest duration' },
  { id: 'transfers', name: 'Fewest transfers' },
  { id: 'walking', name: 'Least walking' },
];

// Numeric metric per sort. Missing timestamps sort last instead of guessing.
const METRICS = {
  arrival: r => (Number.isFinite(r.arrive_ts) ? r.arrive_ts : Infinity),
  duration: r => (Number.isFinite(r.duration) ? r.duration : Infinity),
  transfers: r => (Number.isFinite(r.transfers) ? r.transfers : Infinity),
  walking: r => (Number.isFinite(r.walk_time) ? r.walk_time : 0),
};
const TIEBREAK = ['arrival', 'duration', 'transfers', 'walking'];

/** The arrival minute riders see (clock times drop seconds), from the full timestamp. */
export const arrivalMinute = r => (Number.isFinite(r.arrive_ts) ? Math.floor(r.arrive_ts / 60) : Infinity);

// Comparison chain per sort. Earliest arrival compares the displayed minute, then prefers
// fewer transfers and less walking, then the exact timestamp and duration.
const CHAINS = {
  arrival: [arrivalMinute, METRICS.transfers, METRICS.walking, METRICS.arrival, METRICS.duration],
  duration: [METRICS.duration, METRICS.arrival, METRICS.transfers, METRICS.walking],
  transfers: [METRICS.transfers, METRICS.arrival, METRICS.duration, METRICS.walking],
  walking: [METRICS.walking, METRICS.arrival, METRICS.duration, METRICS.transfers],
};

/** Sort itineraries by `by` (default: earliest arrival); the routing service's order breaks final ties. */
export function sortTrips(routes, by = 'arrival') {
  const chain = CHAINS[by] || CHAINS.arrival;
  return routes.map((r, i) => [r, i]).sort((a, b) => {
    for (const f of chain) {
      const d = f(a[0]) - f(b[0]);
      if (d) return d;                 // NaN (both missing) counts as a tie
    }
    return a[1] - b[1];
  }).map(x => x[0]);
}

const BADGE_FOR = { arrival: 'Earliest arrival', duration: 'Shortest trip', transfers: 'Fewest transfers', walking: 'Least walking' };

/** Transparent labels: an option gets a badge only when it is the single best on that measure. */
export function tripBadges(routes) {
  const badges = routes.map(() => []);
  if (routes.length < 2) return badges;
  // "Earliest arrival" goes to the option that ranks first under that sort.
  const top = sortTrips(routes, 'arrival')[0];
  if (Number.isFinite(top.arrive_ts)) badges[routes.indexOf(top)].push(BADGE_FOR.arrival);
  for (const key of TIEBREAK.filter(k => k !== 'arrival')) {
    const values = routes.map(METRICS[key]);
    const min = Math.min(...values);
    if (!Number.isFinite(min)) continue;
    const winners = values.flatMap((v, i) => (v === min ? [i] : []));
    if (winners.length === 1) badges[winners[0]].push(BADGE_FOR[key]);
  }
  return badges;
}

/**
 * "Shortest ride, but leaves later…" — only when the shortest option really trades a
 * later arrival for less time on board (no earliest-arriving option is as short).
 */
export function arrivalNote(route, routes) {
  const earliest = Math.min(...routes.map(arrivalMinute));
  if (!Number.isFinite(earliest) || !Number.isFinite(route.arrive_ts)) return '';
  const later = arrivalMinute(route) - earliest;
  const shortest = Math.min(...routes.map(METRICS.duration));
  if (later <= 0 || route.duration !== shortest) return '';
  if (routes.some(r => arrivalMinute(r) === earliest && r.duration === shortest)) return '';
  return `Shortest ride, but leaves later and arrives ${later} min after the earliest option`;
}

const SCOPE_TEXT = { stop: 'at', segment: 'on your ride', direction: 'this direction', route: 'line-wide' };

/** One-line alert text for an itinerary, at the scope the MBTA feed supports. */
export function tripAlertText(a) {
  const what = a.service_effect || a.header || 'Service alert';
  const short = what.length > 110 ? `${what.slice(0, 107)}…` : what;
  if (a.scope === 'stop' && a.stop_names?.length) return `${a.step} · ${effectLabel(a.effect)} at ${[...new Set(a.stop_names)].join(', ')}: ${short}`;
  if (a.scope === 'segment' && a.stop_names?.length) return `${a.step} · ${effectLabel(a.effect)} at ${[...new Set(a.stop_names)].join(', ')}, which your ride passes through: ${short}`;
  return `${a.step} (${SCOPE_TEXT[a.scope] || 'line-wide'}): ${short}`;
}

let sortBy = 'arrival';

function noticeHtml(n) {
  const where = n.role === 'destination' ? 'your destination' : 'your starting point';
  const scope = n.direction_specific ? ' in one direction' : '';
  return `<div class="trip-notice" role="note">
    ${icon('alert', { size: 18 })}
    <div><strong>${e(n.station)} (${e(where)}): ${e(effectLabel(n.effect).toLowerCase())}${e(scope)}.</strong>
    ${n.explanation ? `<span>${e(n.explanation)}</span>` : ''}
    <p class="muted small">${e(n.header)}</p></div>
  </div>`;
}

function renderResults(el, unsorted, notices = []) {
  const routes = sortTrips(unsorted, sortBy);
  const legacyAlerts = store.get('alerts') || [];
  const badges = tripBadges(routes);
  el.innerHTML = `
    ${notices.map(noticeHtml).join('')}
    ${sectionHeader(`${routes.length} option${routes.length === 1 ? '' : 's'}`)}
    ${routes.length > 1 ? `<div class="segmented segmented--scroll trip-sort" role="group" aria-label="Sort trip options">${SORTS.map(s =>
      `<button class="segmented__btn" data-sort="${s.id}" aria-pressed="${s.id === sortBy}">${e(s.name)}</button>`).join('')}</div>` : ''}
    <div class="stack">${routes.map((r, i) => {
      let affected;
      if (Array.isArray(r.alerts)) affected = r.alerts.map(tripAlertText);
      else {
        // Older responses: match on route only, and say so.
        affected = [];
        for (const step of r.steps) {
          const hits = alertsForRoutes(legacyAlerts, routeIdsForStep(step));
          if (hits.length) affected.push(tripAlertText({ ...hits[0], step: step.name, scope: 'route' }));
        }
      }
      return tripOptionCard(r, i, { badges: badges[i], alerts: affected, note: arrivalNote(r, routes) });
    }).join('')}</div>
    <p class="fine-print">${icon('info', { size: 14 })}Journey times come from the routing service and are estimates. Fares are estimated. Check alerts before you travel.</p>`;
  el.querySelectorAll('[data-sort]').forEach(btn => btn.addEventListener('click', () => {
    sortBy = btn.dataset.sort;
    renderResults(el, unsorted, notices);
    el.querySelector(`[data-sort="${sortBy}"]`)?.focus();
  }));
}

async function search(resultsEl, q, formApi) {
  lastQuery = q;
  const { token, signal } = gate.start();
  formApi?.clearErrors();
  resultsEl.innerHTML = `<div class="stack">${skeleton(3, { height: 112 })}</div>`;
  resultsEl.setAttribute('aria-busy', 'true');
  resultsEl.setAttribute('data-has-results', '');
  const body = { origin: q.origin.name, destination: q.destination.name };
  if (q.origin.lat != null) body.origin_coords = `${q.origin.lat},${q.origin.lon}`;
  if (q.destination.lat != null) body.dest_coords = `${q.destination.lat},${q.destination.lon}`;
  if (q.origin.id) body.origin_id = q.origin.id;
  if (q.destination.id) body.dest_id = q.destination.id;
  const { data, error, aborted } = await api.trip(body, { signal });
  // Never let an older or invalidated response overwrite the current endpoints' state.
  if (aborted || !gate.isCurrent(token)) return;
  resultsEl.removeAttribute('aria-busy');
  if (error) {
    resultsEl.innerHTML = errorState({ title: 'The trip planner couldn’t be reached', body: error });
    resultsEl.querySelector('[data-action="retry"]').addEventListener('click', () => search(resultsEl, q, formApi));
    return;
  }
  if (data.code === 'unresolved') {
    const fields = data.unresolved || [];
    for (const f of fields) {
      const place = f === 'origin' ? q.origin.name : q.destination.name;
      formApi?.fieldError(f, `We couldn’t find “${place}” in the Boston area. Check the spelling, add a street or town, or pick a suggestion.`);
    }
    formApi?.focusField(fields[0]);
    resultsEl.innerHTML = emptyState({ icon: 'pin', title: fields.length > 1 ? 'Neither place was found' : `${fields[0] === 'origin' ? 'Starting point' : 'Destination'} not found`, body: 'Fix the highlighted field and search again.' });
    return;
  }
  if (!data.routes || !data.routes.length) {
    resultsEl.innerHTML = `${(data.notices || []).map(noticeHtml).join('')}${emptyState({ icon: 'route', title: 'No transit options for this trip right now', body: 'Both places were found, but the routing service returned no itineraries. Service may have ended for the night, or try a nearby station.' })}`;
    return;
  }
  store.rememberTrip({ origin: q.origin, destination: q.destination });
  lastRoutes = data.routes;
  lastNotices = data.notices || [];
  renderResults(resultsEl, data.routes, lastNotices);
}

/** An endpoint changed: drop in-flight work and any results that no longer match the form. */
function invalidateResults(resultsEl, form) {
  gate.invalidate();
  lastRoutes = null;
  lastNotices = [];
  if (!resultsEl.hasAttribute('data-has-results')) return;   // still the initial "Where to?" state
  resultsEl.removeAttribute('data-has-results');
  resultsEl.removeAttribute('aria-busy');
  resultsEl.innerHTML = emptyState({ icon: 'route', title: 'Plan your updated trip',
    body: 'Your starting point or destination changed. Select Plan journey to see options for this trip.' });
}

export const tripView = {
  enter(el) {
    const pending = store.get('pendingTrip') || {};
    store.set('pendingTrip', null);
    const recents = store.get('recentTrips');
    el.innerHTML = `
      <div class="page page--split">
        <section class="page__aside">
          <header class="page-head"><h1>Trip Planner</h1><p class="muted">Door-to-door transit directions across subway, bus, Commuter Rail and ferry.</p></header>
          <div class="card card--pad">${tripFormHtml({ large: true })}</div>
          ${recents.length ? `<div class="card card--pad"><h2 class="subhead">Recent trips</h2><ul class="plain-list recent-trips">${recents.map((t, i) =>
            `<li><button class="link-row" data-recent="${i}">${icon('history', { size: 16 })}<span>${e(t.origin.name)} → ${e(t.destination.name)}</span></button></li>`).join('')}</ul></div>` : ''}
        </section>
        <section class="page__main" aria-live="polite" data-results>
          ${emptyState({ icon: 'route', title: 'Where to?', body: 'Pick a station from the suggestions or type an address. Options appear here with times, transfers and walking.' })}
        </section>
      </div>`;
    const results = el.querySelector('[data-results]');
    const form = el.querySelector('[data-trip-form]');
    const initial = { origin: pending.origin || lastQuery?.origin, destination: pending.destination || lastQuery?.destination };
    const formApi = bindTripForm(form, { initial, onSubmit: q => search(results, q, formApi), onChange: () => invalidateResults(results, form) });
    el.querySelectorAll('[data-recent]').forEach(btn => btn.addEventListener('click', () => {
      const t = recents[Number(btn.dataset.recent)];
      form.querySelector('[name="origin"]').value = t.origin.name;
      Object.assign(form.querySelector('[name="origin"]').dataset, t.origin.lat != null ? { lat: t.origin.lat, lon: t.origin.lon, id: t.origin.id || '' } : {});
      form.querySelector('[name="destination"]').value = t.destination.name;
      Object.assign(form.querySelector('[name="destination"]').dataset, t.destination.lat != null ? { lat: t.destination.lat, lon: t.destination.lon, id: t.destination.id || '' } : {});
      search(results, t, formApi);
    }));
    if (pending.origin && pending.destination) search(results, { origin: pending.origin, destination: pending.destination }, formApi);
    else if (pending.origin || pending.destination) form.querySelector(pending.origin ? '[name="destination"]' : '[name="origin"]').focus();
    else if (lastQuery && lastRoutes) { results.setAttribute('data-has-results', ''); renderResults(results, lastRoutes, lastNotices); }
  },
  leave() { gate.abort(); },
};
