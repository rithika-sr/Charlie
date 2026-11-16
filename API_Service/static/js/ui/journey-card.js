// Journey card under a Charlie AI reply. Everything shown comes from the structured
// journey the server returned (the same object the Trip Planner renders); the card
// adds no times, lines or fares of its own.

import { escapeHtml as e, formatClock } from '../core/format.js';
import { effectLabel } from '../core/alerts.js';
import { tripOptionCard, lineBadge } from './components.js';
import { icon } from './icons.js';
import { sortTrips, tripBadges, tripAlertText, arrivalNote } from '../views/trip.js';

const PROVIDERS = { transitous: 'Transitous', google: 'Google Maps' };
const ROLE = { origin: 'start', destination: 'destination' };

/** Endpoint as the chat API and the Trip Planner take it: name plus id/coords when known. */
export function endpointFor(ep) {
  if (!ep) return null;
  const out = { name: ep.kind === 'station' ? ep.name : (ep.query || ep.name) };
  if (ep.id) out.id = ep.id;
  if (Number.isFinite(ep.lat) && Number.isFinite(ep.lon)) Object.assign(out, { lat: ep.lat, lon: ep.lon });
  return out;
}

/** The two endpoints a card's actions use: resolved ones when available, else what was asked. */
export function journeyEndpoints(j) {
  const req = j.request || {};
  const pick = f => (j[f] && j[f].status === 'ok' ? endpointFor(j[f]) : req[f] || null);
  return { origin: pick('origin'), destination: pick('destination') };
}

function actionButton(i, kind, label, iconName, extra = '') {
  return `<button type="button" class="btn btn--ghost btn--sm" data-journey="${i}" data-journey-action="${kind}" ${extra}>${icon(iconName, { size: 14 })}${e(label)}</button>`;
}

function noticeHtml(n) {
  const where = n.role === 'destination' ? 'your destination' : 'your starting point';
  return `<div class="trip-notice" role="note">${icon('alert', { size: 18 })}
    <div><strong>${e(n.station)} (${e(where)}): ${e(effectLabel(n.effect).toLowerCase())}${n.direction_specific ? ' in one direction' : ''}.</strong>
    ${n.explanation ? `<span>${e(n.explanation)}</span>` : ''}<p class="muted small">${e(n.header)}</p></div></div>`;
}

function caveatsHtml(j) {
  const list = j.caveats || [];
  if (!list.length) return '';
  return `<ul class="journey__caveats plain-list">${list.map(c =>
    `<li>${icon(c.kind === 'accessibility' ? 'accessible' : c.kind === 'time' ? 'clock' : 'info', { size: 14 })}<span>${e(c.text)}</span></li>`).join('')}</ul>`;
}

function candidateChips(i, field, names, stations) {
  return `<div class="chips" role="group" aria-label="Choose the ${e(ROLE[field])}">${names.map(n => {
    const s = (stations || []).find(x => x.name === n);
    const lines = s ? s.lines.map(l => lineBadge(l, { size: 'sm' })).join('') : '';
    return `<button type="button" class="chip chip--btn" data-journey="${i}" data-journey-action="pick" data-field="${field}" data-name="${e(n)}">${icon('pin', { size: 14 })}${e(n)}${lines}</button>`;
  }).join('')}</div>`;
}

function optionsHtml(j) {
  const routes = sortTrips(j.routes || [], j.sort || 'arrival');
  const badges = tripBadges(routes);
  const card = (r, idx) => tripOptionCard(r, idx, {
    badges: badges[idx], open: false, note: arrivalNote(r, routes),
    alerts: Array.isArray(r.alerts) ? r.alerts.map(tripAlertText) : [],
  });
  const [best, ...rest] = routes;
  return `${card(best, 0)}${rest.length ? `<details class="journey__more">
      <summary>${icon('chevronDown', { size: 14 })}${rest.length} more option${rest.length > 1 ? 's' : ''}</summary>
      <div class="stack">${rest.map((r, k) => card(r, k + 1)).join('')}</div>
    </details>` : ''}`;
}

/**
 * Card HTML for message `i`. `stations` (optional) adds line badges to candidate chips.
 * Actions are rendered as buttons carrying data-journey-action:
 * refresh | reverse | planner | pick | locate | similar.
 */
export function journeyCardHtml(j, i, { stations = [] } = {}) {
  if (!j) return '';
  const code = j.code;
  const ends = journeyEndpoints(j);
  const planner = actionButton(i, 'planner', 'Edit in Trip Planner', 'route');
  const route = ends.origin && ends.destination
    ? `<p class="journey__ends"><span>${e(ends.origin.name)}</span>${icon('chevronRight', { size: 14 })}<span>${e(ends.destination.name)}</span></p>` : '';

  if (code === 'ambiguous') {
    return `<div class="msg-card journey" data-code="${code}">${(j.ambiguous || []).map(f =>
      candidateChips(i, f, (j.candidates || {})[f] || [], stations)).join('')}</div>`;
  }
  if (code === 'needs_origin') {
    return `<div class="msg-card journey" data-code="${code}"><div class="btn-row">${actionButton(i, 'locate', 'Use my location', 'locate')}</div></div>`;
  }
  if (code === 'needs_destination' || code === 'same_place') return '';
  if (code === 'unresolved') {
    return `<div class="msg-card journey" data-code="${code}"><div class="btn-row">${planner}<a class="btn btn--ghost btn--sm" href="#/stations">${icon('search', { size: 14 })}Browse stations</a></div></div>`;
  }
  if (code === 'routing_unavailable') {
    return `<div class="msg-card msg-card--warn journey" data-code="${code}" role="status">${route}
      <p class="small">No routes are shown because the routing service didn’t answer.</p>
      <div class="btn-row">${actionButton(i, 'refresh', 'Try again', 'refresh')}${planner}</div></div>`;
  }
  if (code === 'no_itineraries') {
    return `<div class="msg-card journey" data-code="${code}">${route}${(j.notices || []).map(noticeHtml).join('')}${caveatsHtml(j)}
      <div class="btn-row">${actionButton(i, 'refresh', 'Try again', 'refresh')}${planner}<a class="btn btn--ghost btn--sm" href="#/alerts">${icon('alert', { size: 14 })}Service alerts</a></div></div>`;
  }
  if (code !== 'ok' || !(j.routes || []).length) return '';

  const similar = (j.destination?.similar || []).slice(0, 1).map(n =>
    actionButton(i, 'similar', `Did you mean ${n}?`, 'pin', `data-field="destination" data-name="${e(n)}"`)).join('')
    + (j.origin?.similar || []).slice(0, 1).map(n =>
      actionButton(i, 'similar', `Start from ${n} instead?`, 'pin', `data-field="origin" data-name="${e(n)}"`)).join('');
  const source = `${PROVIDERS[j.provider] || 'Routing service'} · checked ${formatClock(j.checked_at)}`;
  return `<div class="msg-card journey" data-code="ok" aria-label="Journey options">
    ${route}
    ${(j.notices || []).map(noticeHtml).join('')}
    ${caveatsHtml(j)}
    ${optionsHtml(j)}
    <div class="msg-card__foot">
      <span class="muted small">${e(source)} · times are estimates</span>
      <div class="btn-row">${actionButton(i, 'refresh', 'Refresh', 'refresh')}${actionButton(i, 'reverse', 'Return trip', 'swap')}${planner}${similar}</div>
    </div>
  </div>`;
}

/** The chat API action for a card button, or null for actions handled locally. */
export function journeyAction(j, kind, { field, name, location } = {}) {
  const { origin, destination } = journeyEndpoints(j);
  const sort = j.sort && j.sort !== 'arrival' ? j.sort : undefined;
  // Keep what the rider asked for ("by 5pm", "step-free"), so the reply still says it wasn't applied.
  const extra = { sort, ...(j.requirements ? { requirements: j.requirements } : {}) };
  if (kind === 'refresh' && origin && destination) return { type: 'trip', kind, origin, destination, ...extra };
  if (kind === 'reverse' && origin && destination) return { type: 'trip', kind, origin: destination, destination: origin, ...extra };
  if ((kind === 'pick' || kind === 'similar') && name) {
    const other = field === 'origin' ? destination : origin;
    if (!other) return null;
    const chosen = { name };
    return field === 'origin' ? { type: 'trip', kind, origin: chosen, destination: other, ...extra } : { type: 'trip', kind, origin: other, destination: chosen, ...extra };
  }
  if (kind === 'locate' && location && destination) {
    return { type: 'trip', kind, origin: { name: 'Your location', lat: location.lat, lon: location.lon }, destination, ...extra };
  }
  return null;
}

/** Label for the user's side of the conversation when they press a card button. */
export function actionLabel(action) {
  if (!action) return '';
  if (action.kind === 'refresh') return `Refresh: ${action.origin.name} → ${action.destination.name}`;
  return `${action.origin.name} → ${action.destination.name}`;
}
