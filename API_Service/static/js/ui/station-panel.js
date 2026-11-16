// Station details (shared by every view): served lines, accessibility, live
// arrivals grouped by line and direction, and active alerts.

import { api } from '../core/api.js';
import { store } from '../core/store.js';
import { navigate } from '../core/router.js';
import { escapeHtml as e, formatClockSeconds } from '../core/format.js';
import { openPanel, updatePanelBody, isPanelOpen } from './panel.js';
import { lineBadge, arrivalRow, alertCard, emptyState, errorState, skeleton, endsAt } from './components.js';
import { icon } from './icons.js';
import { SUBWAY_IDS } from '../core/constants.js';

const REFRESH_MS = 20000;
let timer = null;
let inflight = null;

export const MODE_OF = { 0: 'subway', 1: 'subway', 2: 'commuter_rail', 3: 'bus', 4: 'ferry' };
const MODE_ORDER = ['subway', 'commuter_rail', 'bus', 'ferry'];
const MODE_NAMES = { all: 'All', subway: 'Subway', commuter_rail: 'Commuter Rail', bus: 'Bus', ferry: 'Ferry' };

export function modeOf(p) {
  return MODE_OF[p.route_type] || (String(p.route_id).startsWith('CR-') ? 'commuter_rail' : SUBWAY_IDS.includes(p.route_id) ? 'subway' : 'bus');
}

/** Groups by line + direction. Subway first, then Commuter Rail, bus, ferry; each by soonest trip. */
export function groupArrivals(predictions) {
  const groups = new Map();
  for (const p of predictions) {
    const key = `${p.route_id}|${p.direction_id}`;
    if (!groups.has(key)) groups.set(key, { route_id: p.route_id, route_name: p.route_name, route_short_name: p.route_short_name, mode: modeOf(p), direction_id: p.direction_id, platforms: new Set(), headsigns: new Set(), items: [] });
    const g = groups.get(key);
    if (p.headsign) g.headsigns.add(p.headsign);
    if (p.platform_name) g.platforms.add(p.platform_name);
    g.items.push(p);
  }
  // Items are already time-sorted by the API, so insertion order is "soonest first" within a mode.
  return [...groups.values()].sort((x, y) => MODE_ORDER.indexOf(x.mode) - MODE_ORDER.indexOf(y.mode));
}

/** Modes with predictions at this station, in display order. */
export function modesPresent(predictions) {
  const set = new Set(predictions.map(modeOf));
  return MODE_ORDER.filter(m => set.has(m));
}

/** Board heading: where trains go (headsigns), never the platform; trips ending here say so. */
export function groupHeading(g, stationName) {
  const dests = [...g.headsigns].filter(h => !endsAt(h, stationName));
  const dest = dests.length ? `To ${dests.join(' / ')}` : (g.headsigns.size ? `Arriving at ${stationName} (ends here)` : 'Upcoming trips');
  const platforms = [...g.platforms];
  // Subway platforms are named after their direction ("Ashmont/Braintree"): don't repeat it.
  const words = s => String(s).toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
  const destWords = new Set(dests.flatMap(words));
  const redundant = pl => words(pl).every(w => destWords.has(w));
  const platform = platforms.length === 1 && !redundant(platforms[0]) ? platforms[0] : '';
  return { dest, platform, rowPlatform: platforms.length > 1 };
}

let modeFilter = { station: null, mode: 'all' };

function accessibilityText(code) {
  if (code === 1) return `<span class="chip chip--ok">${icon('accessible', { size: 14 })}Accessible station</span>`;
  if (code === 2) return `<span class="chip chip--muted">${icon('accessible', { size: 14 })}Not accessible</span>`;
  return '';
}

function render(data) {
  const { station, predictions = [], alerts = [] } = data;
  const service = alerts.filter(a => a.category === 'service');
  const access = alerts.filter(a => a.category === 'accessibility');
  if (modeFilter.station !== station.id) modeFilter = { station: station.id, mode: 'all' };
  const modes = modesPresent(predictions);
  const shown = modeFilter.mode === 'all' ? predictions : predictions.filter(p => modeOf(p) === modeFilter.mode);
  const groups = groupArrivals(shown);
  const filters = modes.length > 1
    ? `<div class="segmented segmented--scroll board-filter" role="group" aria-label="Filter departures by mode">${['all', ...modes].map(m =>
        `<button class="segmented__btn" data-mode="${m}" aria-pressed="${m === modeFilter.mode}">${e(MODE_NAMES[m])}<span class="count tabular">${m === 'all' ? predictions.length : predictions.filter(p => modeOf(p) === m).length}</span></button>`).join('')}</div>`
    : '';

  let arrivals;
  if (groups.length) {
    arrivals = groups.map(g => {
      const h = groupHeading(g, station.name);
      return `<section class="board" aria-label="${e(g.route_name)}: ${e(h.dest)}">
        <header class="board__head">${lineBadge(g.route_id, { size: 'sm', route: { short_name: g.route_short_name, name: g.route_name } })}
          <span class="board__dir">${e(h.dest)}</span>${h.platform ? `<span class="board__platform">${e(h.platform)}</span>` : ''}</header>
        <ul class="arrivals">${g.items.slice(0, 3).map(p => arrivalRow(p, { platform: h.rowPlatform, stationName: station.name })).join('')}</ul>
      </section>`;
    }).join('');
  } else if (modeFilter.mode !== 'all') {
    arrivals = emptyState({ icon: 'clock', title: `No ${MODE_NAMES[modeFilter.mode]} departures reported right now`, body: 'Other services at this station still have predictions.', action: '<button class="btn btn--secondary btn--sm" data-mode="all">Show all departures</button>' });
  } else {
    arrivals = emptyState({ icon: 'clock', title: 'No predictions right now', body: 'The MBTA isn’t reporting upcoming trips for this station. Service may have ended for the night.' });
  }

  return `
    <div class="station-detail">
      <div class="station-detail__lines">${(station.lines || []).map(l => lineBadge(l)).join('')}${accessibilityText(station.wheelchair_boarding)}</div>
      <div class="btn-row">
        ${document.body.dataset.route === 'map' ? '' : `<button class="btn btn--secondary btn--sm" data-act="map">${icon('map', { size: 16 })}View on map</button>`}
        <button class="btn btn--secondary btn--sm" data-act="from">${icon('route', { size: 16 })}From here</button>
        <button class="btn btn--secondary btn--sm" data-act="to">To here</button>
      </div>
      ${service.length ? `<div class="stack-sm">${service.slice(0, 3).map(a => alertCard(a, { compact: true })).join('')}</div>` : ''}
      <h3 class="subhead">Departures</h3>
      ${filters}
      <div class="boards">${arrivals}</div>
      ${access.length ? `<details class="disclosure"><summary>${icon('accessible', { size: 16 })}${access.length} accessibility alert${access.length > 1 ? 's' : ''}</summary>
        <ul class="plain-list">${access.map(a => `<li>${e(a.header)}</li>`).join('')}</ul></details>` : ''}
      <p class="fine-print">${icon('clock', { size: 14 })}Updated ${e(formatClockSeconds(data.updated_at || new Date()))} · refreshes every 20 s</p>
    </div>`;
}

function bind(body, station, data) {
  body.querySelectorAll('[data-mode]').forEach(btn => btn.addEventListener('click', () => {
    modeFilter = { station: station.id, mode: btn.dataset.mode };
    const next = updatePanelBody(`station:${station.id}`, render(data));
    if (next) { bind(next, station, data); next.querySelector(`[data-mode="${modeFilter.mode}"]`)?.focus(); }
  }));
  body.querySelector('[data-act="map"]')?.addEventListener('click', () => navigate('map', [], { station: station.id }));
  const plan = which => {
    const s = store.stationById(station.id) || { id: station.id, name: station.name, lat: station.latitude, lon: station.longitude };
    store.set('pendingTrip', { [which]: { name: s.name, id: s.id, lat: s.lat ?? station.latitude, lon: s.lon ?? station.longitude } });
    navigate('trip');
  };
  body.querySelector('[data-act="from"]')?.addEventListener('click', () => plan('origin'));
  body.querySelector('[data-act="to"]')?.addEventListener('click', () => plan('destination'));
  body.querySelector('[data-action="retry"]')?.addEventListener('click', () => load(station.id));
}

async function load(id, { quiet = false } = {}) {
  const key = `station:${id}`;
  inflight?.abort();
  inflight = new AbortController();
  if (!quiet) updatePanelBody(key, skeleton(5, { height: 52 }));
  const { data, error, aborted } = await api.station(id, { signal: inflight.signal });
  if (aborted || !isPanelOpen(key)) return;
  if (error) {
    if (quiet) return; // keep the last good board on a failed background refresh
    const body = updatePanelBody(key, errorState({ title: 'Couldn’t load arrivals', body: error }));
    body?.querySelector('[data-action="retry"]')?.addEventListener('click', () => load(id));
    return;
  }
  const body = updatePanelBody(key, render(data));
  if (body) bind(body, data.station, data);
}

export function openStation(id) {
  const meta = store.stationById(id);
  const key = `station:${id}`;
  store.rememberStation(id);
  openPanel({
    key,
    eyebrow: 'Station',
    title: meta ? meta.name : 'Station',
    html: skeleton(5, { height: 52 }),
    onClose: () => { clearInterval(timer); timer = null; inflight?.abort(); document.dispatchEvent(new CustomEvent('station-closed', { detail: id })); },
  });
  load(id);
  clearInterval(timer);
  timer = setInterval(() => { if (!document.hidden && isPanelOpen(key)) load(id, { quiet: true }); }, REFRESH_MS);
  document.dispatchEvent(new CustomEvent('station-opened', { detail: id }));
}
