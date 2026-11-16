// Overview: "What's happening on the MBTA, and how do I get where I'm going?"

import { api } from '../core/api.js';
import { store } from '../core/store.js';
import { navigate } from '../core/router.js';
import { SUBWAY_IDS, LINES, QUICK_STATIONS } from '../core/constants.js';
import { escapeHtml as e, greeting, formatDay } from '../core/format.js';
import { filterAlerts } from '../core/alerts.js';
import { tripFormHtml, bindTripForm } from '../ui/trip-form.js';
import { lineBadge, arrivalRow, alertCard, emptyState, errorState, skeleton, sectionHeader } from '../ui/components.js';
import { icon, modeIcon } from '../ui/icons.js';
import { openStation } from '../ui/station-panel.js';
import { requestLocation } from '../ui/location.js';

const REFRESH_MS = 30000;
let timer = null;
let controller = null;
let root = null;

const STATUS_TEXT = { alert: 'Major disruption', warning: 'Service alert', normal: 'No current alerts' };

function statusRow(id, s) {
  const status = s ? s.status : 'unknown';
  const label = s ? STATUS_TEXT[status] : 'Status unavailable';
  return `<a class="status-row status-row--${status}" href="#/lines/${encodeURIComponent(id)}">
    ${lineBadge(id)}
    <span class="status-row__name">${e(LINES[id].name)}</span>
    <span class="status-row__state"><span class="state-dot" aria-hidden="true"></span>${e(label)}</span>
    ${s && s.headline ? `<span class="status-row__headline">${e(s.headline)}</span>` : ''}
  </a>`;
}

function renderStatus(dashboard, error) {
  const el = root?.querySelector('[data-status]');
  if (!el) return;
  if (error && !dashboard) {
    el.innerHTML = errorState({ title: 'Line status unavailable', body: error });
    el.querySelector('[data-action="retry"]')?.addEventListener('click', refresh);
    return;
  }
  const st = dashboard.subway_status;
  const disrupted = SUBWAY_IDS.filter(id => st[id] && st[id].status !== 'normal');
  const summary = root.querySelector('[data-summary]');
  if (summary) {
    summary.innerHTML = disrupted.length
      ? `${icon('alert', { size: 16 })}<span><strong>${disrupted.length} of ${SUBWAY_IDS.length}</strong> subway lines have active service alerts</span>`
      : `${icon('check', { size: 16 })}<span>No active service alerts on the subway</span>`;
    summary.className = `summary-pill ${disrupted.length ? 'summary-pill--warn' : 'summary-pill--ok'}`;
  }
  el.innerHTML = `<div class="status-list">${SUBWAY_IDS.map(id => statusRow(id, st[id])).join('')}</div>`;
}

function renderAlerts(alerts, error) {
  const el = root?.querySelector('[data-alerts]');
  if (!el) return;
  if (error && !alerts) { el.innerHTML = errorState({ title: 'Alerts unavailable', body: error }); el.querySelector('[data-action="retry"]')?.addEventListener('click', refresh); return; }
  const top = filterAlerts(alerts, 'subway').filter(a => a.active_now).slice(0, 3);
  el.innerHTML = top.length ? top.map(a => alertCard(a, { compact: true })).join('') :
    emptyState({ icon: 'check', title: 'No active subway alerts', body: 'Bus, rail and ferry alerts are on the Alerts page.' });
}

function stationsToShow() {
  const ids = [];
  const loc = store.get('location');
  if (loc && loc.stop_id) ids.push({ id: loc.stop_id, near: true });
  for (const id of store.get('recentStations')) if (!ids.some(x => x.id === id)) ids.push({ id });
  return ids.slice(0, 2);
}

async function renderBoards(signal) {
  const el = root?.querySelector('[data-boards]');
  if (!el) return;
  const picks = stationsToShow();
  if (!picks.length) {
    el.innerHTML = `<p class="muted">Pick a station to see live arrivals here.</p>
      <div class="chips">${QUICK_STATIONS.map(id => store.stationById(id)).filter(Boolean).map(s =>
        `<button class="chip chip--btn" data-station="${e(s.id)}">${e(s.name)}</button>`).join('')}</div>
      <button class="btn btn--ghost btn--sm" data-locate>${icon('locate', { size: 16 })}Use my location</button>`;
    el.querySelector('[data-locate]').addEventListener('click', async () => { if (await requestLocation()) renderBoards(); });
    return;
  }
  if (!el.querySelector('.mini-board')) el.innerHTML = skeleton(2, { height: 120 });
  const results = await Promise.all(picks.map(p => api.station(p.id, { signal })));
  if (signal?.aborted || !root) return;
  el.innerHTML = picks.map((p, i) => {
    const { data, error } = results[i];
    const name = store.stationById(p.id)?.name || data?.station?.name || 'Station';
    const preds = data ? data.predictions.slice(0, 3) : [];
    return `<article class="mini-board">
      <header class="mini-board__head">
        <button class="mini-board__title" data-station="${e(p.id)}">${p.near ? `<span class="chip chip--accent">Near you</span>` : ''}<span>${e(name)}</span>${icon('chevronRight', { size: 16 })}</button>
        <span class="mini-board__lines">${(store.stationById(p.id)?.lines || []).map(l => lineBadge(l, { size: 'sm' })).join('')}</span>
      </header>
      ${error ? `<p class="muted small">${e(error)}</p>` : preds.length ? `<ul class="arrivals">${preds.map(pr => arrivalRow(pr, { showRoute: true, stationName: data.station?.name || name })).join('')}</ul>` : `<p class="muted small">No predictions right now.</p>`}
    </article>`;
  }).join('');
}

async function refresh() {
  controller?.abort();
  controller = new AbortController();
  const signal = controller.signal;
  const [dash, alerts] = await Promise.all([api.dashboard({ signal }), api.alerts({ signal })]);
  if (signal.aborted || !root) return;
  if (dash.data) store.set('dashboard', dash.data);
  if (alerts.data) store.set('alerts', alerts.data.alerts);
  renderStatus(dash.data || store.get('dashboard'), dash.error);
  renderAlerts(alerts.data ? alerts.data.alerts : store.get('alerts'), alerts.error);
  renderBoards(signal);
}

export const overviewView = {
  enter(el) {
    root = el;
    const now = new Date();
    el.innerHTML = `
      <div class="page page--overview">
        <header class="hero">
          <div>
            <p class="hero__date">${e(formatDay(now))}</p>
            <h1 class="hero__title">${e(greeting(now))}, Boston</h1>
          </div>
          <div class="summary-pill" data-summary role="status">${icon('clock', { size: 16 })}<span>Checking service…</span></div>
        </header>

        <nav class="quick" aria-label="Quick access">
          <a class="quick__item" href="#/map">${icon('map', { size: 18 })}Live map</a>
          <a class="quick__item" href="#/lines?mode=subway">${modeIcon('subway', { size: 18 })}Subway</a>
          <a class="quick__item" href="#/lines?mode=bus">${modeIcon('bus', { size: 18 })}Bus</a>
          <a class="quick__item" href="#/lines?mode=commuter_rail">${modeIcon('commuter_rail', { size: 18 })}Commuter Rail</a>
          <a class="quick__item" href="#/lines?mode=ferry">${modeIcon('ferry', { size: 18 })}Ferry</a>
          <a class="quick__item quick__item--ai" href="#/ai">${icon('sparkle', { size: 18 })}Ask Charlie</a>
        </nav>

        <div class="overview-grid">
          <section class="card card--pad card--plan" aria-labelledby="plan-h">
            ${sectionHeader('Plan a journey', { id: 'plan-h' })}
            ${tripFormHtml()}
          </section>

          <section class="card card--pad" aria-labelledby="boards-h">
            ${sectionHeader('Your stations', { id: 'boards-h', action: '<a class="link small" href="#/stations">All stations</a>' })}
            <div data-boards>${skeleton(2, { height: 120 })}</div>
          </section>

          <section class="card card--pad" aria-labelledby="status-h">
            ${sectionHeader('Subway status', { id: 'status-h', action: '<a class="link small" href="#/lines">All lines</a>' })}
            <div data-status>${skeleton(4, { height: 40 })}</div>
          </section>

          <section class="card card--pad" aria-labelledby="alerts-h">
            ${sectionHeader('Subway alerts now', { id: 'alerts-h', action: '<a class="link small" href="#/alerts">All alerts</a>' })}
            <div class="stack-sm" data-alerts>${skeleton(3, { height: 72 })}</div>
          </section>
        </div>
      </div>`;

    bindTripForm(el.querySelector('[data-trip-form]'), {
      onSubmit: q => { store.set('pendingTrip', q); navigate('trip'); },
    });
    el.addEventListener('click', ev => {
      const st = ev.target.closest('[data-station]');
      if (st) openStation(st.dataset.station);
    });

    if (store.get('dashboard')) renderStatus(store.get('dashboard'));
    if (store.get('alerts')) renderAlerts(store.get('alerts'));
    refresh();
    timer = setInterval(() => { if (!document.hidden) refresh(); }, REFRESH_MS);
    this.unsub = store.subscribe(key => { if (key === 'location' || key === 'recentStations') renderBoards(); });
  },
  leave() {
    clearInterval(timer); timer = null;
    controller?.abort();
    this.unsub?.();
    root = null;
  },
};
