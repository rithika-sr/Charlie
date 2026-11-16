// App entry: shell wiring, routing, global search, freshness and location.

import { api } from './core/api.js';
import { store } from './core/store.js';
import { parseHash, navigate } from './core/router.js';
import { routeIdentity } from './core/constants.js';
import { escapeHtml as e, formatClock, relativeTime } from './core/format.js';
import { searchResults } from './core/search.js';
import { filterAlerts } from './core/alerts.js';
import { initPanel, closePanel } from './ui/panel.js';
import { combobox } from './ui/combobox.js';
import { lineBadge } from './ui/components.js';
import { icon } from './ui/icons.js';
import { openStation } from './ui/station-panel.js';
import { requestLocation } from './ui/location.js';
import { toast } from './ui/toast.js';
import { overviewView } from './views/overview.js';
import { mapView } from './views/map.js';
import { tripView } from './views/trip.js';
import { stationsView } from './views/stations.js';
import { linesView } from './views/lines.js';
import { alertsView } from './views/alerts.js';
import { assistantView } from './views/assistant.js';

const VIEWS = {
  overview: { view: overviewView, title: 'Overview' },
  map: { view: mapView, title: 'Live Map' },
  trip: { view: tripView, title: 'Trip Planner' },
  stations: { view: stationsView, title: 'Stations' },
  lines: { view: linesView, title: 'Lines & Routes' },
  alerts: { view: alertsView, title: 'Service Alerts' },
  ai: { view: assistantView, title: 'Charlie AI' },
};

const viewEl = document.getElementById('view');
let current = null;

function setActiveNav(name) {
  document.querySelectorAll('[data-nav]').forEach(a => {
    if (a.dataset.nav === name) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  });
}

async function onRoute() {
  const route = parseHash();
  const entry = VIEWS[route.name];
  if (current) {
    current.view.leave?.();
    if (!(route.name === 'map' && route.query.station)) closePanel();
  }
  current = entry;
  setActiveNav(route.name);
  document.title = route.name === 'overview' ? 'Charlie — Boston transit, live' : `${entry.title} · Charlie`;
  document.body.dataset.route = route.name;
  document.body.classList.remove('search-open');

  if (route.name === 'map') {
    viewEl.hidden = true;
    viewEl.innerHTML = '';
    document.getElementById('map-view').hidden = false;
    await mapView.enter(route);
  } else {
    document.getElementById('map-view').hidden = true;
    viewEl.hidden = false;
    viewEl.scrollTop = 0;
    window.scrollTo(0, 0);
    entry.view.enter(viewEl, route);
    const h1 = viewEl.querySelector('h1');
    if (h1 && document.activeElement?.closest('[data-nav]')) { h1.tabIndex = -1; h1.focus({ preventScroll: true }); }
  }
}

// ---- Freshness ----------------------------------------------------------
function renderFreshness() {
  const el = document.querySelector('[data-fresh]');
  if (!el) return;
  const last = store.get('lastUpdated');
  const stale = store.get('stale');
  el.classList.toggle('is-stale', stale);
  el.innerHTML = stale
    ? `${icon('wifiOff', { size: 14 })}<span>Offline${last ? ` · ${e(formatClock(last))}` : ''}</span>`
    : `<span class="live-dot" aria-hidden="true"></span><span>Live${last ? ` · <span class="tabular" title="Last data received ${e(relativeTime(last))}">${e(formatClock(last))}</span>` : ''}</span>`;
}

function renderAlertCount() {
  const alerts = store.get('alerts');
  const n = alerts ? filterAlerts(alerts, 'all').filter(a => a.active_now).length : null;
  document.querySelectorAll('[data-alert-count]').forEach(el => {
    el.hidden = !n;
    el.textContent = n > 99 ? '99+' : String(n || '');
    el.setAttribute('aria-label', `${n} active service alerts`);
  });
}

// ---- Global search ------------------------------------------------------
function searchItems(q) {
  return searchResults(q, store.get('stations') || [], store.get('routes'));
}

function searchOption(item) {
  if (item.kind === 'station') {
    return `<span class="opt-icon">${icon('pin', { size: 16 })}</span><span class="opt-name">${e(item.name)}</span><span class="opt-lines">${item.lines.map(l => lineBadge(l, { size: 'sm' })).join('')}</span>`;
  }
  const id = routeIdentity(item.id, item);
  return `<span class="opt-icon">${lineBadge(item.id, { size: 'sm', route: item, label: false })}</span><span class="opt-name">${e(id.mode === 'bus' ? `${item.short_name || item.id} · ${item.name}` : id.name)}</span><span class="opt-sub">${e({ subway: 'Subway', bus: 'Bus', commuter_rail: 'Commuter Rail', ferry: 'Ferry' }[id.mode])}</span>`;
}

function setupSearch() {
  const input = document.getElementById('global-search');
  combobox(input, {
    source: q => searchItems(q),
    render: searchOption,
    emptyText: 'No stations or routes match',
    onSelect: item => {
      input.value = '';
      input.blur();
      document.body.classList.remove('search-open');
      if (item.kind === 'station') openStation(item.id);
      else navigate('lines', [item.id]);
    },
  });
  document.querySelectorAll('[data-open-search]').forEach(b => b.addEventListener('click', () => {
    document.body.classList.add('search-open');
    input.focus();
  }));
  document.querySelector('[data-close-search]')?.addEventListener('click', () => document.body.classList.remove('search-open'));
  input.addEventListener('keydown', ev => { if (ev.key === 'Escape') { document.body.classList.remove('search-open'); input.blur(); } });
  document.addEventListener('keydown', ev => {
    const typing = ev.target.closest('input, textarea, [contenteditable]');
    if (ev.key === '/' && !typing) { ev.preventDefault(); document.body.classList.add('search-open'); input.focus(); }
  });
}

function setupLocation() {
  document.querySelectorAll('[data-locate-global]').forEach(btn => btn.addEventListener('click', async () => {
    const loc = await requestLocation();
    if (loc && parseHash().name === 'map') mapView.showUserOnMap(loc);
  }));
}

async function boot() {
  initPanel();
  setupSearch();
  setupLocation();
  store.subscribe(key => {
    if (key === 'freshness') renderFreshness();
    if (key === 'alerts') renderAlertCount();
  });
  setInterval(renderFreshness, 15000);
  renderFreshness();

  const { data: stations, error } = await api.stations();
  if (stations) store.set('stations', stations);
  else toast(`Station list unavailable: ${error}`, { tone: 'error', timeout: 6000 });

  window.addEventListener('hashchange', onRoute);
  await onRoute();

  // Background data for search and badges.
  api.routes().then(({ data }) => { if (data) store.set('routes', data); });
  if (!store.get('alerts')) api.alerts().then(({ data }) => { if (data) store.set('alerts', data.alerts); });
}

boot();
