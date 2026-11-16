// Lines & Routes: directory per mode, and a line detail page with the station
// sequence, next trains in each direction, alerts and a map shortcut.

import { api } from '../core/api.js';
import { store } from '../core/store.js';
import { navigate } from '../core/router.js';
import { SUBWAY_IDS, LINES, MODES, routeIdentity } from '../core/constants.js';
import { escapeHtml as e, countdown, formatClockSeconds } from '../core/format.js';
import { rank } from '../core/search.js';
import { lineBadge, alertCard, emptyState, errorState, skeleton, sectionHeader } from '../ui/components.js';
import { icon, modeIcon } from '../ui/icons.js';
import { openStation } from '../ui/station-panel.js';

const REFRESH_MS = 30000;
let timer = null;
let controller = null;
let busQuery = '';

const STATUS_TEXT = { alert: 'Major disruption', warning: 'Service alert', normal: 'No current alerts' };

function routesFor(mode) {
  const r = store.get('routes');
  if (!r) return null;
  if (mode === 'subway') return SUBWAY_IDS.map(id => r.subway.find(x => x.id === id) || { id, name: LINES[id].name, direction_destinations: [] });
  return r[mode] || [];
}

function endpoints(route) {
  const d = (route.direction_destinations || []).filter(Boolean);
  return d.length === 2 ? `${d[1]} ↔ ${d[0]}` : d.join(' · ');
}

function directoryRow(route, mode) {
  const dash = store.get('dashboard');
  const s = mode === 'subway' && dash ? dash.subway_status[route.id] : null;
  const alerts = (store.get('alerts') || []).filter(a => a.category === 'service' && a.active_now && a.affected_routes.includes(route.id));
  const status = s ? `<span class="status-tag status-tag--${s.status}"><span class="state-dot" aria-hidden="true"></span>${e(STATUS_TEXT[s.status])}</span>`
    : alerts.length ? `<span class="status-tag status-tag--warning"><span class="state-dot" aria-hidden="true"></span>${alerts.length} alert${alerts.length > 1 ? 's' : ''}</span>` : '';
  const id = routeIdentity(route.id, route);
  const title = mode === 'bus' ? (route.short_name ? `${route.short_name}` : route.id) : id.name;
  return `<a class="route-row" href="#/lines/${encodeURIComponent(route.id)}">
    ${lineBadge(route.id, { route })}
    <span class="route-row__text"><span class="route-row__name">${e(mode === 'bus' ? route.name : title)}</span><span class="route-row__sub">${e(endpoints(route))}</span></span>
    ${status}
    ${icon('chevronRight', { size: 18, className: 'route-row__chev' })}
  </a>`;
}

function renderDirectory(el, mode) {
  const list = el.querySelector('[data-routes]');
  let routes = routesFor(mode);
  if (!routes) { list.innerHTML = skeleton(6, { height: 56 }); return; }
  if (mode === 'bus' && busQuery.trim()) routes = rank(routes, busQuery, r => [r.short_name || r.id, r.name], 200);
  list.innerHTML = routes.length ? routes.map(r => directoryRow(r, mode)).join('')
    : emptyState({ icon: 'search', title: 'No routes match', body: 'Search by number (e.g. 66) or by a place on the route.' });
}

function nextText(preds) {
  if (!preds || !preds.length) return '<span class="muted">—</span>';
  return preds.map(p => `<span class="tabular">${e(countdown(p).label)}</span>`).join('<span class="muted">, </span>');
}

function renderDetail(el, data) {
  const { line, stations = [], vehicles = [], alerts = [] } = data;
  const id = routeIdentity(line.id, { name: line.name, short_name: line.short_name });
  const service = alerts.filter(a => a.category === 'service');
  // Same rule as line status: severity 1–2 are informational notices.
  const disruptions = service.filter(a => a.active_now && (a.severity || 0) >= 3);
  const access = alerts.filter(a => a.category === 'accessibility');
  const dirs = line.direction_destinations || [];
  const names = line.direction_names || [];
  const scheduled = line.type === 2 || line.type === 4;
  const mapLink = `#/map?route=${encodeURIComponent(line.id)}`;

  el.querySelector('[data-detail]').innerHTML = `
    <header class="line-hero" style="--line:${id.color};--line-bright:${id.bright}">
      <a class="back-link" href="#/lines?mode=${e(id.mode)}">${icon('back', { size: 16 })}${e((MODES.find(m => m.id === id.mode) || {}).name || 'Lines')}</a>
      <div class="line-hero__row">
        ${lineBadge(line.id, { size: 'lg', route: { name: line.name, short_name: line.short_name } })}
        <div><h1 class="line-hero__title">${e(line.name)}</h1><p class="muted">${e(dirs.length === 2 ? `${dirs[1]} ↔ ${dirs[0]}` : dirs.join(' · '))}</p></div>
      </div>
      <div class="line-hero__facts">
        <span class="fact"><strong class="tabular">${vehicles.length}</strong> vehicle${vehicles.length === 1 ? '' : 's'} reporting</span>
        <span class="fact"><strong class="tabular">${disruptions.length}</strong> active service alert${disruptions.length === 1 ? '' : 's'}</span>
        ${service.length > disruptions.length ? `<span class="fact"><strong class="tabular">${service.length - disruptions.length}</strong> notice${service.length - disruptions.length === 1 ? '' : 's'} or upcoming</span>` : ''}
        ${access.length ? `<span class="fact"><strong class="tabular">${access.length}</strong> accessibility</span>` : ''}
        <span class="fact muted">Updated ${e(formatClockSeconds(data.updated_at || new Date()))}</span>
      </div>
      <div class="btn-row"><a class="btn btn--secondary btn--sm" href="${mapLink}">${icon('map', { size: 16 })}Show on live map</a></div>
    </header>

    ${service.length ? `<section class="block">${sectionHeader('Service alerts')}<div class="stack-sm">${service.map(a => alertCard(a, { open: a === disruptions[0] })).join('')}</div></section>` : ''}

    <section class="block">
      ${sectionHeader(id.mode === 'bus' ? 'Stops' : 'Stations', { action: dirs.length === 2 ? `<span class="muted small">Next ${e(scheduled ? 'departures' : 'trains')}</span>` : '' })}
      ${scheduled ? `<p class="fine-print">${icon('info', { size: 14 })}${line.type === 4 ? 'Ferries' : 'Commuter Rail trains'} run on a schedule. Live predictions appear here when the MBTA publishes them for upcoming trips.</p>` : ''}
      ${stations.length ? `<ol class="sequence" style="--line:${id.bright}">
        ${dirs.length === 2 ? `<li class="sequence__legend" aria-hidden="true"><span></span><span>${e(names[1] || '')} → ${e(dirs[1])}</span><span>${e(names[0] || '')} → ${e(dirs[0])}</span></li>` : ''}
        ${stations.map(s => `<li class="sequence__stop">
          <button class="sequence__name" data-stop="${e(s.id)}">${e(s.name)}${s.wheelchair_boarding === 1 ? `<span class="sr-only">, accessible</span>${icon('accessible', { size: 14, className: 'muted' })}` : ''}</button>
          <span class="sequence__next" aria-label="Toward ${e(dirs[1] || '')}">${nextText(s.next && s.next['1'])}</span>
          <span class="sequence__next" aria-label="Toward ${e(dirs[0] || '')}">${nextText(s.next && s.next['0'])}</span>
        </li>`).join('')}
      </ol>` : emptyState({ icon: 'pin', title: 'No stop list available' })}
    </section>`;

  el.querySelectorAll('[data-stop]').forEach(btn => btn.addEventListener('click', () => {
    const sid = btn.dataset.stop;
    if (store.stationById(sid)) openStation(sid);
    else navigate('map', [], { route: line.id });
  }));
}

async function loadDetail(el, routeId, { quiet = false } = {}) {
  controller?.abort();
  controller = new AbortController();
  const box = el.querySelector('[data-detail]');
  if (!quiet) box.innerHTML = `<div class="stack">${skeleton(1, { height: 140 })}${skeleton(8, { height: 40 })}</div>`;
  const { data, error, aborted } = await api.line(routeId, { signal: controller.signal });
  if (aborted) return;
  if (error) {
    if (quiet) return;
    box.innerHTML = errorState({ title: 'Couldn’t load this line', body: error });
    box.querySelector('[data-action="retry"]')?.addEventListener('click', () => loadDetail(el, routeId));
    return;
  }
  renderDetail(el, data);
}

export const linesView = {
  async enter(el, { params, query }) {
    clearInterval(timer);
    if (params[0]) {
      el.innerHTML = `<div class="page page--narrow" data-detail></div>`;
      loadDetail(el, params[0]);
      timer = setInterval(() => { if (!document.hidden) loadDetail(el, params[0], { quiet: true }); }, REFRESH_MS);
      return;
    }
    const mode = MODES.some(m => m.id === query.mode) ? query.mode : 'subway';
    el.innerHTML = `
      <div class="page page--narrow">
        <header class="page-head"><h1>Lines &amp; Routes</h1><p class="muted">Every MBTA line with live status, stations and next trains.</p></header>
        <div class="tabs" role="tablist" aria-label="Transit mode">
          ${MODES.map(m => `<a class="tab" role="tab" href="#/lines?mode=${m.id}" aria-selected="${m.id === mode}">${modeIcon(m.id, { size: 16 })}${e(m.name)}</a>`).join('')}
        </div>
        ${mode === 'bus' ? `<div class="search-field"><label class="sr-only" for="bus-q">Search bus routes</label><input id="bus-q" class="input" type="search" placeholder="Search by route number or place (e.g. 66, Harvard)" value="${e(busQuery)}" autocomplete="off"></div>` : ''}
        <div class="card route-list" role="tabpanel" data-routes></div>
      </div>`;
    const input = el.querySelector('#bus-q');
    input?.addEventListener('input', () => { busQuery = input.value; renderDirectory(el, mode); });
    renderDirectory(el, mode);
    if (!store.get('routes')) {
      const { data } = await api.routes();
      if (data) { store.set('routes', data); renderDirectory(el, mode); }
      else el.querySelector('[data-routes]').innerHTML = errorState({ title: 'Couldn’t load routes', retry: false });
    }
    if (!store.get('dashboard')) {
      const { data } = await api.dashboard();
      if (data) { store.set('dashboard', data); renderDirectory(el, mode); }
    }
  },
  leave() { clearInterval(timer); timer = null; controller?.abort(); },
};
