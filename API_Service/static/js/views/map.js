// Live Map view. The map DOM and Leaflet instance persist across navigation;
// polling runs only while this view is visible.

import { store } from '../core/store.js';
import { navigate } from '../core/router.js';
import { LINE_GROUPS, routeIdentity } from '../core/constants.js';
import { escapeHtml as e, formatClockSeconds, relativeTime, vehicleStatusText } from '../core/format.js';
import { rank } from '../core/search.js';
import { LiveMap, CR_COLOR, BUS_COLOR } from '../map/live-map.js';
import { icon } from '../ui/icons.js';
import { lineBadge } from '../ui/components.js';
import { combobox } from '../ui/combobox.js';
import { openPanel, updatePanelBody, isPanelOpen, closePanel } from '../ui/panel.js';
import { openStation } from '../ui/station-panel.js';
import { requestLocation } from '../ui/location.js';

let liveMap = null;
let host = null;
let initPromise = null;
let active = false;

function vehicleHtml(v) {
  if (!v) return `<p class="muted">This vehicle is no longer reporting its position.</p>`;
  const id = routeIdentity(v.route?.id, { name: v.route?.name });
  const status = vehicleStatusText(v.current_status, v.stop_name);
  return `<div class="vehicle-detail">
    <div class="vehicle-detail__head">${lineBadge(v.route?.id, { route: { name: v.route?.name } })}<span class="vehicle-detail__dest">${v.headsign ? `To ${e(v.headsign)}` : e(id.name)}</span></div>
    <dl class="facts">
      ${status ? `<div><dt>Status</dt><dd>${e(status)}</dd></div>` : ''}
      <div><dt>Position reported</dt><dd>${v.updated_at ? `${e(formatClockSeconds(v.updated_at))} <span class="muted">(${e(relativeTime(v.updated_at))})</span>` : 'Unknown'}</dd></div>
      <div><dt>Vehicle</dt><dd class="tabular">#${e(v.label || v.id)}</dd></div>
      ${v.bearing != null ? `<div><dt>Heading</dt><dd class="tabular">${e(Math.round(v.bearing))}°</dd></div>` : ''}
    </dl>
    <div class="btn-row"><a class="btn btn--secondary btn--sm" href="#/lines/${encodeURIComponent(v.route?.id || '')}">${icon('lines', { size: 16 })}Open ${e(id.name)}</a></div>
    <p class="fine-print">Positions come from the MBTA and refresh about every 10 seconds. Markers move only when a new position is reported.</p>
  </div>`;
}

function setStatus({ count = 0, updatedAt, error, loading }) {
  const chip = host.querySelector('[data-map-status]');
  if (!chip) return;
  const focus = liveMap?.focusRoute;
  const what = focus ? routeIdentity(focus).name : (liveMap?.groups.has('CR') && liveMap.crRouteIds.length ? 'subway & rail' : 'subway');
  chip.classList.toggle('is-error', Boolean(error));
  chip.classList.toggle('is-loading', Boolean(loading));
  const time = updatedAt ? formatClockSeconds(updatedAt) : '—';
  chip.innerHTML = error
    ? `${icon('wifiOff', { size: 14 })}<span>Couldn’t refresh · showing ${e(time)}</span>`
    : `<span class="live-dot" aria-hidden="true"></span><span><strong class="tabular">${count}</strong> ${what} vehicle${count === 1 ? '' : 's'} · <span class="tabular">${e(time)}</span></span>`;
}

function renderFocusBanner() {
  const el = host.querySelector('[data-focus]');
  const focus = liveMap.focusRoute;
  el.hidden = !focus;
  if (focus) el.innerHTML = `<span>Showing ${lineBadge(focus, { size: 'sm', label: false })} ${e(routeIdentity(focus).name)} only</span><button class="btn btn--ghost btn--sm" data-clear-focus>Show all subway</button>`;
  el.querySelector('[data-clear-focus]')?.addEventListener('click', () => navigate('map'));
}

function buildHost() {
  host = document.getElementById('map-view');
  host.innerHTML = `
    <div class="map-canvas" id="leaflet-map" role="region" aria-label="Live MBTA map"></div>
    <div class="map-top">
      <div class="map-search combo">
        ${icon('search', { size: 18, className: 'combo__icon' })}
        <label class="sr-only" for="map-search">Find a station on the map</label>
        <input id="map-search" class="input input--floating" type="search" placeholder="Find a station" />
      </div>
      <div class="map-status" data-map-status role="status" aria-live="off">Loading live vehicles…</div>
    </div>
    <div class="map-focus" data-focus hidden></div>
    <div class="map-tools" role="toolbar" aria-label="Map tools">
      <button class="map-tool map-tool--label" data-tool="layers" aria-expanded="false" aria-controls="map-layers" aria-label="Layers and line filters">${icon('layers')}<span class="map-tool__text" aria-hidden="true">Layers</span><span class="map-tool__count" data-hidden-count hidden></span></button>
      <button class="map-tool" data-tool="locate" aria-label="Show my location">${icon('locate')}</button>
      <button class="map-tool" data-tool="fit" aria-label="Fit the whole network">${icon('recenter')}</button>
    </div>
    <div class="map-layers" id="map-layers" role="dialog" aria-label="Map layers and line filters" hidden>
      <fieldset><legend>Lines</legend>
        ${LINE_GROUPS.map(g => `<label class="check"><input type="checkbox" data-group="${g.id}" checked><span class="swatch" style="--c:${routeIdentity(g.routes[0]).bright}"></span>${e(g.name)}</label>`).join('')}
        <label class="check"><input type="checkbox" data-group="CR" checked><span class="swatch" style="--c:${CR_COLOR}"></span>Commuter Rail</label>
        <label class="check"><input type="checkbox" data-group="Bus" checked><span class="swatch" style="--c:${BUS_COLOR}"></span>Bus routes</label>
      </fieldset>
      <fieldset><legend>Show</legend>
        <label class="check"><input type="checkbox" data-layer="vehicles" checked>Live vehicles</label>
        <label class="check"><input type="checkbox" data-layer="stations" checked>Stations</label>
        <label class="check"><input type="checkbox" data-layer="lines" checked>Line paths</label>
      </fieldset>
    </div>
    <div class="map-loading" data-map-loading>Loading map…</div>`;

  const layers = host.querySelector('#map-layers');
  const layersBtn = host.querySelector('[data-tool="layers"]');
  const setLayersOpen = open => {
    layers.hidden = !open;
    layersBtn.setAttribute('aria-expanded', String(open));
    if (open) layers.querySelector('input')?.focus();
  };
  layersBtn.addEventListener('click', () => setLayersOpen(layers.hidden));
  layers.addEventListener('keydown', ev => { if (ev.key === 'Escape') { ev.stopPropagation(); setLayersOpen(false); layersBtn.focus(); } });
  document.addEventListener('pointerdown', ev => {
    if (!layers.hidden && !layers.contains(ev.target) && !layersBtn.contains(ev.target)) setLayersOpen(false);
  });
  // A count on the Layers button when something is switched off, so hidden filters are visible.
  const updateHiddenCount = () => {
    const off = [...layers.querySelectorAll('input[type="checkbox"]')].filter(i => !i.checked).length;
    const badge = host.querySelector('[data-hidden-count]');
    badge.hidden = !off;
    badge.textContent = off ? `${off} off` : '';
    layersBtn.setAttribute('aria-label', off ? `Layers and line filters, ${off} turned off` : 'Layers and line filters');
  };
  layers.addEventListener('change', ev => {
    const t = ev.target;
    if (t.dataset.group) liveMap.setGroup(t.dataset.group, t.checked);
    if (t.dataset.layer) liveMap.setLayer(t.dataset.layer, t.checked);
    updateHiddenCount();
  });
  host.querySelector('[data-tool="fit"]').addEventListener('click', () => liveMap.focusRoute ? liveMap.fitRoute(liveMap.focusRoute) : liveMap.fitNetwork());
  host.querySelector('[data-tool="locate"]').addEventListener('click', async () => {
    const loc = await requestLocation({ quiet: true });
    if (loc) liveMap.showUser(loc.lat, loc.lon);
  });

  combobox(host.querySelector('#map-search'), {
    source: q => rank(store.get('stations'), q, s => s.name, 8),
    render: s => `<span class="opt-name">${e(s.name)}</span><span class="opt-lines">${s.lines.map(l => lineBadge(l, { size: 'sm' })).join('')}</span>`,
    onSelect: (s, input) => { input.value = ''; input.blur(); selectStation(s.id, true); },
    emptyText: 'No stations match',
  });

  document.addEventListener('vehicle-updated', ev => {
    if (isPanelOpen('vehicle')) updatePanelBody('vehicle', vehicleHtml(ev.detail));
  });
  document.addEventListener('station-closed', ev => { if (liveMap && liveMap.selectedStation === ev.detail) liveMap.selectStation(null); });
  document.addEventListener('visibilitychange', () => {
    if (!active || !liveMap) return;
    if (document.hidden) liveMap.stop(); else liveMap.start();
  });
}

function selectStation(id, fly = false) {
  liveMap.selectVehicle(null);
  liveMap.selectStation(id, { fly });
  openStation(id);
}

async function ensureMap() {
  if (!host) buildHost();
  if (!initPromise) {
    liveMap = new LiveMap(host.querySelector('#leaflet-map'), {
      onStatus: setStatus,
      onStationSelect: id => selectStation(id),
      onVehicleSelect: v => {
        liveMap.selectStation(null);
        openPanel({ key: 'vehicle', eyebrow: 'Live vehicle', title: routeIdentity(v.route?.id, { name: v.route?.name }).name, html: vehicleHtml(v), onClose: () => liveMap.selectVehicle(null) });
      },
    });
    initPromise = liveMap.init().then(() => {
      host.querySelector('[data-map-loading]')?.remove();
      const loc = store.get('location');
      if (loc && loc.lat) liveMap.showUser(loc.lat, loc.lon, { fly: false });
    }).catch(err => {
      initPromise = null;
      host.querySelector('[data-map-loading]').innerHTML = `${icon('wifiOff')}<span>${e(err.message || 'The map couldn’t load.')}</span>`;
      throw err;
    });
  }
  return initPromise;
}

export const mapView = {
  async enter({ query }) {
    active = true;
    document.body.classList.add('route-map');
    try { await ensureMap(); } catch { return; }
    if (!active) return;
    liveMap.invalidate();
    await liveMap.setFocus(query.route || null);
    renderFocusBanner();
    liveMap.start();
    if (query.station) setTimeout(() => selectStation(query.station, true), 150);
  },
  leave() {
    active = false;
    document.body.classList.remove('route-map');
    liveMap?.stop();
    if (isPanelOpen('vehicle')) closePanel();
  },
  showUserOnMap(loc) { if (liveMap && loc) liveMap.showUser(loc.lat, loc.lon); },
};
