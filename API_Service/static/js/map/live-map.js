// Leaflet controller for the live map. Loaded lazily the first time the map opens.
// Vehicles are diffed by id and updated in place (no marker rebuilds → no flicker).

import { api } from '../core/api.js';
import { store } from '../core/store.js';
import { LINES, LINE_GROUPS, SUBWAY_IDS, BOSTON, routeIdentity } from '../core/constants.js';
import { decodePolyline } from '../core/polyline.js';
import { escapeHtml as e, formatClockSeconds, vehicleStatusText } from '../core/format.js';
import { alertsForRoutes } from '../core/alerts.js';
import { clusterPoints } from '../core/cluster.js';

const LEAFLET_VERSION = '1.9.4';
const REFRESH_MS = 10000;
const ESRI = 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas';

let leafletPromise = null;
export function loadLeaflet() {
  if (window.L) return Promise.resolve(window.L);
  if (!leafletPromise) {
    leafletPromise = new Promise((resolve, reject) => {
      const css = document.createElement('link');
      css.rel = 'stylesheet';
      css.href = `https://unpkg.com/leaflet@${LEAFLET_VERSION}/dist/leaflet.css`;
      document.head.appendChild(css);
      const js = document.createElement('script');
      js.src = `https://unpkg.com/leaflet@${LEAFLET_VERSION}/dist/leaflet.js`;
      js.onload = () => resolve(window.L);
      js.onerror = () => { leafletPromise = null; reject(new Error('Map library failed to load')); };
      document.head.appendChild(js);
    });
  }
  return leafletPromise;
}

const groupOf = routeId => (LINES[routeId] ? LINES[routeId].group : String(routeId || '').startsWith('CR-') ? 'CR' : null);

// Map-only layers beyond the subway line groups.
export const CR_COLOR = '#B05AA4';      // Commuter Rail purple, lifted for the dark map
export const BUS_COLOR = '#FFC72C';     // MBTA bus yellow
const SILVER_COLOR = '#A3ADB5';
const BUS_STATUS_TTL = 10000;
// At network zoom (≤ 12) nearby vehicles on the same line merge into one count marker.
export const CLUSTER_MAX_ZOOM = 12;
const CLUSTER_RADIUS = 26;
const GROUP_NAMES = { Red: 'Red Line', Orange: 'Orange Line', Blue: 'Blue Line', Green: 'Green Line', Mattapan: 'Mattapan Trolley', CR: 'Commuter Rail' };

const VEHICLE_SVG = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2.5 19.5 20 12 16.2 4.5 20z"/></svg>';

export class LiveMap {
  constructor(el, { onVehicleSelect, onStationSelect, onStatus } = {}) {
    this.el = el;
    this.onVehicleSelect = onVehicleSelect || (() => {});
    this.onStationSelect = onStationSelect || (() => {});
    this.onStatus = onStatus || (() => {});
    this.vehicles = new Map();      // id -> { marker, data }
    this.stationMarkers = new Map();
    this.shapeLayers = new Map();   // routeId -> [polylines]
    this.groups = new Set([...LINE_GROUPS.map(g => g.id), 'CR', 'Bus']);
    this.modeLines = { CR: new Map(), Bus: new Map() };   // routeId -> [polylines]
    this.busMeta = new Map();                             // routeId -> { short_name, long_name, color }
    this.crRouteIds = [];
    this.busStatus = new Map();                           // routeId -> { at, count, vehicles, error }
    this.busVehicles = new Map();                         // id -> marker (hovered/pinned route only)
    this.hoverBusId = null;
    this.pinnedBusId = null;
    this.layers = { vehicles: true, stations: true, lines: true };
    this.focusRoute = null;
    this.selectedVehicle = null;
    this.selectedStation = null;
    this.timer = null;
    this.seq = 0;
    this.controller = null;
    this.lastUpdate = null;
  }

  async init() {
    const L = await loadLeaflet();
    this.L = L;
    const map = L.map(this.el, { zoomControl: false, minZoom: 9, maxZoom: 16, worldCopyJump: false, keyboard: true });
    this.map = map;
    map.setView(BOSTON.center, BOSTON.zoom);
    L.control.zoom({ position: 'bottomright' }).addTo(map);
    L.tileLayer(`${ESRI}/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}`, {
      maxZoom: 16, maxNativeZoom: 16,
      attribution: 'Tiles &copy; <a href="https://www.esri.com" target="_blank" rel="noopener">Esri</a> &mdash; Esri, HERE, Garmin, FAO, NOAA, USGS, &copy; OpenStreetMap contributors · Data: <a href="https://www.mbta.com/developers" target="_blank" rel="noopener">MBTA</a>',
    }).addTo(map);
    // Place-name labels. Esri only serves real label tiles up to z13 here (higher levels are
    // placeholders, which made Leaflet stretch the z13 tile); the base layer has street names.
    L.tileLayer(`${ESRI}/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}`, { maxZoom: 13, maxNativeZoom: 13, pane: 'overlayPane' }).addTo(map);

    map.createPane('busRoutes').style.zIndex = 360;
    map.createPane('railRoutes').style.zIndex = 370;
    map.createPane('routes').style.zIndex = 380;
    // Canvas renderers keep 300+ bus paths fast; `tolerance` widens the hover target.
    this.busRenderer = L.canvas({ pane: 'busRoutes', tolerance: 6 });
    this.railRenderer = L.canvas({ pane: 'railRoutes' });
    // Station pins sit above vehicle markers: a click aimed at a station opens the station.
    // Vehicles stay reachable around the (smaller) pin, by keyboard, and as clusters.
    map.createPane('vehicles').style.zIndex = 600;
    map.createPane('stations').style.zIndex = 650;
    map.getPane('tooltipPane').style.zIndex = 700;   // keep hover cards above vehicle markers
    this.shapeGroup = L.layerGroup().addTo(map);
    this.stationGroup = L.layerGroup().addTo(map);
    this.vehicleGroup = L.layerGroup().addTo(map);
    this.clusterGroup = L.layerGroup().addTo(map);
    this.clustered = new Set();
    this.userGroup = L.layerGroup().addTo(map);
    this.crGroup = L.layerGroup().addTo(map);
    this.busGroup = L.layerGroup().addTo(map);
    this.busVehicleGroup = L.layerGroup().addTo(map);
    this.busTip = L.tooltip({ direction: 'top', offset: [0, -10], className: 'bus-tip', opacity: 1 });
    map.on('click', () => this.unpinBus());

    map.on('zoomend', () => { this.applyZoomClass(); this.refreshVehicles(); });
    this.applyZoomClass();
    await Promise.all([this.drawShapes(SUBWAY_IDS), this.drawStations()]);
    // Commuter Rail and bus paths load after the subway is on screen.
    this.drawModeShapes();
  }

  async drawModeShapes() {
    const [cr, bus] = await Promise.all([api.modeShapes('commuter_rail'), api.modeShapes('bus')]);
    if (cr.data) {
      for (const s of cr.data.shapes) {
        const line = this.L.polyline(decodePolyline(s.polyline), {
          renderer: this.railRenderer, color: CR_COLOR, weight: 3, opacity: 0.8, lineCap: 'round', lineJoin: 'round', interactive: false,
        });
        if (!this.modeLines.CR.has(s.route_id)) this.modeLines.CR.set(s.route_id, []);
        this.modeLines.CR.get(s.route_id).push(line);
      }
      this.crRouteIds = [...this.modeLines.CR.keys()];
      if (this.timer) this.tick();   // start showing Commuter Rail trains
    }
    if (bus.data) {
      for (const s of bus.data.shapes) {
        const silver = /^SL/i.test(s.short_name);
        const line = this.L.polyline(decodePolyline(s.polyline), {
          renderer: this.busRenderer, color: silver ? SILVER_COLOR : BUS_COLOR, weight: 2, opacity: 0.22, lineCap: 'round', lineJoin: 'round',
        });
        line.on('mouseover', ev => this.hoverBus(s.route_id, ev.latlng));
        line.on('mousemove', ev => { if (this.hoverBusId === s.route_id && !this.pinnedBusId) this.busTip.setLatLng(ev.latlng); });
        line.on('mouseout', () => this.unhoverBus(s.route_id));
        line.on('click', ev => { this.L.DomEvent.stopPropagation(ev); this.pinBus(s.route_id, ev.latlng); });
        if (!this.modeLines.Bus.has(s.route_id)) this.modeLines.Bus.set(s.route_id, []);
        this.modeLines.Bus.get(s.route_id).push(line);
        this.busMeta.set(s.route_id, { short_name: s.short_name || s.route_id, long_name: s.long_name, silver });
      }
    }
    this.refreshModeLayers();
  }

  refreshModeLayers() {
    for (const [group, layer] of [['CR', this.crGroup], ['Bus', this.busGroup]]) {
      layer.clearLayers();
      if (this.focusRoute || !this.layers.lines || !this.groups.has(group)) continue;
      for (const lines of this.modeLines[group].values()) lines.forEach(l => layer.addLayer(l));
    }
    if (!this.groups.has('Bus') || this.focusRoute) this.unpinBus(true);
  }

  // ---- Bus routes: hover (or tap) for route number + live status -------------
  busTipHtml(routeId) {
    const meta = this.busMeta.get(routeId) || { short_name: routeId, long_name: '' };
    const status = this.busStatus.get(routeId);
    const label = meta.silver ? `Silver Line ${meta.short_name.replace(/^SL/i, '')}` : `Route ${meta.short_name}`;
    let live = '<span class="bus-tip__muted">Checking live buses…</span>';
    if (status && status.error) live = '<span class="bus-tip__muted">Live status unavailable right now</span>';
    else if (status) {
      const n = status.count;
      const moving = status.vehicles.filter(v => v.current_status === 'IN_TRANSIT_TO').length;
      live = n
        ? `<strong class="tabular">${n}</strong> bus${n === 1 ? '' : 'es'} running now${moving ? ` · ${moving} in transit` : ''}<br><span class="bus-tip__muted">Updated ${e(formatClockSeconds(status.at))}</span>`
        : 'No buses reporting right now';
    }
    const alert = alertsForRoutes(store.get('alerts') || [], [routeId])[0];
    const alertLine = alert ? `<div class="bus-tip__alert">⚠ ${e((alert.service_effect || alert.header || 'Service alert').slice(0, 90))}</div>` : '';
    const hint = this.pinnedBusId === routeId ? 'Buses shown on the map · click the map to clear' : 'Click to keep this route’s buses on the map';
    return `<div class="bus-tip__head"><span class="bus-tip__badge${meta.silver ? ' is-silver' : ''}">${e(meta.short_name)}</span><span>${e(label)}</span></div>
      ${meta.long_name ? `<div class="bus-tip__name">${e(meta.long_name)}</div>` : ''}
      <div class="bus-tip__live">${live}</div>${alertLine}<div class="bus-tip__hint">${hint}</div>`;
  }

  highlightBus(routeId, on) {
    for (const l of this.modeLines.Bus.get(routeId) || []) {
      l.setStyle(on ? { weight: 5, opacity: 1 } : { weight: 2, opacity: 0.22 });
      if (on) l.bringToFront();
    }
  }

  showBusTip(routeId, latlng) {
    if (latlng) this.busTip.setLatLng(latlng);
    this.busTip.setContent(this.busTipHtml(routeId));
    if (!this.map.hasLayer(this.busTip)) this.busTip.addTo(this.map);
  }

  hoverBus(routeId, latlng) {
    if (this.pinnedBusId && this.pinnedBusId !== routeId) return;
    clearTimeout(this.busLeaveTimer);
    if (this.hoverBusId && this.hoverBusId !== routeId) this.highlightBus(this.hoverBusId, false);
    this.hoverBusId = routeId;
    this.highlightBus(routeId, true);
    this.showBusTip(routeId, latlng);
    clearTimeout(this.busHoverTimer);
    this.busHoverTimer = setTimeout(() => this.loadBusStatus(routeId), 250);   // skip routes merely crossed
  }

  unhoverBus(routeId) {
    if (this.pinnedBusId === routeId || this.hoverBusId !== routeId) return;
    clearTimeout(this.busHoverTimer);
    this.busLeaveTimer = setTimeout(() => {
      if (this.hoverBusId !== routeId || this.pinnedBusId) return;
      this.highlightBus(routeId, false);
      this.hoverBusId = null;
      this.busTip.remove();
      this.clearBusVehicles();
    }, 250);
  }

  pinBus(routeId, latlng) {
    if (this.pinnedBusId && this.pinnedBusId !== routeId) { this.highlightBus(this.pinnedBusId, false); this.clearBusVehicles(); }
    this.pinnedBusId = routeId;
    this.hoverBusId = routeId;
    this.highlightBus(routeId, true);
    this.showBusTip(routeId, latlng);
    this.loadBusStatus(routeId);
  }

  unpinBus(silent = false) {
    const id = this.pinnedBusId || (silent ? this.hoverBusId : null);
    if (!id) return;
    this.pinnedBusId = null;
    this.hoverBusId = null;
    this.highlightBus(id, false);
    this.busTip.remove();
    this.clearBusVehicles();
  }

  async loadBusStatus(routeId, { force = false } = {}) {
    const cachedStatus = this.busStatus.get(routeId);
    if (!force && cachedStatus && !cachedStatus.error && Date.now() - cachedStatus.fetchedAt < BUS_STATUS_TTL) {
      this.renderBusVehicles(routeId);
      return;
    }
    const { data, error } = await api.vehicles(routeId);
    const status = error || !data
      ? { error: error || 'unavailable', fetchedAt: Date.now() }
      : { count: data.vehicles.length, vehicles: data.vehicles, at: data.updated_at, fetchedAt: Date.now() };
    if (status.error && cachedStatus && !cachedStatus.error) return;   // keep the last good status
    this.busStatus.set(routeId, status);
    if (this.hoverBusId === routeId || this.pinnedBusId === routeId) {
      if (this.map.hasLayer(this.busTip)) this.busTip.setContent(this.busTipHtml(routeId));
      this.renderBusVehicles(routeId);
    }
  }

  renderBusVehicles(routeId) {
    if (this.hoverBusId !== routeId && this.pinnedBusId !== routeId) return;
    const status = this.busStatus.get(routeId);
    if (!status || status.error) return;
    const seen = new Set();
    const color = this.busMeta.get(routeId)?.silver ? SILVER_COLOR : BUS_COLOR;
    for (const v of status.vehicles) {
      if (v.latitude == null) continue;
      seen.add(v.id);
      const existing = this.busVehicles.get(v.id);
      if (existing) {
        existing.setLatLng([v.latitude, v.longitude]);
        existing.getElement()?.querySelector('.veh')?.style.setProperty('--rot', `${Number(v.bearing) || 0}deg`);
        continue;
      }
      const label = `Route ${this.busMeta.get(routeId)?.short_name || routeId} bus ${v.label || v.id}${v.headsign ? ` to ${v.headsign}` : ''}${v.current_status ? ` · ${vehicleStatusText(v.current_status, v.stop_name)}` : ''}`;
      const marker = this.L.marker([v.latitude, v.longitude], {
        pane: 'vehicles', keyboard: true, title: label,
        icon: this.L.divIcon({ className: 'veh-icon veh-icon--bus', html: `<span class="veh" style="--c:${color};--rot:${Number(v.bearing) || 0}deg">${VEHICLE_SVG}</span>`, iconSize: [26, 26], iconAnchor: [13, 13] }),
      });
      marker.on('click', ev => { this.L.DomEvent.stopPropagation(ev); this.onVehicleSelect(v); });
      marker.on('mouseover', () => clearTimeout(this.busLeaveTimer));
      marker.on('add', () => { const el = marker.getElement(); if (el) { el.setAttribute('role', 'button'); el.setAttribute('aria-label', label); } });
      marker.addTo(this.busVehicleGroup);
      this.busVehicles.set(v.id, marker);
    }
    for (const [id, marker] of this.busVehicles) {
      if (!seen.has(id)) { this.busVehicleGroup.removeLayer(marker); this.busVehicles.delete(id); }
    }
  }

  clearBusVehicles() {
    this.busVehicleGroup.clearLayers();
    this.busVehicles.clear();
  }

  applyZoomClass() {
    const z = this.map.getZoom();
    this.el.classList.toggle('z-far', z <= 11);
    this.el.classList.toggle('z-mid', z > 11 && z < 14);
    this.el.classList.toggle('z-near', z >= 14);
  }

  async drawShapes(routeIds, color = null) {
    const missing = routeIds.filter(r => !this.shapeLayers.has(r));
    if (!missing.length) return;
    const { data } = await api.shapes(missing.join(','));
    if (!data) return;
    for (const s of data.shapes) {
      const id = routeIdentity(s.route_id);
      const line = this.L.polyline(decodePolyline(s.polyline), {
        pane: 'routes', color: color || id.bright, weight: 4, opacity: 0.9, lineCap: 'round', lineJoin: 'round', interactive: false,
      });
      if (!this.shapeLayers.has(s.route_id)) this.shapeLayers.set(s.route_id, []);
      this.shapeLayers.get(s.route_id).push(line);
    }
    this.refreshShapes();
  }

  refreshShapes() {
    this.shapeGroup.clearLayers();
    if (!this.layers.lines) return;
    for (const [routeId, lines] of this.shapeLayers) {
      const visible = this.focusRoute ? routeId === this.focusRoute : (LINES[routeId] && this.groups.has(groupOf(routeId)));
      if (visible) lines.forEach(l => this.shapeGroup.addLayer(l));
    }
  }

  drawStations() {
    const L = this.L;
    for (const st of store.get('stations')) {
      const lines = st.lines || [];
      const multi = new Set(lines.map(groupOf)).size > 1;
      const color = multi ? '#F8FAFC' : routeIdentity(lines[0]).bright;
      const marker = L.marker([st.lat, st.lon], {
        pane: 'stations', keyboard: true, title: st.name, riseOnHover: true,
        icon: L.divIcon({ className: 'stop-icon', html: `<span class="stop${multi ? ' stop--hub' : ''}" style="--c:${color}"></span>`, iconSize: [16, 16], iconAnchor: [8, 8] }),
      });
      marker.on('click', ev => { this.L.DomEvent.stopPropagation(ev); this.stationClick(st.id); });
      marker.on('add', () => { const el = marker.getElement(); if (el) { el.setAttribute('aria-label', `${st.name} station`); el.setAttribute('role', 'button'); } });
      this.stationMarkers.set(st.id, { marker, station: st });
    }
    this.refreshStations();
  }

  /** Visible station pins whose centers are within `px` screen pixels of station `id`. */
  stationsNear(id, px = 14) {
    const base = this.stationMarkers.get(id);
    if (!base) return [];
    const p = this.map.latLngToContainerPoint(base.marker.getLatLng());
    const near = [];
    for (const [sid, { marker, station }] of this.stationMarkers) {
      if (!this.stationGroup.hasLayer(marker)) continue;
      const d = p.distanceTo(this.map.latLngToContainerPoint(marker.getLatLng()));
      if (d <= px) near.push({ id: sid, station, d: sid === id ? -1 : d });
    }
    return near.sort((a, b) => a.d - b.d);
  }

  /** A click on a pin that overlaps others (e.g. Park Street / Downtown Crossing at network
   *  zoom) asks which station was meant instead of guessing. */
  stationClick(id) {
    const near = this.stationsNear(id);
    if (near.length <= 1) { this.onStationSelect(id); return; }
    const box = document.createElement('div');
    box.className = 'station-pick';
    box.setAttribute('role', 'group');
    box.setAttribute('aria-label', 'Choose a station');
    box.innerHTML = `<p class="station-pick__title">${near.length} stations here</p>` + near.map(n =>
      `<button type="button" class="station-pick__btn" data-pick="${e(n.id)}">${e(n.station.name)}</button>`).join('');
    box.addEventListener('click', ev => {
      const btn = ev.target.closest('[data-pick]');
      if (!btn) return;
      this.map.closePopup();
      this.onStationSelect(btn.dataset.pick);
    });
    this.L.popup({ className: 'pick-popup', closeButton: true, autoPan: true, offset: [0, -6] })
      .setLatLng(this.stationMarkers.get(id).marker.getLatLng()).setContent(box).openOn(this.map);
    setTimeout(() => box.querySelector('button')?.focus(), 0);
  }

  refreshStations() {
    this.stationGroup.clearLayers();
    if (!this.layers.stations) return;
    for (const { marker, station } of this.stationMarkers.values()) {
      const visible = this.focusRoute
        ? (station.lines || []).some(l => l === this.focusRoute || (this.focusRoute.startsWith('Green') && l.startsWith('Green')))
        : (station.lines || []).some(l => this.groups.has(groupOf(l)));
      if (visible) this.stationGroup.addLayer(marker);
    }
  }

  vehicleVisible(v) {
    if (!this.layers.vehicles) return false;
    if (this.focusRoute) return true;
    return this.groups.has(groupOf(v.route?.id));
  }

  vehicleLabel(v) {
    const name = routeIdentity(v.route?.id, { name: v.route?.name }).name;
    return `${name} vehicle ${v.label || v.id}${v.headsign ? ` to ${v.headsign}` : ''}`;
  }

  renderVehicles(list) {
    const L = this.L;
    const seen = new Set();
    for (const v of list) {
      if (v.latitude == null || v.longitude == null) continue;
      seen.add(v.id);
      const color = LINES[v.route?.id] ? LINES[v.route.id].bright : groupOf(v.route?.id) === 'CR' ? CR_COLOR : (v.route?.color || '#94A3B8');
      const existing = this.vehicles.get(v.id);
      if (existing) {
        existing.data = v;
        existing.marker.setLatLng([v.latitude, v.longitude]);
        const el = existing.marker.getElement();
        if (el) {
          el.querySelector('.veh')?.style.setProperty('--rot', `${Number(v.bearing) || 0}deg`);
          el.setAttribute('aria-label', this.vehicleLabel(v));
        }
      } else {
        const marker = L.marker([v.latitude, v.longitude], {
          pane: 'vehicles', keyboard: true, title: this.vehicleLabel(v),
          icon: L.divIcon({ className: 'veh-icon', html: `<span class="veh" style="--c:${e(color)};--rot:${Number(v.bearing) || 0}deg">${VEHICLE_SVG}</span>`, iconSize: [26, 26], iconAnchor: [13, 13] }),
        });
        marker.on('click', () => this.selectVehicle(v.id));
        marker.on('add', () => {
          const el = marker.getElement();
          if (!el) return;
          el.setAttribute('role', 'button');
          el.setAttribute('aria-label', this.vehicleLabel(this.vehicles.get(v.id)?.data || v));
          el.classList.toggle('is-selected', this.selectedVehicle === v.id);
        });
        this.vehicles.set(v.id, { marker, data: v });
      }
    }
    for (const [id, entry] of this.vehicles) {
      if (!seen.has(id)) { this.vehicleGroup.removeLayer(entry.marker); this.vehicles.delete(id); }
    }
    this.refreshVehicles();
  }

  refreshVehicles() {
    this.clusterVehicles();
    for (const [id, { marker, data }] of this.vehicles) {
      const show = this.vehicleVisible(data) && !this.clustered.has(id);
      const on = this.vehicleGroup.hasLayer(marker);
      if (show && !on) this.vehicleGroup.addLayer(marker);
      if (!show && on) this.vehicleGroup.removeLayer(marker);
    }
  }

  /** Merge overlapping same-line vehicles at network zoom; individual markers from zoom 13. */
  clusterVehicles() {
    this.clusterGroup.clearLayers();
    this.clustered = new Set();
    if (!this.map || this.map.getZoom() > CLUSTER_MAX_ZOOM) return;
    const points = [];
    for (const [id, { data }] of this.vehicles) {
      if (id === this.selectedVehicle || !this.vehicleVisible(data)) continue;
      const group = groupOf(data.route?.id);
      if (!group) continue;
      const pt = this.map.latLngToLayerPoint([data.latitude, data.longitude]);
      points.push({ id, x: pt.x, y: pt.y, group });
    }
    for (const c of clusterPoints(points, CLUSTER_RADIUS)) {
      if (c.ids.length < 2) continue;
      c.ids.forEach(id => this.clustered.add(id));
      const color = c.group === 'CR' ? CR_COLOR : routeIdentity(LINE_GROUPS.find(g => g.id === c.group)?.routes[0]).bright;
      const label = `${c.ids.length} ${GROUP_NAMES[c.group] || ''} vehicles here — zoom in to see each one`;
      const latlngs = c.ids.map(id => { const d = this.vehicles.get(id).data; return [d.latitude, d.longitude]; });
      const marker = this.L.marker(this.map.layerPointToLatLng([c.x, c.y]), {
        pane: 'vehicles', keyboard: true, title: label,
        icon: this.L.divIcon({ className: 'veh-icon', html: `<span class="veh-cluster" style="--c:${e(color)}">${c.ids.length}</span>`, iconSize: [26, 26], iconAnchor: [13, 13] }),
      });
      marker.on('click', ev => {
        this.L.DomEvent.stopPropagation(ev);
        const bounds = this.L.latLngBounds(latlngs);
        this.map.flyToBounds(bounds.pad(0.6), { maxZoom: Math.max(CLUSTER_MAX_ZOOM + 2, this.map.getZoom() + 2), duration: 0.5 });
      });
      marker.on('add', () => { const el = marker.getElement(); if (el) { el.setAttribute('role', 'button'); el.setAttribute('aria-label', label); } });
      marker.addTo(this.clusterGroup);
    }
  }

  visibleVehicleCount() {
    let n = 0;
    for (const { data } of this.vehicles.values()) if (this.vehicleVisible(data)) n++;
    return n;
  }

  selectVehicle(id, { pan = false } = {}) {
    if (this.selectedVehicle) this.vehicles.get(this.selectedVehicle)?.marker.getElement()?.classList.remove('is-selected');
    this.selectedVehicle = id;
    if (this.map && this.map.getZoom() <= CLUSTER_MAX_ZOOM) this.refreshVehicles();
    const entry = id && this.vehicles.get(id);
    if (entry) {
      entry.marker.getElement()?.classList.add('is-selected');
      if (pan) this.map.panTo(entry.marker.getLatLng());
      this.onVehicleSelect(entry.data);
    }
  }

  selectStation(id, { fly = false } = {}) {
    if (this.selectedStation) this.stationMarkers.get(this.selectedStation)?.marker.getElement()?.classList.remove('is-selected');
    this.selectedStation = id;
    const entry = id && this.stationMarkers.get(id);
    if (!entry) return;
    entry.marker.getElement()?.classList.add('is-selected');
    if (fly) {
      // On phones the bottom sheet covers the lower half, so aim the station above it.
      const zoom = Math.max(this.map.getZoom(), 15);
      const sheetOffset = window.matchMedia('(max-width: 767px)').matches ? this.el.clientHeight * 0.24 : 0;
      const point = this.map.project(entry.marker.getLatLng(), zoom).add([0, sheetOffset]);
      this.map.flyTo(this.map.unproject(point, zoom), zoom, { duration: 0.6 });
    }
  }

  getVehicle(id) { return this.vehicles.get(id)?.data || null; }

  routesParam() {
    if (this.focusRoute) return this.focusRoute;
    const rail = this.groups.has('CR') ? this.crRouteIds : [];
    return [...SUBWAY_IDS, ...rail].join(',');
  }

  async tick() {
    if (document.hidden) return;
    const mySeq = ++this.seq;
    this.controller?.abort();
    this.controller = new AbortController();
    this.onStatus({ loading: true, updatedAt: this.lastUpdate, count: this.visibleVehicleCount() });
    const { data, error, aborted } = await api.vehicles(this.routesParam(), { signal: this.controller.signal });
    if (aborted || mySeq !== this.seq) return;     // a newer request superseded this one
    if (error) {
      this.onStatus({ error, updatedAt: this.lastUpdate, count: this.visibleVehicleCount() });
      return;                                        // keep the last good markers on screen
    }
    this.lastUpdate = data.updated_at || new Date().toISOString();
    this.renderVehicles(data.vehicles || []);
    this.onStatus({ updatedAt: this.lastUpdate, count: this.visibleVehicleCount() });
    if (this.pinnedBusId) this.loadBusStatus(this.pinnedBusId, { force: true });
    if (this.selectedVehicle) {
      const v = this.vehicles.get(this.selectedVehicle);
      document.dispatchEvent(new CustomEvent('vehicle-updated', { detail: v ? v.data : null }));
    }
  }

  start() {
    if (this.timer) return;
    this.tick();
    this.timer = setInterval(() => this.tick(), REFRESH_MS);
  }

  stop() {
    clearInterval(this.timer);
    this.timer = null;
    this.controller?.abort();
  }

  setGroup(id, on) {
    if (on) this.groups.add(id); else this.groups.delete(id);
    this.refreshShapes(); this.refreshStations(); this.refreshVehicles(); this.refreshModeLayers();
    if (id === 'CR' && on && this.timer) this.tick();
    this.onStatus({ updatedAt: this.lastUpdate, count: this.visibleVehicleCount() });
  }

  setLayer(name, on) {
    this.layers[name] = on;
    this.refreshShapes(); this.refreshStations(); this.refreshVehicles(); this.refreshModeLayers();
    this.onStatus({ updatedAt: this.lastUpdate, count: this.visibleVehicleCount() });
  }

  async setFocus(routeId) {
    if (this.focusRoute === routeId) return;
    this.focusRoute = routeId || null;
    // Vehicles for a different route set: clear and refetch.
    for (const { marker } of this.vehicles.values()) this.vehicleGroup.removeLayer(marker);
    this.clusterGroup.clearLayers();
    this.clustered.clear();
    this.vehicles.clear();
    if (routeId && !this.shapeLayers.has(routeId)) await this.drawShapes([routeId]);
    this.refreshShapes(); this.refreshStations(); this.refreshModeLayers();
    if (routeId) this.fitRoute(routeId); else this.fitNetwork();
    if (this.timer) this.tick();
  }

  fitRoute(routeId) {
    const lines = this.shapeLayers.get(routeId);
    if (!lines || !lines.length) return;
    const bounds = lines.reduce((b, l) => b.extend(l.getBounds()), this.L.latLngBounds(lines[0].getBounds()));
    this.map.fitBounds(bounds, { padding: [40, 40] });
  }

  fitNetwork() {
    const all = [...this.shapeLayers.entries()].filter(([id]) => LINES[id]).flatMap(([, ls]) => ls);
    if (!all.length) { this.map.setView(BOSTON.center, BOSTON.zoom); return; }
    const bounds = all.reduce((b, l) => b.extend(l.getBounds()), this.L.latLngBounds(all[0].getBounds()));
    this.map.fitBounds(bounds, { padding: [30, 30] });
  }

  showUser(lat, lon, { fly = true } = {}) {
    this.userGroup.clearLayers();
    this.L.circle([lat, lon], { radius: 60, color: '#4C8DF6', weight: 1, fillOpacity: 0.12, interactive: false }).addTo(this.userGroup);
    this.L.marker([lat, lon], {
      keyboard: false, interactive: false,
      icon: this.L.divIcon({ className: 'me-icon', html: '<span class="me" aria-hidden="true"></span>', iconSize: [18, 18], iconAnchor: [9, 9] }),
    }).addTo(this.userGroup);
    if (fly) this.map.flyTo([lat, lon], 15, { duration: 0.6 });
  }

  invalidate() { this.map?.invalidateSize(); }
}
