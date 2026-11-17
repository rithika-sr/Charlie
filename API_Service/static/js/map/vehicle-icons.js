// Live vehicle markers: a mode glyph (bus, subway, light rail, train, ferry) in the line's
// color, a small pointer that turns to the vehicle's heading, and smooth gliding between
// live position updates. Pure helpers are exported for tests; MarkerMotion drives Leaflet.

import { LINES, inkFor } from '../core/constants.js';

/** Which glyph a route gets. */
export function vehicleKind(routeId) {
  const id = String(routeId || '');
  if (id.startsWith('Green-') || id === 'Mattapan') return 'lightrail';
  if (LINES[id]) return 'subway';
  if (id.startsWith('CR-')) return 'train';
  if (id.startsWith('Boat-')) return 'ferry';
  return 'bus';
}

export const KIND_NAMES = { bus: 'Bus', subway: 'Subway train', lightrail: 'Light rail train', train: 'Commuter Rail train', ferry: 'Ferry' };

// Original 24×24 glyphs, drawn as single even-odd paths so they take one fill color.
export const VEHICLE_GLYPHS = {
  bus: 'M6.5 2.5h11A2.5 2.5 0 0 1 20 5v11.2a2 2 0 0 1-1.2 1.8V20a1 1 0 0 1-1 1h-1.2a1 1 0 0 1-1-1v-1.3H8.4V20a1 1 0 0 1-1 1H6.2a1 1 0 0 1-1-1v-2a2 2 0 0 1-1.2-1.8V5a2.5 2.5 0 0 1 2.5-2.5zM8 4.2v1.3h8V4.2zM6 7v5.2h12V7zm1.8 7a1.3 1.3 0 1 0 0 2.6 1.3 1.3 0 0 0 0-2.6zm8.4 0a1.3 1.3 0 1 0 0 2.6 1.3 1.3 0 0 0 0-2.6z',
  subway: 'M7.5 2h9A3.5 3.5 0 0 1 20 5.5v9.3a3.2 3.2 0 0 1-2.4 3.1l1.9 3.1h-2.3l-1.7-2.8H8.5L6.8 21H4.5l1.9-3.1A3.2 3.2 0 0 1 4 14.8V5.5A3.5 3.5 0 0 1 7.5 2zM6 6.2v4.6h5.2V6.2zm6.8 0v4.6H18V6.2zM7.9 13a1.3 1.3 0 1 0 0 2.6 1.3 1.3 0 0 0 0-2.6zm8.2 0a1.3 1.3 0 1 0 0 2.6 1.3 1.3 0 0 0 0-2.6z',
  lightrail: 'M8.5 1h7v1.4h-2.8L14.4 5H17a3 3 0 0 1 3 3v7.2a3 3 0 0 1-2.2 2.9l1.7 2.9h-2.3l-1.6-2.7H8.4L6.8 21H4.5l1.7-2.9A3 3 0 0 1 4 15.2V8a3 3 0 0 1 3-3h2.6l1.7-2.6H8.5zM6 8.4v4h12v-4zm1.9 6a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5zm8.2 0a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5z',
  train: 'M9 2h6c2.4 0 4.2 1.2 4.8 3.4L20 14.8a3.2 3.2 0 0 1-2.4 3.1l1.9 3.1h-2.3l-1.7-2.8H8.5L6.8 21H4.5l1.9-3.1A3.2 3.2 0 0 1 4 14.8l.2-9.4C4.8 3.2 6.6 2 9 2zm-.2 2.4c-1.3 0-2.1.6-2.4 1.7L6.2 9.6h11.6l-.2-3.5c-.3-1.1-1.1-1.7-2.4-1.7zM12 12a1.6 1.6 0 1 0 0 3.2 1.6 1.6 0 0 0 0-3.2zm-5.1.6a1 1 0 1 0 0 2 1 1 0 0 0 0-2zm10.2 0a1 1 0 1 0 0 2 1 1 0 0 0 0-2z',
  ferry: 'M10 2h4v2.2h3.2L19 10.8l2.2.8-2.4 6.2c-1.6.1-2.7-.5-3.6-1.3-1 .8-2.1 1.3-3.2 1.3s-2.2-.5-3.2-1.3c-.9.8-2 1.4-3.6 1.3L2.8 11.6l2.2-.8L6.8 4.2H10zM8.4 6l-1 3.8L12 8.2l4.6 1.6-1-3.8zM2 19.6c1.6.6 3 .4 4.4-.6 1.7 1.2 3.7 1.6 5.6 0 1.9 1.6 3.9 1.2 5.6 0 1.4 1 2.8 1.2 4.4.6V21c-1.6.6-3 .4-4.4-.4-1.8 1.2-3.8 1.4-5.6.2-1.8 1.2-3.8 1-5.6-.2-1.4.8-2.8 1-4.4.4z',
};

const hasBearing = b => b !== null && b !== undefined && b !== '' && Number.isFinite(Number(b));

/** Marker HTML: heading pointer + glyph disc (+ route tag, shown when zoomed in). */
export function vehicleIconHtml({ color, routeId, bearing, code = '' }) {
  const kind = vehicleKind(routeId);
  const ink = inkFor(color);
  const rot = hasBearing(bearing) ? Number(bearing) : 0;
  return `<span class="veh veh--${kind}${hasBearing(bearing) ? '' : ' veh--still'}" style="--c:${color};--ink:${ink};--rot:${rot}deg">` +
    `<span class="veh__heading" aria-hidden="true"></span>` +
    `<span class="veh__body"><svg viewBox="0 0 24 24" aria-hidden="true"><path fill-rule="evenodd" d="${VEHICLE_GLYPHS[kind]}"/></svg></span>` +
    (code ? `<span class="veh__tag" aria-hidden="true">${String(code).replace(/[<>&"]/g, '')}</span>` : '') +
    `</span>`;
}

/** Next rotation in degrees that turns the short way round (350° → 10° turns +20°, not −340°). */
export function nextRotation(previous, bearing) {
  if (!hasBearing(bearing)) return previous;
  const b = ((Number(bearing) % 360) + 360) % 360;
  if (!Number.isFinite(previous)) return b;
  const cur = ((previous % 360) + 360) % 360;
  const delta = ((b - cur + 540) % 360) - 180;
  return previous + delta;
}

/** Update an existing marker's heading and tag in place (no rebuild, so it keeps gliding). */
export function updateVehicleIcon(el, { bearing, code }) {
  const veh = el && el.querySelector('.veh');
  if (!veh) return;
  if (hasBearing(bearing)) {
    const prev = parseFloat(veh.dataset.rot);
    const rot = nextRotation(Number.isFinite(prev) ? prev : parseFloat(veh.style.getPropertyValue('--rot')), bearing);
    veh.dataset.rot = String(rot);
    veh.style.setProperty('--rot', `${rot}deg`);
    veh.classList.remove('veh--still');
  }
  const tag = veh.querySelector('.veh__tag');
  if (tag && code != null) tag.textContent = code;
}

// ---------- smooth movement ----------

/** Approximate metres between two [lat, lon] points (fine at city scale). */
export function metersBetween(a, b) {
  const dLat = (b[0] - a[0]) * 111320;
  const dLon = (b[1] - a[1]) * 111320 * Math.cos(((a[0] + b[0]) / 2) * Math.PI / 180);
  return Math.hypot(dLat, dLon);
}

/** Position `p` (0–1) of the way from `from` to `to`. */
export function lerpLatLng(from, to, p) {
  const t = Math.min(1, Math.max(0, p));
  return [from[0] + (to[0] - from[0]) * t, from[1] + (to[1] - from[1]) * t];
}

export const MIN_MOVE_M = 3;        // GPS jitter: don't animate tiny changes
export const MAX_GLIDE_M = 2500;    // bigger jumps (new trip, bad fix) snap instead of sliding across town

/**
 * Glides Leaflet markers to each new live position over `duration` ms, at constant speed,
 * so a vehicle keeps moving bit by bit until the next update arrives. A new update starts
 * from wherever the marker is drawn now, so there's never a jump backwards.
 */
export class MarkerMotion {
  constructor({ map = null, now = () => performance.now(), raf, caf, reducedMotion } = {}) {
    this.map = map;
    this.now = now;
    this.raf = raf || (cb => requestAnimationFrame(cb));
    this.caf = caf || (id => cancelAnimationFrame(id));
    this.reduced = reducedMotion ?? (typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches);
    this.tracks = new Map();        // marker -> { from, to, start, duration }
    this.frameId = null;
    this.lastFrame = 0;
  }

  /** Move `marker` to `to` ([lat, lon]); returns 'glide' | 'snap' | 'none'. */
  move(marker, to, duration) {
    const cur = marker.getLatLng();
    const from = [cur.lat, cur.lng];
    const dist = metersBetween(from, to);
    if (dist < MIN_MOVE_M && !this.tracks.has(marker)) return 'none';
    if (this.reduced || !(duration > 0) || dist > MAX_GLIDE_M) {
      this.tracks.delete(marker);
      marker.setLatLng(to);
      return 'snap';
    }
    this.tracks.set(marker, { from, to, start: this.now(), duration });
    this.kick();
    return 'glide';
  }

  /** Forget a marker (removed from the map). */
  cancel(marker) { this.tracks.delete(marker); }

  clear() {
    this.tracks.clear();
    if (this.frameId != null) this.caf(this.frameId);
    this.frameId = null;
  }

  kick() {
    if (this.frameId == null && this.tracks.size) this.frameId = this.raf(t => this.frame(t));
  }

  frame(t = this.now()) {
    this.frameId = null;
    // Leaflet repositions markers itself during a zoom animation; wait for it to finish.
    const zooming = this.map && this.map._animatingZoom;
    if (!zooming && t - this.lastFrame >= 30) {           // ~30 fps is plenty for map motion
      this.lastFrame = t;
      for (const [marker, tr] of this.tracks) {
        const p = (this.now() - tr.start) / tr.duration;
        marker.setLatLng(lerpLatLng(tr.from, tr.to, p));
        if (p >= 1) this.tracks.delete(marker);
      }
    }
    this.kick();
  }
}
