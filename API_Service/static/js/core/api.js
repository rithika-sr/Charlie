// Data access. Every call resolves to { data, error } so views can render
// loading / success / error states explicitly instead of catching exceptions.

import { store } from './store.js';

const DEFAULT_TIMEOUT = 15000;

export async function request(path, { method = 'GET', body, signal, timeout = DEFAULT_TIMEOUT } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new DOMException('Timed out', 'TimeoutError')), timeout);
  const onAbort = () => controller.abort(signal.reason);
  if (signal) {
    if (signal.aborted) controller.abort(signal.reason);
    else signal.addEventListener('abort', onAbort, { once: true });
  }
  try {
    const res = await fetch(path, {
      method,
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
    const data = await res.json().catch(() => null);
    if (!res.ok || (data && data.error && !data.reply)) {
      return { data: null, error: (data && data.error) || `Request failed (${res.status})` };
    }
    store.markFresh();
    return { data, error: null };
  } catch (err) {
    if (signal && signal.aborted) return { data: null, error: null, aborted: true };
    const timedOut = err && err.name === 'TimeoutError';
    store.markStale();
    return { data: null, error: timedOut ? 'The MBTA took too long to respond.' : 'Network error — check your connection.' };
  } finally {
    clearTimeout(timer);
    if (signal) signal.removeEventListener('abort', onAbort);
  }
}

export const api = {
  stations: () => cachedOnce('stations', () => request('/api/stations')),
  routes: () => cachedOnce('routes', () => request('/api/routes')),
  modeShapes: (mode) => cachedOnce(`mode:${mode}`, () => request(`/api/mode-shapes?mode=${mode}`, { timeout: 30000 })),
  shapes: (routes) => cachedOnce(`shapes:${routes}`, () => request(`/api/shapes?route=${encodeURIComponent(routes)}`, { timeout: 25000 })),
  dashboard: (opts) => request('/api/dashboard', opts),
  alerts: (opts) => request('/api/alerts', opts),
  station: (id, opts) => request(`/api/station/${encodeURIComponent(id)}`, opts),
  line: (id, opts) => request(`/api/line/${encodeURIComponent(id)}`, opts),
  vehicles: (routes, opts) => request(`/api/vehicles?route=${encodeURIComponent(routes)}&limit=300`, opts),
  trip: (body, opts) => request('/api/trip-planner', { method: 'POST', body, timeout: 30000, ...opts }),
  setLocation: (lat, lon) => request('/api/set-location', { method: 'POST', body: { lat, lon } }),
  chat: (body) => request('/api/chat', { method: 'POST', body, timeout: 45000 }),
};

// Static-ish datasets are fetched once per session; failures are not cached.
const memo = new Map();
function cachedOnce(key, fn) {
  if (!memo.has(key)) {
    memo.set(key, fn().then(r => { if (r.error) memo.delete(key); return r; }));
  }
  return memo.get(key);
}
