// Tiny app store: shared state + pub/sub, with safe localStorage persistence
// for per-viewer conveniences (recent stations/trips, location, chat context).

const listeners = new Set();

function load(key, fallback) {
  try {
    const raw = localStorage.getItem(`charlie:${key}`);
    return raw ? JSON.parse(raw) : fallback;
  } catch { return fallback; }
}

function save(key, value) {
  try { localStorage.setItem(`charlie:${key}`, JSON.stringify(value)); } catch { /* storage unavailable */ }
}

// Migrate the location saved by the previous UI.
function initialLocation() {
  const current = load('location', null);
  if (current) return current;
  try { return JSON.parse(localStorage.getItem('charlie-location') || 'null'); } catch { return null; }
}

const state = {
  stations: [],          // [{id, name, lat, lon, lines}]
  routes: null,          // {subway, bus, commuter_rail, ferry}
  alerts: null,          // normalized alert list
  dashboard: null,
  location: initialLocation(),     // {lat, lon, station, stop_id, lines, distance}
  recentStations: load('recentStations', []),
  recentTrips: load('recentTrips', []),
  chat: { messages: [], history: [] },
  pendingTrip: null,
  lastUpdated: null,
  stale: false,
};

export const store = {
  get: key => state[key],
  set(key, value) {
    state[key] = value;
    if (key === 'location') save('location', value);
    if (key === 'recentStations') save('recentStations', value);
    if (key === 'recentTrips') save('recentTrips', value);
    listeners.forEach(fn => fn(key, value));
  },
  subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); },
  markFresh() { state.stale = false; state.lastUpdated = new Date(); listeners.forEach(fn => fn('freshness', state.lastUpdated)); },
  markStale() { state.stale = true; listeners.forEach(fn => fn('freshness', state.lastUpdated)); },

  stationById: id => state.stations.find(s => s.id === id),

  rememberStation(id) {
    const list = [id, ...state.recentStations.filter(x => x !== id)].slice(0, 6);
    store.set('recentStations', list);
  },
  rememberTrip(trip) {
    const key = t => `${t.origin.name}→${t.destination.name}`;
    const list = [trip, ...state.recentTrips.filter(t => key(t) !== key(trip))].slice(0, 5);
    store.set('recentTrips', list);
  },
};
