// Shared transit definitions: line identity, colors and modes.
// `color` is the official MBTA color (badge backgrounds, with white text).
// `bright` is a lighter tone of the same hue for strokes and dots on the dark map/UI.

export const LINES = {
  'Red':      { name: 'Red Line',         code: 'RL', group: 'Red',      color: '#DA291C', bright: '#E5484D', mode: 'subway' },
  'Mattapan': { name: 'Mattapan Trolley', code: 'M',  group: 'Mattapan', color: '#DA291C', bright: '#E5484D', mode: 'subway' },
  'Orange':   { name: 'Orange Line',      code: 'OL', group: 'Orange',   color: '#ED8B00', bright: '#F1A23A', mode: 'subway' },
  'Blue':     { name: 'Blue Line',        code: 'BL', group: 'Blue',     color: '#003DA5', bright: '#4C8DF6', mode: 'subway' },
  'Green-B':  { name: 'Green Line B',     code: 'B',  group: 'Green',    color: '#00843D', bright: '#35A56F', mode: 'subway' },
  'Green-C':  { name: 'Green Line C',     code: 'C',  group: 'Green',    color: '#00843D', bright: '#35A56F', mode: 'subway' },
  'Green-D':  { name: 'Green Line D',     code: 'D',  group: 'Green',    color: '#00843D', bright: '#35A56F', mode: 'subway' },
  'Green-E':  { name: 'Green Line E',     code: 'E',  group: 'Green',    color: '#00843D', bright: '#35A56F', mode: 'subway' },
};

export const SUBWAY_IDS = ['Red', 'Orange', 'Blue', 'Green-B', 'Green-C', 'Green-D', 'Green-E', 'Mattapan'];

// Map filter groups (one toggle per color family).
export const LINE_GROUPS = [
  { id: 'Red', name: 'Red', routes: ['Red'] },
  { id: 'Orange', name: 'Orange', routes: ['Orange'] },
  { id: 'Blue', name: 'Blue', routes: ['Blue'] },
  { id: 'Green', name: 'Green', routes: ['Green-B', 'Green-C', 'Green-D', 'Green-E'] },
  { id: 'Mattapan', name: 'Mattapan', routes: ['Mattapan'] },
];

export const MODE_COLORS = {
  bus: '#FFC72C',
  silver: '#7C878E',
  commuter_rail: '#80276C',
  ferry: '#008EAA',
};

export const MODES = [
  { id: 'subway', name: 'Subway' },
  { id: 'bus', name: 'Bus' },
  { id: 'commuter_rail', name: 'Commuter Rail' },
  { id: 'ferry', name: 'Ferry' },
];

export const QUICK_STATIONS = ['place-pktrm', 'place-dwnxg', 'place-sstat', 'place-north', 'place-harsq', 'place-bbsta'];

export const BOSTON = { center: [42.3601, -71.0789], zoom: 12 };

/** Identity for any route id, including buses, Commuter Rail and ferries. */
export function routeIdentity(routeId, route = {}) {
  if (LINES[routeId]) return { id: routeId, ...LINES[routeId] };
  const id = String(routeId || '');
  if (id.startsWith('CR-')) {
    const name = route.name || `${id.slice(3)} Line`;
    return { id, name, code: 'CR', group: 'CR', color: MODE_COLORS.commuter_rail, bright: '#B05AA4', mode: 'commuter_rail' };
  }
  if (id.startsWith('Boat-')) {
    return { id, name: route.name || 'Ferry', code: 'F', group: 'Ferry', color: MODE_COLORS.ferry, bright: '#2BB3CF', mode: 'ferry' };
  }
  const short = route.short_name || id;
  if (/^SL/i.test(short) || ['741', '742', '743', '746', '749', '751'].includes(id)) {
    return { id, name: route.name ? `Silver Line ${short.replace(/^SL/i, '')}` : `Silver Line`, code: short, group: 'Silver', color: MODE_COLORS.silver, bright: '#A3ADB5', mode: 'bus' };
  }
  return { id, name: `Route ${short}`, longName: route.name || '', code: short, group: 'Bus', color: MODE_COLORS.bus, bright: MODE_COLORS.bus, mode: 'bus', darkText: true };
}

export function modeName(mode) {
  return (MODES.find(m => m.id === mode) || {}).name || 'Transit';
}

/** Text color (near-black or white) with the higher WCAG contrast on `hex`. */
export function inkFor(hex) {
  const m = String(hex || '').replace('#', '').match(/^([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i);
  if (!m) return '#FFFFFF';
  const lin = v => { const c = parseInt(v, 16) / 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
  const L = 0.2126 * lin(m[1]) + 0.7152 * lin(m[2]) + 0.0722 * lin(m[3]);
  const white = 1.05 / (L + 0.05);
  const dark = (L + 0.05) / 0.0603; // #1A1A1A luminance ≈ 0.0103
  return dark > white ? '#1A1A1A' : '#FFFFFF';
}
