// Station and route search ranking. Pure functions.

const ABBREVIATIONS = [
  [/\bst\b/g, 'street'], [/\bsq\b/g, 'square'], [/\bctr\b/g, 'center'], [/\bave\b/g, 'avenue'],
  [/\bmt\b/g, 'mount'], [/\bjct\b/g, 'junction'], [/\bstn\b/g, 'station'],
];

export function normalize(value) {
  let s = String(value || '').toLowerCase()
    .normalize('NFD').replace(/[̀-ͯ]/g, '')
    .replace(/[’'`.]/g, '')
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
  for (const [re, full] of ABBREVIATIONS) s = s.replace(re, full);
  return s;
}

/** Characters of `query` appear in order inside `text` (loose fuzzy match). */
function isSubsequence(query, text) {
  let i = 0;
  for (const ch of text) if (ch === query[i]) i++;
  return i === query.length;
}

/** Lower is better; null means no match. */
export function scoreMatch(query, name) {
  const q = normalize(query), n = normalize(name);
  if (!q) return null;
  if (n === q) return 0;
  if (n.startsWith(q)) return 1;
  if (n.split(' ').some(w => w.startsWith(q))) return 2;
  if (n.includes(q)) return 3;
  const compact = q.replace(/ /g, '');
  if (compact.length >= 3 && isSubsequence(compact, n.replace(/ /g, ''))) return 4;
  return null;
}

/**
 * Rank items by how well `getText(item)` matches the query.
 * Ties keep shorter names first, then alphabetical.
 */
export function rank(items, query, getText = x => x.name, limit = 8) {
  const scored = [];
  for (const item of items) {
    const texts = [].concat(getText(item)).filter(Boolean);
    let best = null;
    for (const t of texts) {
      const s = scoreMatch(query, t);
      if (s !== null && (best === null || s < best)) best = s;
    }
    if (best !== null) scored.push({ item, score: best, name: texts[0] });
  }
  scored.sort((a, b) => a.score - b.score || a.name.length - b.name.length || a.name.localeCompare(b.name));
  return scored.slice(0, limit).map(s => s.item);
}

/**
 * Route queries as riders type them: "39", "Route 39", "Bus 39", "rt 39", "#39".
 * Returns { core, explicit } where `explicit` means a route word was used.
 */
export function parseRouteQuery(query) {
  const q = normalize(query);
  const m = q.match(/^(?:route|rte|rt|bus|line|no|number)\s+(.+)$/) || String(query).trim().match(/^#\s*(\S+)$/);
  return m ? { core: normalize(m[1]), explicit: true } : { core: q, explicit: false };
}

/**
 * Rank routes for a query. An exact route number ("39" → route 39) ranks first,
 * then route numbers that start with it ("39" → 390?), then name matches.
 */
export function rankRoutes(routes, query, limit = 5) {
  const { core } = parseRouteQuery(query);
  if (!core) return [];
  const scored = [];
  for (const r of routes) {
    const short = normalize(r.short_name || r.id);
    let s = null;
    if (short === core || normalize(r.id) === core) s = 0;
    else if (short.startsWith(core) && /^\d/.test(core)) s = 1;
    else {
      const byName = [r.short_name, r.name].filter(Boolean).map(t => scoreMatch(core, t)).filter(v => v !== null);
      if (byName.length) s = 2 + Math.min(...byName);
    }
    if (s !== null) scored.push({ r, s });
  }
  scored.sort((a, b) => a.s - b.s || String(a.r.short_name || a.r.id).length - String(b.r.short_name || b.r.id).length
    || String(a.r.short_name || a.r.name).localeCompare(String(b.r.short_name || b.r.name), undefined, { numeric: true }));
  return scored.slice(0, limit).map(x => ({ ...x.r, exact: x.s === 0 }));
}

/** Global search: stations and routes. Exact route numbers ("39", "Route 39") lead. */
export function searchResults(q, stations, routesByType) {
  const { explicit } = parseRouteQuery(q);
  const st = rank(stations || [], q, s => s.name, 6).map(s => ({ kind: 'station', ...s }));
  const r = routesByType;
  const routes = r ? rankRoutes([...(r.subway || []), ...(r.commuter_rail || []), ...(r.ferry || []), ...(r.bus || [])], q, 5)
    .map(x => ({ kind: 'route', ...x })) : [];
  if (explicit || routes.some(x => x.exact)) {
    const exact = routes.filter(x => x.exact);
    return [...exact, ...st, ...routes.filter(x => !x.exact)];
  }
  return [...st, ...routes];
}
