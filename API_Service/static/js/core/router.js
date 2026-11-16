// Hash router: #/view/param?query

export const ROUTES = ['overview', 'map', 'trip', 'stations', 'lines', 'alerts', 'ai'];

export function parseHash(hash = location.hash) {
  const raw = hash.replace(/^#\/?/, '');
  const [path, queryString = ''] = raw.split('?');
  const parts = path.split('/').filter(Boolean).map(decodeURIComponent);
  const name = ROUTES.includes(parts[0]) ? parts[0] : 'overview';
  const query = Object.fromEntries(new URLSearchParams(queryString));
  return { name, params: ROUTES.includes(parts[0]) ? parts.slice(1) : [], query };
}

export function href(name, ...params) {
  const path = [name, ...params.map(p => encodeURIComponent(p))].join('/');
  return `#/${path}`;
}

export function navigate(name, params = [], query = null) {
  const qs = query ? `?${new URLSearchParams(query)}` : '';
  const target = href(name, ...params) + qs;
  if (location.hash === target) window.dispatchEvent(new HashChangeEvent('hashchange'));
  else location.hash = target;
}
