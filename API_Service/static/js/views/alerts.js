// Service Alerts center: prioritized, filterable, with active periods and sources.

import { api } from '../core/api.js';
import { store } from '../core/store.js';
import { escapeHtml as e, formatClockSeconds } from '../core/format.js';
import { filterAlerts, countByFilter } from '../core/alerts.js';
import { alertCard, emptyState, errorState, skeleton } from '../ui/components.js';
import { icon } from '../ui/icons.js';

const FILTERS = [
  { id: 'all', name: 'All service' }, { id: 'subway', name: 'Subway' }, { id: 'bus', name: 'Bus' },
  { id: 'commuter_rail', name: 'Commuter Rail' }, { id: 'ferry', name: 'Ferry' }, { id: 'accessibility', name: 'Accessibility' },
];
const REFRESH_MS = 60000;
let filter = 'all';
let timer = null;
let updatedAt = null;

function render(el, alerts) {
  const counts = countByFilter(alerts);
  el.querySelector('[data-filters]').innerHTML = FILTERS.map(f =>
    `<button class="segmented__btn" data-filter="${f.id}" aria-pressed="${f.id === filter}">${e(f.name)}<span class="count tabular">${counts[f.id]}</span></button>`).join('');
  const list = filterAlerts(alerts, filter);
  const now = list.filter(a => a.active_now);
  const upcoming = list.filter(a => !a.active_now);
  const body = el.querySelector('[data-list]');
  if (!list.length) {
    body.innerHTML = emptyState({ icon: 'check', title: filter === 'accessibility' ? 'No accessibility alerts' : 'No alerts in this category', body: 'The MBTA isn’t reporting any alerts here right now.' });
  } else {
    body.innerHTML = `
      ${now.length ? `<section><h2 class="subhead">Happening now <span class="muted">· ${now.length}</span></h2><div class="stack-sm">${now.map(a => alertCard(a)).join('')}</div></section>` : ''}
      ${upcoming.length ? `<section><h2 class="subhead">Upcoming <span class="muted">· ${upcoming.length}</span></h2><div class="stack-sm">${upcoming.map(a => alertCard(a)).join('')}</div></section>` : ''}`;
  }
  el.querySelector('[data-updated]').textContent = updatedAt ? `Updated ${formatClockSeconds(updatedAt)}` : '';
}

async function load(el, { quiet = false } = {}) {
  if (!quiet && !store.get('alerts')) el.querySelector('[data-list]').innerHTML = `<div class="stack-sm">${skeleton(5, { height: 92 })}</div>`;
  const { data, error } = await api.alerts();
  if (!el.isConnected) return;
  if (error) {
    if (store.get('alerts')) return; // keep showing the last good list
    el.querySelector('[data-list]').innerHTML = errorState({ title: 'Couldn’t load alerts', body: error });
    el.querySelector('[data-action="retry"]')?.addEventListener('click', () => load(el));
    return;
  }
  updatedAt = data.updated_at;
  store.set('alerts', data.alerts);
  render(el, data.alerts);
}

export const alertsView = {
  enter(el, { query }) {
    if (FILTERS.some(f => f.id === query.filter)) filter = query.filter;
    el.innerHTML = `
      <div class="page page--narrow">
        <header class="page-head page-head--row">
          <div><h1>Service Alerts</h1><p class="muted">Live from the MBTA, ordered by what affects riders now.</p></div>
          <span class="muted small" data-updated></span>
        </header>
        <div class="segmented segmented--scroll" role="group" aria-label="Filter alerts" data-filters></div>
        <div class="alerts-list" data-list></div>
        <p class="fine-print">${icon('info', { size: 14 })}Severity is the MBTA’s own 0–10 rating. Accessibility alerts (elevators, escalators, facilities) are listed separately.</p>
      </div>`;
    el.querySelector('[data-filters]').addEventListener('click', ev => {
      const btn = ev.target.closest('[data-filter]');
      if (!btn) return;
      filter = btn.dataset.filter;
      render(el, store.get('alerts') || []);
    });
    if (store.get('alerts')) render(el, store.get('alerts'));
    load(el);
    timer = setInterval(() => { if (!document.hidden) load(el, { quiet: true }); }, REFRESH_MS);
  },
  leave() { clearInterval(timer); timer = null; },
};
