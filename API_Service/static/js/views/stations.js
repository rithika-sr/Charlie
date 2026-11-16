// Stations: search all 126 subway stations, filter by line, open live arrivals.

import { store } from '../core/store.js';
import { LINE_GROUPS, LINES } from '../core/constants.js';
import { escapeHtml as e } from '../core/format.js';
import { rank } from '../core/search.js';
import { stationRow, emptyState } from '../ui/components.js';
import { openStation } from '../ui/station-panel.js';

let state = { query: '', group: 'all' };

function matchesGroup(st, group) {
  if (group === 'all') return true;
  return (st.lines || []).some(l => LINES[l] && LINES[l].group === group);
}

function results() {
  const all = store.get('stations').filter(s => matchesGroup(s, state.group));
  if (state.query.trim()) return rank(all, state.query, s => s.name, 200);
  return [...all].sort((a, b) => a.name.localeCompare(b.name));
}

function renderList(el) {
  const list = results();
  const listEl = el.querySelector('[data-list]');
  el.querySelector('[data-count]').textContent = `${list.length} station${list.length === 1 ? '' : 's'}`;
  listEl.innerHTML = list.length
    ? list.map(s => `<li>${stationRow(s)}</li>`).join('')
    : `<li>${emptyState({ icon: 'search', title: 'No stations match', body: 'Try part of the name, like “harv” or “downtown”.' })}</li>`;
}

function renderRecents(el) {
  const recents = store.get('recentStations').map(id => store.stationById(id)).filter(Boolean);
  const box = el.querySelector('[data-recents]');
  box.hidden = !recents.length || Boolean(state.query);
  box.querySelector('ul').innerHTML = recents.map(s => `<li>${stationRow(s, { recent: true })}</li>`).join('');
}

export const stationsView = {
  enter(el) {
    el.innerHTML = `
      <div class="page">
        <header class="page-head"><h1>Stations</h1><p class="muted">Live arrivals, accessibility and alerts for every subway and light-rail station.</p></header>
        <div class="toolbar">
          <div class="search-field">
            <label class="sr-only" for="station-q">Search stations</label>
            <input id="station-q" class="input" type="search" placeholder="Search 126 stations" value="${e(state.query)}" autocomplete="off" />
          </div>
          <div class="segmented" role="group" aria-label="Filter by line">
            <button class="segmented__btn" data-group="all" aria-pressed="${state.group === 'all'}">All</button>
            ${LINE_GROUPS.map(g => `<button class="segmented__btn" data-group="${g.id}" aria-pressed="${state.group === g.id}"><span class="swatch" style="--c:${LINES[g.routes[0]].bright}"></span>${e(g.name)}</button>`).join('')}
          </div>
        </div>
        <section class="card" data-recents hidden><h2 class="subhead subhead--pad">Recently viewed</h2><ul class="station-list"></ul></section>
        <section class="card">
          <div class="list-head"><h2 class="subhead" data-count role="status"></h2></div>
          <ul class="station-list" data-list></ul>
        </section>
      </div>`;
    const input = el.querySelector('#station-q');
    input.addEventListener('input', () => { state.query = input.value; renderList(el); renderRecents(el); });
    el.querySelectorAll('[data-group]').forEach(btn => btn.addEventListener('click', () => {
      state.group = btn.dataset.group;
      el.querySelectorAll('[data-group]').forEach(b => b.setAttribute('aria-pressed', String(b === btn)));
      renderList(el);
    }));
    el.addEventListener('click', ev => {
      const row = ev.target.closest('[data-station]');
      if (row) openStation(row.dataset.station);
    });
    renderList(el);
    renderRecents(el);
    this.unsub = store.subscribe(key => { if (key === 'recentStations') renderRecents(el); });
  },
  leave() { this.unsub?.(); },
};
