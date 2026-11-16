// Reusable HTML builders. Every builder escapes data it renders.

import { escapeHtml as e, countdown, formatClock, activePeriodText, relativeTime, renderRichText } from '../core/format.js';
import { routeIdentity, LINES, inkFor } from '../core/constants.js';
import { effectLabel, severityBand } from '../core/alerts.js';
import { icon, modeIcon } from './icons.js';

/** Line badge: color plus a text code, so meaning never relies on color alone. */
export function lineBadge(routeId, { route = {}, size = 'md', label = true } = {}) {
  const id = routeIdentity(routeId, route);
  const text = inkFor(id.color);
  const code = LINES[routeId] && routeId.startsWith('Green-') ? `GL ${id.code}` : id.code;
  return `<span class="line-badge line-badge--${size}" style="--line:${id.color};--line-ink:${text}" title="${e(id.name)}">` +
    `<span class="line-badge__code" aria-hidden="true">${e(code)}</span>${label ? `<span class="sr-only">${e(id.name)}</span>` : ''}</span>`;
}

export function lineDot(routeId) {
  const id = routeIdentity(routeId);
  return `<span class="line-dot" style="--line:${id.bright}" aria-hidden="true"></span>`;
}

export function countdownBlock(pred, { large = false } = {}) {
  const c = countdown(pred);
  const [value, ...rest] = c.label.split(' ');
  const isNumber = /^\d+$/.test(value);
  return `<span class="countdown countdown--${c.state}${large ? ' countdown--lg' : ''}">` +
    (isNumber ? `<span class="countdown__value">${e(value)}</span><span class="countdown__unit">${e(rest.join(' '))}</span>`
              : `<span class="countdown__word">${e(c.label)}</span>`) +
    `</span>`;
}

const sameName = (a, b) => String(a || '').toLowerCase().replace(/[^a-z0-9]/g, '') === String(b || '').toLowerCase().replace(/[^a-z0-9]/g, '');

/** True when a trip's headsign is this station ("Harvard", "Harvard via Allston"): it ends here. */
export function endsAt(headsign, stationName) {
  if (!headsign || !stationName) return false;
  return sameName(headsign, stationName) || sameName(String(headsign).split(/\s+via\s+/i)[0], stationName);
}

/**
 * One row on an arrival board: destination first, then arrival/departure time,
 * platform/berth (when `platform`), status and countdown.
 * `showRoute` adds the line badge (mixed-mode lists); `stationName` marks trips that end here.
 */
export function arrivalRow(pred, { showRoute = false, platform = false, stationName = '' } = {}) {
  const clock = formatClock(pred.time);
  const endsHere = pred.event === 'arrival' && endsAt(pred.headsign, stationName);
  const verb = pred.event === 'departure' ? 'Departs' : 'Arrives';
  const status = pred.status && !/boarding/i.test(pred.status) ? `<span class="arrival__status">${e(pred.status)}</span>` : '';
  const meta = [`${verb} ${e(clock)}`, endsHere ? 'ends here' : '', platform && pred.platform_name ? e(pred.platform_name) : '', status].filter(Boolean).join(' · ');
  const badge = showRoute ? lineBadge(pred.route_id, { size: 'sm', route: { short_name: pred.route_short_name, name: pred.route_name } }) : '';
  return `<li class="arrival">
    ${badge ? `<span class="arrival__route">${badge}</span>` : ''}
    <div class="arrival__main">
      <span class="arrival__dest">${e(pred.headsign || pred.route_name || 'Train')}</span>
      <span class="arrival__meta">${meta}</span>
    </div>
    ${countdownBlock(pred)}
  </li>`;
}

export function alertCard(alert, { compact = false, open = false } = {}) {
  const band = severityBand(alert.severity);
  const routes = (alert.affected_routes || []).slice(0, compact ? 3 : 6);
  const extra = (alert.affected_routes || []).length - routes.length;
  const period = activePeriodText(alert.active_period);
  const title = alert.service_effect || effectLabel(alert.effect);
  const updated = alert.updated_at ? `Updated ${relativeTime(alert.updated_at)}` : '';
  const source = alert.url || 'https://www.mbta.com/alerts';
  const body = alert.description ? `<div class="alert-card__desc"><p>${renderRichText(alert.description)}</p></div>` : '';
  const tag = compact ? 'div' : 'details';
  return `<${tag} class="alert-card alert-card--${band}" ${!compact && open ? 'open' : ''}>
    ${compact ? '<div class="alert-card__summary">' : '<summary class="alert-card__summary">'}
      <span class="alert-card__bar" aria-hidden="true"></span>
      <div class="alert-card__content">
        <div class="alert-card__top">
          <span class="severity severity--${band}">${icon(alert.category === 'accessibility' ? 'accessible' : 'alert', { size: 14 })}${e(effectLabel(alert.effect))}</span>
          <span class="alert-card__routes">${routes.map(r => lineBadge(r, { size: 'sm' })).join('')}${extra > 0 ? `<span class="chip chip--muted">+${extra}</span>` : ''}</span>
        </div>
        <div class="alert-card__title">${e(title)}</div>
        <p class="alert-card__header">${e(alert.header)}</p>
        <div class="alert-card__meta">
          ${period ? `<span>${icon('clock', { size: 14 })}${e(period)}</span>` : ''}
          ${!alert.active_now ? '<span class="chip chip--muted">Upcoming</span>' : ''}
          ${!compact && updated ? `<span>${e(updated)}</span>` : ''}
          <span class="sr-only">Severity ${e(alert.severity)} of 10</span>
        </div>
      </div>
      ${compact ? '' : `<span class="alert-card__chevron">${icon('chevronDown', { size: 18 })}</span>`}
    ${compact ? '</div>' : '</summary>'}
    ${compact ? '' : `<div class="alert-card__body">${body}
      <div class="alert-card__footer"><span>Severity ${e(alert.severity)}/10${alert.cause ? ` · Cause: ${e(alert.cause.toLowerCase().replace(/_/g, ' '))}` : ''}</span>
      <a class="link" href="${e(source)}" target="_blank" rel="noopener">Official MBTA alert ${icon('external', { size: 14 })}</a></div></div>`}
  </${tag}>`;
}

export function emptyState({ icon: name = 'info', title, body = '', action = '' }) {
  return `<div class="state state--empty">${icon(name, { size: 28 })}<h3>${e(title)}</h3>${body ? `<p>${e(body)}</p>` : ''}${action}</div>`;
}

export function errorState({ title = 'Something went wrong', body = '', retry = true }) {
  return `<div class="state state--error" role="alert">${icon('wifiOff', { size: 28 })}<h3>${e(title)}</h3>${body ? `<p>${e(body)}</p>` : ''}` +
    (retry ? `<button class="btn btn--secondary" data-action="retry">${icon('refresh', { size: 16 })}Try again</button>` : '') + `</div>`;
}

export function skeleton(rows = 3, { height = 44 } = {}) {
  return `<div class="skeleton-group" aria-busy="true" aria-label="Loading">${Array.from({ length: rows }, () =>
    `<div class="skeleton" style="height:${height}px"></div>`).join('')}</div>`;
}

export function sectionHeader(title, { action = '', id = '' } = {}) {
  return `<div class="section-head"><h2 class="section-title" ${id ? `id="${id}"` : ''}>${e(title)}</h2>${action}</div>`;
}

export function stationRow(station, { right = '', recent = false } = {}) {
  return `<button class="station-row" data-station="${e(station.id)}">
    <span class="station-row__icon">${icon(recent ? 'history' : 'pin', { size: 18 })}</span>
    <span class="station-row__name">${e(station.name)}</span>
    <span class="station-row__lines">${(station.lines || []).map(l => lineBadge(l, { size: 'sm' })).join('')}</span>
    ${right || icon('chevronRight', { size: 18, className: 'station-row__chev' })}
  </button>`;
}

const BADGE_HELP = {
  'Earliest arrival': 'Gets you there first among these options',
  'Shortest trip': 'Lowest total travel time among these options',
  'Fewest transfers': 'Fewest changes between vehicles among these options',
  'Least walking': 'Least time on foot among these options',
};

const dayKey = new Intl.DateTimeFormat('en-CA', { timeZone: 'America/New_York', year: 'numeric', month: '2-digit', day: '2-digit' });
const weekday = new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', weekday: 'short' });

/** Clock time from an epoch (seconds), with the day when it isn't today in Boston. */
export function tripClock(ts, fallback = '', now = new Date()) {
  if (!Number.isFinite(ts)) return fallback;
  const d = new Date(ts * 1000);
  const label = formatClock(d);
  if (dayKey.format(d) === dayKey.format(now)) return label;
  const tomorrow = new Date(now.getTime() + 86400000);
  return `${label} ${dayKey.format(d) === dayKey.format(tomorrow) ? 'tomorrow' : weekday.format(d)}`;
}

const STEP_MODE = { subway: 'subway', tram: 'subway', 'silver-line': 'bus', bus: 'bus', 'commuter-rail': 'commuter_rail', ferry: 'ferry', transit: 'subway' };

/** Trip option card. `badges` holds transparent labels such as "Shortest trip". */
export function tripOptionCard(route, index, { badges = [], alerts = [], note = '' } = {}) {
  const segments = route.steps.map((s, i) => {
    const color = s.color || '#64748B';
    return `${i ? `<span class="seg-sep" aria-hidden="true">${icon('chevronRight', { size: 14 })}</span>` : ''}<span class="seg" style="--seg:${e(color)};--seg-ink:${inkFor(color)}">${modeIcon(STEP_MODE[s.type] || 'subway', { size: 14 })}${e(s.short_name || s.name)}</span>`;
  }).join('');
  const transfers = route.transfers === 0 ? 'No transfers' : `${route.transfers} transfer${route.transfers > 1 ? 's' : ''}`;
  const steps = (route.detailed_steps || []).map(st => `
    <li class="tstep tstep--${st.type}" style="--step:${e(st.color || 'var(--border-strong)')}">
      <span class="tstep__icon">${icon(st.type === 'walk' ? 'walk' : 'train', { size: 16 })}</span>
      <div><div class="tstep__title">${e(st.instruction)}</div>
      ${st.details ? `<div class="tstep__sub">${e(st.details)}</div>` : ''}
      ${st.time || st.duration ? `<div class="tstep__time">${e([st.time, st.duration].filter(Boolean).join(' · '))}</div>` : ''}</div>
    </li>`).join('');
  return `<details class="trip-card" ${index === 0 ? 'open' : ''}>
    <summary class="trip-card__summary">
      <div class="trip-card__time">
        <span class="trip-card__duration">${e(route.duration_text || `${route.duration} min`)}</span>
        ${route.depart_time || Number.isFinite(route.depart_ts)
          ? `<span class="trip-card__clock"><span class="muted">Leave</span> ${e(tripClock(route.depart_ts, route.depart_time))}</span>
             <span class="trip-card__clock"><span class="muted">Arrive</span> <strong>${e(tripClock(route.arrive_ts, route.arrive_time))}</strong></span>`
          : `<span class="trip-card__range">${e(route.time_range || '')}</span>`}
      </div>
      <div class="trip-card__segs">${segments}</div>
      <div class="trip-card__facts">
        ${badges.map(b => `<span class="chip chip--accent" title="${e(BADGE_HELP[b] || '')}">${e(b)}</span>`).join('')}
        <span>${e(transfers)}</span>
        ${route.walk_time ? `<span>${icon('walk', { size: 14 })}${e(route.walk_time)} min walk</span>` : ''}
        ${route.fare ? `<span title="Estimated fare">${e(route.fare)} est.</span>` : ''}
      </div>
      ${note ? `<div class="trip-card__note">${icon('clock', { size: 14 })}${e(note)}</div>` : ''}
      ${alerts.length ? `<div class="trip-card__alerts">${icon('alert', { size: 14 })}<ul class="plain-list">${alerts.map(a => `<li>${e(a)}</li>`).join('')}</ul></div>` : ''}
      <span class="trip-card__chev">${icon('chevronDown', { size: 18 })}</span>
    </summary>
    <ol class="trip-card__steps">${steps}</ol>
  </details>`;
}
