// Formatting helpers. Pure functions (no DOM) so they can be unit-tested in Node.

const HTML_ESCAPES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };

export function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, c => HTML_ESCAPES[c]);
}

export function toDate(value) {
  if (!value) return null;
  const d = value instanceof Date ? value : new Date(value);
  return Number.isNaN(d.getTime()) ? null : d;
}

const clockFmt = new Intl.DateTimeFormat('en-US', { hour: 'numeric', minute: '2-digit', timeZone: 'America/New_York' });
const clockSecFmt = new Intl.DateTimeFormat('en-US', { hour: 'numeric', minute: '2-digit', second: '2-digit', timeZone: 'America/New_York' });
const dayFmt = new Intl.DateTimeFormat('en-US', { weekday: 'long', month: 'long', day: 'numeric', timeZone: 'America/New_York' });
const shortDayFmt = new Intl.DateTimeFormat('en-US', { weekday: 'short', month: 'short', day: 'numeric', timeZone: 'America/New_York' });
const hourFmt = new Intl.DateTimeFormat('en-US', { hour: 'numeric', hourCycle: 'h23', timeZone: 'America/New_York' });

export const formatClock = v => { const d = toDate(v); return d ? clockFmt.format(d) : ''; };
export const formatClockSeconds = v => { const d = toDate(v); return d ? clockSecFmt.format(d) : ''; };
export const formatDay = v => { const d = toDate(v); return d ? dayFmt.format(d) : ''; };
export const formatShortDay = v => { const d = toDate(v); return d ? shortDayFmt.format(d) : ''; };

/** "Good morning/afternoon/evening" in Boston time. */
export function greeting(now = new Date()) {
  const hour = Number(hourFmt.format(now)) % 24;
  if (hour >= 5 && hour < 12) return 'Good morning';
  if (hour >= 12 && hour < 17) return 'Good afternoon';
  return 'Good evening';
}

/**
 * Countdown for a prediction. Only uses states the data supports:
 * - status text from the MBTA (e.g. "Boarding", "Stopped 2 stops away") wins;
 * - under a minute: "Arriving" for arrivals, "Departing" for departure-only trips;
 * - otherwise minutes (or hours) until the event.
 */
export function countdown(pred, now = new Date()) {
  const status = (pred.status || '').trim();
  if (/boarding/i.test(status)) return { label: 'Boarding', minutes: 0, state: 'boarding' };
  const t = toDate(pred.time || pred.arrival_time);
  if (!t) return { label: status || 'No prediction', minutes: null, state: 'unknown' };
  const minutes = Math.floor((t.getTime() - now.getTime()) / 60000);
  if (minutes < 1) {
    const departing = pred.event === 'departure';
    return { label: departing ? 'Departing' : 'Arriving', minutes: 0, state: 'now' };
  }
  if (minutes >= 60) {
    const h = Math.floor(minutes / 60), m = minutes % 60;
    return { label: m ? `${h} hr ${m} min` : `${h} hr`, minutes, state: 'later' };
  }
  return { label: `${minutes} min`, minutes, state: minutes <= 3 ? 'soon' : 'later' };
}

export function relativeTime(value, now = new Date()) {
  const d = toDate(value);
  if (!d) return '';
  const sec = Math.round((now.getTime() - d.getTime()) / 1000);
  if (sec < 45) return 'just now';
  const min = Math.round(sec / 60);
  if (min < 60) return `${min} min ago`;
  const hr = Math.round(min / 60);
  if (hr < 24) return `${hr} hr ago`;
  return formatShortDay(d);
}

/** Human description of an alert's active periods. */
export function activePeriodText(periods, now = new Date()) {
  if (!Array.isArray(periods) || periods.length === 0) return '';
  const current = periods.find(p => {
    const s = toDate(p.start), e = toDate(p.end);
    return (!s || s <= now) && (!e || now <= e);
  });
  const fmt = d => `${formatShortDay(d)}, ${formatClock(d)}`;
  if (current) {
    const end = toDate(current.end);
    return end ? `Until ${fmt(end)}` : 'Ongoing until further notice';
  }
  const next = periods.map(p => toDate(p.start)).filter(d => d && d > now).sort((a, b) => a - b)[0];
  return next ? `Starts ${fmt(next)}` : '';
}

/** "STOPPED_AT" -> "Stopped at" */
export function vehicleStatusText(status, stopName) {
  const map = { STOPPED_AT: 'Stopped at', INCOMING_AT: 'Arriving at', IN_TRANSIT_TO: 'In transit to' };
  const verb = map[status] || '';
  if (verb && stopName) return `${verb} ${stopName}`;
  return verb || '';
}

export function titleCase(value) {
  return String(value || '').toLowerCase().replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

/** Very small, safe markdown subset for assistant replies (bold, italics, line breaks). */
export function renderRichText(text) {
  return escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*])\*(?!\s)(.+?)\*(?!\*)/g, '$1<em>$2</em>')
    .replace(/\n{2,}/g, '</p><p>')
    .replace(/\n/g, '<br>');
}
