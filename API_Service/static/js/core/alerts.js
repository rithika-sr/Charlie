// Alert filtering and labels. Pure functions.

const EFFECT_LABELS = {
  SHUTTLE: 'Shuttle buses', SUSPENSION: 'Suspension', DELAY: 'Delays', DETOUR: 'Detour',
  STATION_CLOSURE: 'Station closed', STOP_CLOSURE: 'Stop closed', STOP_MOVED: 'Stop moved',
  SERVICE_CHANGE: 'Service change', SCHEDULE_CHANGE: 'Schedule change', TRACK_CHANGE: 'Track change',
  CANCELLATION: 'Cancellation', NO_SERVICE: 'No service', MODIFIED_SERVICE: 'Modified service',
  ELEVATOR_CLOSURE: 'Elevator closed', ESCALATOR_CLOSURE: 'Escalator closed', ACCESS_ISSUE: 'Access issue',
  STATION_ISSUE: 'Station issue', FACILITY_ISSUE: 'Facility issue', PARKING_CLOSURE: 'Parking closed',
  PARKING_ISSUE: 'Parking issue', BIKE_ISSUE: 'Bike facility issue', SNOW_ROUTE: 'Snow route',
  EXTRA_SERVICE: 'Extra service', POLICY_CHANGE: 'Policy change', SUMMARY: 'Summary',
};

export function effectLabel(effect) {
  if (EFFECT_LABELS[effect]) return EFFECT_LABELS[effect];
  return String(effect || 'Alert').toLowerCase().replace(/_/g, ' ').replace(/^\w/, c => c.toUpperCase());
}

/** Visual band for the MBTA's 0–10 severity (the number itself is always shown too). */
export function severityBand(severity) {
  const s = Number(severity) || 0;
  if (s >= 7) return 'major';
  if (s >= 4) return 'moderate';
  return 'minor';
}

/** Fallback mode detection from route ids when `modes` is missing. */
export function modesForAlert(alert) {
  if (Array.isArray(alert.modes) && alert.modes.length) return alert.modes;
  const modes = new Set();
  for (const r of alert.affected_routes || []) {
    if (/^(Red|Orange|Blue|Green|Mattapan)/.test(r)) modes.add('subway');
    else if (r.startsWith('CR-')) modes.add('commuter_rail');
    else if (r.startsWith('Boat-')) modes.add('ferry');
    else modes.add('bus');
  }
  return [...modes];
}

/**
 * filter: 'all' | 'subway' | 'bus' | 'commuter_rail' | 'ferry' | 'accessibility'
 * 'all' and the mode filters show service alerts; 'accessibility' shows elevator/escalator/facility alerts.
 */
export function filterAlerts(alerts, filter = 'all') {
  return (alerts || []).filter(a => {
    const isAccess = a.category === 'accessibility';
    if (filter === 'accessibility') return isAccess;
    if (isAccess) return false;
    if (filter === 'all') return true;
    return modesForAlert(a).includes(filter);
  });
}

export function countByFilter(alerts) {
  const filters = ['all', 'subway', 'bus', 'commuter_rail', 'ferry', 'accessibility'];
  return Object.fromEntries(filters.map(f => [f, filterAlerts(alerts, f).length]));
}

/** Active-now first, then severity, then most recently updated (mirrors the API's sort). */
export function prioritize(alerts) {
  return [...(alerts || [])].sort((a, b) =>
    (b.active_now === true) - (a.active_now === true) ||
    (a.category === 'service' ? 0 : 1) - (b.category === 'service' ? 0 : 1) ||
    (b.severity || 0) - (a.severity || 0) ||
    String(b.updated_at || '').localeCompare(String(a.updated_at || '')));
}

/** Active service alerts that affect any of the given route ids. */
export function alertsForRoutes(alerts, routeIds, minSeverity = 3) {
  const ids = new Set(routeIds);
  return (alerts || []).filter(a => a.category === 'service' && a.active_now && (a.severity || 0) >= minSeverity &&
    (a.affected_routes || []).some(r => ids.has(r)));
}
