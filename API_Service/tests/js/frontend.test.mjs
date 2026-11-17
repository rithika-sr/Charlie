// Frontend logic tests — run with: node --test tests/js/
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { rank, normalize, scoreMatch } from '../../static/js/core/search.js';
import { countdown, escapeHtml, activePeriodText, renderRichText, vehicleStatusText } from '../../static/js/core/format.js';
import { filterAlerts, countByFilter, prioritize, alertsForRoutes, severityBand } from '../../static/js/core/alerts.js';
import { decodePolyline } from '../../static/js/core/polyline.js';
import { routeIdentity, LINES, SUBWAY_IDS } from '../../static/js/core/constants.js';
import { parseHash } from '../../static/js/core/router.js';
import { routeIdsForStep, tripBadges } from '../../static/js/views/trip.js';
import { mentions } from '../../static/js/views/assistant.js';
import { groupArrivals } from '../../static/js/ui/station-panel.js';

// The real station list the app serves (extracted from app.py's STATION_DATA).
const APP = readFileSync(new URL('../../app.py', import.meta.url), 'utf8');
const STATIONS = [...APP.matchAll(/"([^"]+)": \{"id": "(place-[a-z0-9]+)", "lat": ([\d.-]+), "lon": ([\d.-]+), "lines": \[([^\]]*)\]\}/g)]
  .map(m => ({ name: m[1], id: m[2], lines: m[5].replace(/"/g, '').split(', ').filter(Boolean) }))
  .filter((s, i, all) => all.findIndex(x => x.id === s.id) === i);

test('station list fixture covers all 126 stations', () => {
  assert.equal(STATIONS.length, 126);
});

test('search: every station is findable by its own name and ranks first', () => {
  for (const s of STATIONS) {
    const top = rank(STATIONS, s.name, x => x.name, 3);
    assert.ok(top.some(x => x.id === s.id), `${s.name} not found`);
  }
});

test('search: partial, abbreviated and fuzzy queries', () => {
  assert.equal(rank(STATIONS, 'harv', x => x.name, 1)[0].name, 'Harvard');
  assert.equal(rank(STATIONS, 'park st', x => x.name, 1)[0].name, 'Park Street');
  assert.equal(rank(STATIONS, 'oak', x => x.name, 1)[0].name, 'Oak Grove');
  assert.ok(rank(STATIONS, 'dwntn', x => x.name, 3).some(x => x.name === 'Downtown Crossing'));
  assert.equal(rank(STATIONS, 'zzzz', x => x.name).length, 0);
  assert.equal(normalize("St. Mary's St"), 'street marys street');
  assert.equal(scoreMatch('', 'Harvard'), null);
});

test('countdown: minutes, arriving/departing, boarding, hours, missing', () => {
  const now = new Date('2026-11-16T13:00:00Z');
  const at = m => new Date(now.getTime() + m * 60000).toISOString();
  assert.equal(countdown({ time: at(4.5), event: 'arrival' }, now).label, '4 min');
  assert.equal(countdown({ time: at(0.4), event: 'arrival' }, now).label, 'Arriving');
  assert.equal(countdown({ time: at(0.4), event: 'departure' }, now).label, 'Departing');
  assert.equal(countdown({ time: at(9), status: 'Boarding' }, now).label, 'Boarding');
  assert.equal(countdown({ time: at(75) }, now).label, '1 hr 15 min');
  assert.equal(countdown({}, now).label, 'No prediction');
  assert.equal(countdown({ time: at(2) }, now).state, 'soon');
});

test('format: escaping and safe rich text', () => {
  assert.equal(escapeHtml('<b>"x"</b>'), '&lt;b&gt;&quot;x&quot;&lt;/b&gt;');
  assert.equal(renderRichText('**Next** <script>'), '<strong>Next</strong> &lt;script&gt;');
  assert.equal(vehicleStatusText('STOPPED_AT', 'Park Street'), 'Stopped at Park Street');
  const now = new Date('2026-11-16T13:00:00Z');
  assert.match(activePeriodText([{ start: '2026-11-16T12:00:00Z', end: null }], now), /Ongoing/);
  assert.match(activePeriodText([{ start: '2026-11-20T12:00:00Z', end: '2026-11-22T12:00:00Z' }], now), /^Starts /);
});

const ALERTS = [
  { id: 'red', category: 'service', active_now: true, severity: 7, modes: ['subway'], affected_routes: ['Red'], updated_at: '2026-11-16T10:00:00Z' },
  { id: 'bus', category: 'service', active_now: true, severity: 3, modes: ['bus'], affected_routes: ['66'], updated_at: '2026-11-16T11:00:00Z' },
  { id: 'cr', category: 'service', active_now: false, severity: 8, modes: ['commuter_rail'], affected_routes: ['CR-Needham'] },
  { id: 'ferry', category: 'service', active_now: true, severity: 5, affected_routes: ['Boat-F1'] },
  { id: 'elev', category: 'accessibility', active_now: true, severity: 9, modes: ['subway'], affected_routes: ['Red'] },
];

test('alerts: filters by mode and keeps accessibility separate', () => {
  assert.deepEqual(filterAlerts(ALERTS, 'all').map(a => a.id), ['red', 'bus', 'cr', 'ferry']);
  assert.deepEqual(filterAlerts(ALERTS, 'subway').map(a => a.id), ['red']);
  assert.deepEqual(filterAlerts(ALERTS, 'ferry').map(a => a.id), ['ferry']);   // mode inferred from route id
  assert.deepEqual(filterAlerts(ALERTS, 'accessibility').map(a => a.id), ['elev']);
  assert.deepEqual(countByFilter(ALERTS), { all: 4, subway: 1, bus: 1, commuter_rail: 1, ferry: 1, accessibility: 1 });
});

test('alerts: priority and route matching', () => {
  assert.deepEqual(prioritize(ALERTS).map(a => a.id), ['red', 'ferry', 'bus', 'elev', 'cr']);
  assert.deepEqual(alertsForRoutes(ALERTS, ['Red']).map(a => a.id), ['red']);
  assert.equal(severityBand(8), 'major');
  assert.equal(severityBand(4), 'moderate');
  assert.equal(severityBand(1), 'minor');
});

test('polyline: decodes the reference example', () => {
  const pts = decodePolyline('_p~iF~ps|U_ulLnnqC_mqNvxq`@');
  assert.deepEqual(pts, [[38.5, -120.2], [40.7, -120.95], [43.252, -126.453]]);
});

test('constants: identities for subway, bus, rail and ferry', () => {
  assert.equal(SUBWAY_IDS.length, 8);
  for (const id of SUBWAY_IDS) assert.ok(LINES[id].name && LINES[id].color);
  assert.equal(routeIdentity('CR-Worcester', { name: 'Framingham/Worcester Line' }).mode, 'commuter_rail');
  assert.equal(routeIdentity('Boat-F1').mode, 'ferry');
  assert.equal(routeIdentity('66', { short_name: '66' }).name, 'Route 66');
  assert.equal(routeIdentity('741', { short_name: 'SL1', name: 'x' }).group, 'Silver');
});

test('router: parses views, params and queries', () => {
  assert.deepEqual(parseHash('#/lines/Green-B'), { name: 'lines', params: ['Green-B'], query: {} });
  assert.deepEqual(parseHash('#/map?route=66'), { name: 'map', params: [], query: { route: '66' } });
  assert.equal(parseHash('#/nope').name, 'overview');
  assert.equal(parseHash('').name, 'overview');
});

test('trip: step → route ids and transparent badges', () => {
  assert.deepEqual(routeIdsForStep({ name: 'Green Line D' }), ['Green-D']);
  assert.deepEqual(routeIdsForStep({ name: 'Red Line' }), ['Red']);
  assert.deepEqual(routeIdsForStep({ name: 'Bus 66' }), ['66']);
  assert.deepEqual(routeIdsForStep({ name: 'Framingham/Worcester Line' }), []);
  const badges = tripBadges([{ duration: 40, transfers: 3, walk_time: 5 }, { duration: 33, transfers: 2, walk_time: 5 }, { duration: 48, transfers: 1, walk_time: 5 }]);
  assert.deepEqual(badges, [[], ['Shortest trip'], ['Fewest transfers']]);
  assert.deepEqual(tripBadges([{ duration: 10, transfers: 0 }]), [[]]);
});

test('assistant: detects stations and lines mentioned in replies', () => {
  const found = mentions('🚇 From **Harvard** on the Red Line: next train to Alewife', STATIONS);
  assert.ok(found.some(m => m.type === 'station' && m.name === 'Harvard'));
  assert.ok(found.some(m => m.type === 'station' && m.name === 'Alewife'));
  assert.ok(found.some(m => m.type === 'line' && m.id === 'Red'));
});

test('station board: groups arrivals by line and direction, keeping order', () => {
  const groups = groupArrivals([
    { route_id: 'Red', direction_id: 1, headsign: 'Alewife', time: '1' },
    { route_id: 'Red', direction_id: 0, headsign: 'Ashmont', time: '2' },
    { route_id: 'Red', direction_id: 1, headsign: 'Alewife', time: '3' },
    { route_id: 'Red', direction_id: 0, headsign: 'Braintree', time: '4' },
  ]);
  assert.equal(groups.length, 2);
  assert.deepEqual(groups[0].items.map(i => i.time), ['1', '3']);
  assert.deepEqual([...groups[1].headsigns], ['Ashmont', 'Braintree']);
});

import { inkFor } from '../../static/js/core/constants.js';
test('badges: text color keeps WCAG contrast on every line color', () => {
  assert.equal(inkFor('#ED8B00'), '#1A1A1A');   // Orange Line
  assert.equal(inkFor('#FFC72C'), '#1A1A1A');   // bus yellow
  assert.equal(inkFor('#003DA5'), '#FFFFFF');   // Blue Line
  assert.equal(inkFor('#80276C'), '#FFFFFF');   // Commuter Rail
  assert.equal(inkFor('#00843D'), '#FFFFFF');   // Green Line
});

import { sortTrips, arrivalNote, tripAlertText, SORTS } from '../../static/js/views/trip.js';
import { createRequestGate } from '../../static/js/core/request-gate.js';
import { tripClock, arrivalRow } from '../../static/js/ui/components.js';
import { searchResults, parseRouteQuery } from '../../static/js/core/search.js';
import { groupHeading, modesPresent } from '../../static/js/ui/station-panel.js';
import { clusterPoints } from '../../static/js/core/cluster.js';
import { arrivalsCardHtml } from '../../static/js/views/assistant.js';

// Park Street → Symphony as observed: the shortest ride leaves later and arrives later.
const T0 = Date.parse('2026-11-16T16:40:00Z') / 1000;
const TRIPS = [
  { id: 'later-short', depart_ts: T0 + 16 * 60, arrive_ts: T0 + 26 * 60, duration: 10, transfers: 1, walk_time: 6 },
  { id: 'first-arrival', depart_ts: T0 + 2 * 60, arrive_ts: T0 + 20 * 60, duration: 18, transfers: 0, walk_time: 9 },
  { id: 'mid', depart_ts: T0 + 5 * 60, arrive_ts: T0 + 23 * 60, duration: 18, transfers: 0, walk_time: 3 },
];

test('trip: immediate trips default to earliest arrival, not duration', () => {
  assert.equal(SORTS[0].id, 'arrival');
  assert.deepEqual(sortTrips(TRIPS).map(x => x.id), ['first-arrival', 'mid', 'later-short']);
  assert.deepEqual(sortTrips(TRIPS, 'duration').map(x => x.id), ['later-short', 'first-arrival', 'mid']);
  assert.deepEqual(sortTrips(TRIPS, 'transfers').map(x => x.id), ['first-arrival', 'mid', 'later-short']);
  assert.deepEqual(sortTrips(TRIPS, 'walking').map(x => x.id), ['mid', 'later-short', 'first-arrival']);
});

test('trip: same arrival time ties fall through transfers → walking → exact time → duration → API order', () => {
  const r = [{ id: 'a', arrive_ts: 6000, duration: 20, transfers: 1, walk_time: 4 }, { id: 'b', arrive_ts: 6000, duration: 20, transfers: 0, walk_time: 9 },
    { id: 'c', arrive_ts: 6000, duration: 15, transfers: 2, walk_time: 1 }, { id: 'd', arrive_ts: 6000, duration: 20, transfers: 0, walk_time: 9 }];
  assert.deepEqual(sortTrips(r).map(x => x.id), ['b', 'd', 'a', 'c']);
  // Without timestamps, earliest arrival falls back to transfers/walking/duration rather than guessing.
  assert.deepEqual(sortTrips([{ id: 'x', duration: 30, transfers: 0 }, { id: 'y', duration: 12, transfers: 0 }]).map(x => x.id), ['y', 'x']);
});

test('trip: same displayed minute prefers fewer transfers; an earlier minute still wins', () => {
  const m = Date.parse('2026-11-16T18:24:00Z') / 1000;               // 1:24 PM Boston
  const oneTransfer = { id: 'transfer', arrive_ts: m + 5, duration: 13, transfers: 1, walk_time: 7 };
  const direct = { id: 'direct', arrive_ts: m + 50, duration: 15, transfers: 0, walk_time: 10 };
  assert.equal(tripClock(oneTransfer.arrive_ts, '', new Date(m * 1000)), tripClock(direct.arrive_ts, '', new Date(m * 1000)));
  assert.deepEqual(sortTrips([oneTransfer, direct]).map(x => x.id), ['direct', 'transfer']);
  assert.deepEqual(tripBadges([oneTransfer, direct])[1], ['Earliest arrival', 'Fewest transfers']);
  // One displayed minute earlier beats fewer transfers.
  const earlier = { ...oneTransfer, id: 'earlier', arrive_ts: m - 10 };    // 1:23:50 PM
  assert.deepEqual(sortTrips([direct, earlier]).map(x => x.id), ['earlier', 'direct']);
  assert.ok(tripBadges([direct, earlier])[1].includes('Earliest arrival'));
  // Same minute, same transfers: less walking first.
  const walkLess = { ...direct, id: 'walkless', arrive_ts: m + 58, walk_time: 3 };
  assert.deepEqual(sortTrips([direct, walkLess]).map(x => x.id), ['walkless', 'direct']);
  // Other sorts keep their own rules.
  assert.deepEqual(sortTrips([oneTransfer, direct], 'duration').map(x => x.id), ['transfer', 'direct']);
});

test('trip: overnight arrivals sort by full timestamp across midnight', () => {
  const late = Date.parse('2026-11-17T04:59:30Z') / 1000;            // 11:59:30 PM Boston
  const beforeMidnight = { id: 'before', arrive_ts: late, duration: 30, transfers: 2, walk_time: 9 };
  const afterMidnight = { id: 'after', arrive_ts: late + 60, duration: 10, transfers: 0, walk_time: 1 };  // 12:00:30 AM
  assert.deepEqual(sortTrips([afterMidnight, beforeMidnight]).map(x => x.id), ['before', 'after']);
  assert.equal(tripClock(afterMidnight.arrive_ts, '', new Date(late * 1000)), '12:00 AM tomorrow');
});

test('trip: midnight boundary sorts by timestamp and labels the next day', () => {
  const late = Date.parse('2026-11-17T04:50:00Z') / 1000;           // 11:50 PM Boston
  const r = [{ id: 'after', arrive_ts: late + 25 * 60, duration: 20 }, { id: 'before', arrive_ts: late + 5 * 60, duration: 40 }];
  assert.deepEqual(sortTrips(r).map(x => x.id), ['before', 'after']);
  const now = new Date('2026-11-17T04:40:00Z');                       // 11:40 PM Boston on the 16th
  assert.equal(tripClock(late + 5 * 60, '', now), '11:55 PM');
  assert.equal(tripClock(late + 25 * 60, '', now), '12:15 AM tomorrow');
});

test('trip: badges and the shortest-but-later note are honest', () => {
  const badges = tripBadges(TRIPS);
  assert.deepEqual(badges, [['Shortest trip'], ['Earliest arrival'], ['Least walking']]);
  assert.match(arrivalNote(TRIPS[0], TRIPS), /Shortest ride, but leaves later and arrives 6 min after/);
  assert.equal(arrivalNote(TRIPS[1], TRIPS), '');
  assert.equal(arrivalNote(TRIPS[2], TRIPS), '');                 // later, but not the shortest
  // No trade-off when an earliest-arriving option is just as short.
  const tie = [{ arrive_ts: 100, duration: 10 }, { arrive_ts: 400, duration: 10 }];
  assert.equal(arrivalNote(tie[1], tie), '');
});

test('trip: alert text says the scope the feed supports', () => {
  assert.match(tripAlertText({ step: 'Green Line E', scope: 'stop', effect: 'STATION_CLOSURE', stop_names: ['Symphony', 'Symphony'], header: 'Symphony station is closed.' }), /^Green Line E · Station closed at Symphony: /);
  assert.match(tripAlertText({ step: 'Red Line', scope: 'route', header: 'Delays of about 10 minutes.' }), /^Red Line \(line-wide\): Delays/);
  assert.match(tripAlertText({ step: 'Orange Line', scope: 'direction', header: 'Northbound trains bypass State.' }), /\(this direction\)/);
  assert.match(tripAlertText({ step: 'Green Line E', scope: 'segment', effect: 'SUSPENSION', stop_names: ['Symphony'], header: 'No service.' }), /Suspension at Symphony, which your ride passes through/);
});

test('search: plain and natural route queries find the exact route first', () => {
  const routes = { subway: [{ id: 'Red', name: 'Red Line', short_name: '' }], commuter_rail: [], ferry: [],
    bus: [{ id: '39', short_name: '39', name: 'Forest Hills - Back Bay' }, { id: '393', short_name: '393', name: 'Lynn' }, { id: '9', short_name: '9', name: 'City Point' }] };
  for (const q of ['39', 'Route 39', 'Bus 39', 'route  39', 'rt 39', '#39']) {
    const out = searchResults(q, STATIONS, routes);
    assert.equal(out[0].kind, 'route', q);
    assert.equal(out[0].id, '39', q);
  }
  assert.deepEqual(parseRouteQuery('Bus 39'), { core: '39', explicit: true });
  const harvard = searchResults('Harvard', STATIONS, routes);
  assert.equal(harvard[0].kind, 'station');
  assert.equal(harvard[0].name, 'Harvard');
});

test('station board: subway first, headsign headings, berths as supporting text', () => {
  const preds = [
    { route_id: '71', route_type: 3, direction_id: 0, headsign: 'Watertown Square', platform_name: 'Upper Busway Berth B4', time: '1' },
    { route_id: 'Red', route_type: 1, direction_id: 0, headsign: 'Braintree', platform_name: 'Ashmont/Braintree', time: '2' },
    { route_id: '86', route_type: 3, direction_id: 1, headsign: 'Harvard', platform_name: 'Upper Busway Berth B3', time: '3', event: 'arrival' },
    { route_id: 'Red', route_type: 1, direction_id: 0, headsign: 'Ashmont', platform_name: 'Ashmont/Braintree', time: '4' },
  ];
  const groups = groupArrivals(preds);
  assert.deepEqual(groups.map(g => g.route_id), ['Red', '71', '86']);
  assert.deepEqual(modesPresent(preds), ['subway', 'bus']);
  assert.deepEqual(groupHeading(groups[0], 'Harvard'), { dest: 'To Braintree / Ashmont', platform: '', rowPlatform: false });
  const bus = groupHeading(groups[1], 'Harvard');
  assert.equal(bus.dest, 'To Watertown Square');
  assert.equal(bus.platform, 'Upper Busway Berth B4');
  assert.equal(groupHeading(groups[2], 'Harvard').dest, 'Arriving at Harvard (ends here)');
  const via = groupArrivals([{ route_id: '66', route_type: 3, direction_id: 1, headsign: 'Harvard via Allston', event: 'arrival', time: '1' }]);
  assert.equal(groupHeading(via[0], 'Harvard').dest, 'Arriving at Harvard (ends here)');
  const row = arrivalRow({ ...preds[2], time: new Date(Date.now() + 5 * 60000).toISOString() }, { stationName: 'Harvard', platform: true, showRoute: true });
  assert.match(row, /arrival__dest">Harvard</);
  assert.match(row, /ends here · Upper Busway Berth B3/);
  assert.match(row, /line-badge/);
});

test('map: clustering merges only nearby vehicles on the same line', () => {
  const c = clusterPoints([
    { id: 'r1', x: 0, y: 0, group: 'Red' }, { id: 'r2', x: 10, y: 5, group: 'Red' },
    { id: 'o1', x: 4, y: 2, group: 'Orange' }, { id: 'r3', x: 200, y: 0, group: 'Red' },
  ], 26);
  assert.deepEqual(c.map(x => [x.group, x.ids]), [['Red', ['r1', 'r2']], ['Orange', ['o1']], ['Red', ['r3']]]);
});

test('assistant: arrivals card keeps the station board shortcut and names each state', () => {
  const ok = arrivalsCardHtml({ status: 'ok', station: { id: 'place-harsq', name: 'Harvard' }, updated_at: '2026-11-16T16:52:00Z',
    groups: [{ route_id: 'Red', label: 'Ashmont/Braintree', items: [{ headsign: 'Braintree', time: new Date(Date.now() + 120000).toISOString(), event: 'arrival' }] }] });
  assert.match(ok, /data-open-station="place-harsq"/);
  assert.match(ok, /To Ashmont\/Braintree/);
  assert.match(ok, /Braintree/);
  assert.match(arrivalsCardHtml({ status: 'ambiguous', candidates: ['Park Street', 'Science Park'] }), /data-suggest="Next train at Science Park"/);
  assert.match(arrivalsCardHtml({ status: 'unavailable', station: { id: 'x', name: 'Harvard' } }), /unavailable/);
  assert.match(arrivalsCardHtml({ status: 'none', station: { id: 'x', name: 'Harvard' } }), /Service alerts/);
});

test('trip: an older or invalidated request can never overwrite newer state', () => {
  const gate = createRequestGate();
  const first = gate.start();
  const second = gate.start();                       // rapid re-plan
  assert.equal(first.signal.aborted, true);
  assert.equal(gate.isCurrent(first.token), false);
  assert.equal(gate.isCurrent(second.token), true);
  gate.invalidate();                                 // rider edits the destination mid-flight
  assert.equal(second.signal.aborted, true);
  assert.equal(gate.isCurrent(second.token), false);
  const third = gate.start();
  assert.equal(gate.isCurrent(third.token), true);
});

import { tripCardsHtml } from '../../static/js/views/assistant.js';
test('assistant: trip replies render every itinerary card plus a Trip Planner shortcut', () => {
  const t0 = Date.parse('2026-11-16T22:00:00Z') / 1000;
  const route = (id, arrive, transfers) => ({ summary: id, duration: 30, duration_text: '30 min', transfers, walk_time: 5, arrive_ts: t0 + arrive * 60, depart_ts: t0,
    depart_time: '5:00 PM', arrive_time: '', steps: [{ name: id, short_name: id, color: '#00843D', type: 'subway' }], detailed_steps: [] });
  const html = tripCardsHtml({ code: 'ok', origin: 'Chiswick Rd', origin_name: 'Chiswick Road', destination_name: 'Hynes Convention Center',
    routes: [route('B', 40, 0), route('D', 33, 0), route('65', 33, 1)], notices: [] }, 3);
  assert.equal((html.match(/class="trip-card"/g) || []).length, 3);
  assert.ok(html.indexOf('>D<') < html.indexOf('>65<') && html.indexOf('>65<') < html.indexOf('>B<'));   // earliest arrival, then fewer transfers
  assert.match(html, /data-plan-trip="3"/);
  assert.match(tripCardsHtml({ code: 'unresolved', origin: 'zzzzz', routes: [] }, 1), /Open in Trip Planner/);
  assert.equal(tripCardsHtml({ code: 'need_origin', routes: [] }, 1), '');
});

// ---------- Leave now / Depart at / Arrive by ----------
import { bostonParts, suggestedParts, whenText, WHEN_MODES } from '../../static/js/ui/trip-form.js';
import { whenSummaryHtml } from '../../static/js/views/trip.js';
import { nearCardHtml } from '../../static/js/views/assistant.js';

test('trip time: Boston wall clock regardless of the browser time zone', () => {
  const at = new Date('2026-11-16T22:55:00Z');              // 5:55 PM in Boston (EST)
  assert.deepEqual(bostonParts(0, at), { date: '2026-11-16', time: '17:55' });
  assert.deepEqual(bostonParts(10, at), { date: '2026-11-16', time: '18:05' });
  assert.deepEqual(bostonParts(0, new Date('2026-07-01T16:00:00Z')), { date: '2026-07-01', time: '12:00' });   // EDT
  assert.deepEqual(suggestedParts(10, new Date('2026-11-16T22:51:00Z')), { date: '2026-11-16', time: '18:05' }); // rounded up to 5 min
  assert.deepEqual(bostonParts(10, new Date('2026-11-17T04:55:00Z')), { date: '2026-11-17', time: '00:05' });   // past midnight
});

test('trip time: labels for each mode', () => {
  const at = new Date('2026-11-16T22:55:00Z');
  assert.deepEqual(WHEN_MODES.map(m => m.id), ['now', 'depart', 'arrive']);
  assert.equal(whenText(null, at), 'Leaving now');
  assert.equal(whenText({ mode: 'depart', time: '2026-11-16T18:05' }, at), 'Leave today at 6:05 PM');
  assert.equal(whenText({ mode: 'arrive', time: '2026-11-17T09:00' }, at), 'Arrive by 9:00 AM tomorrow');
  assert.equal(whenText({ mode: 'depart', time: '2026-11-17T00:00:00-05:00' }, at), 'Leave tomorrow at 12:00 AM');
  assert.match(whenSummaryHtml({ missedDeadline: true, when: { mode: 'arrive', time: '2026-11-16T18:00' } }), /No trip gets you there by that time/);
});

test('assistant: nearest-stop card shows stop, walk and too-soon warnings, or asks for location', () => {
  const card = nearCardHtml({ kind: 'near', status: 'ok', updated_at: new Date().toISOString(), groups: [{
    route_id: '39', route_name: 'Route 39', label: 'Back Bay Station', stop_id: '1357', stop_name: '360 Huntington Ave',
    distance_m: 47, walk_min: 1, items: [
      { headsign: 'Back Bay', time: new Date(Date.now() + 30000).toISOString(), event: 'arrival', catchable: false },
      { headsign: 'Back Bay', time: new Date(Date.now() + 14 * 60000).toISOString(), event: 'arrival', catchable: true }] }] }, 3);
  assert.match(card, /360 Huntington Ave<\/button> · 47 m · ~1 min walk/);
  assert.match(card, /data-open-station="1357"/);
  assert.equal((card.match(/Too soon to walk there/g) || []).length, 1);
  const ask = nearCardHtml({ kind: 'near', status: 'need_location' }, 5);
  assert.match(ask, /data-share-loc="5"/);
});

test('trip: arrive-by results can list the latest departure first', () => {
  const r = (id, dep, arr, transfers = 0) => ({ id, depart_ts: dep, arrive_ts: arr, duration: (arr - dep) / 60, transfers });
  const routes = [r('a', 1000, 2000), r('b', 1600, 2500), r('c', 1300, 2200), r('d', 1600, 2400, 1)];
  assert.deepEqual(sortTrips(routes, 'departure').map(x => x.id), ['b', 'd', 'c', 'a']);   // tie → fewer transfers
});

// ---------- live vehicle icons and motion ----------
import { vehicleKind, vehicleIconHtml, nextRotation, lerpLatLng, metersBetween, MarkerMotion } from '../../static/js/map/vehicle-icons.js';

test('vehicles: each mode gets its own glyph, colored for contrast', () => {
  assert.deepEqual(['Red', 'Orange', 'Blue', 'Green-B', 'Mattapan', 'CR-Worcester', 'Boat-F1', '39', '741'].map(vehicleKind),
    ['subway', 'subway', 'subway', 'lightrail', 'lightrail', 'train', 'ferry', 'bus', 'bus']);
  const bus = vehicleIconHtml({ color: '#FFC72C', routeId: '39', bearing: 90, code: '39' });
  assert.match(bus, /veh--bus/); assert.match(bus, /--ink:#1A1A1A/); assert.match(bus, /--rot:90deg/); assert.match(bus, /veh__tag[^>]*>39</);
  const red = vehicleIconHtml({ color: '#DA291C', routeId: 'Red', bearing: null });
  assert.match(red, /veh--subway veh--still/); assert.match(red, /--ink:#FFFFFF/); assert.doesNotMatch(red, /veh__tag/);
});

test('vehicles: heading turns the short way round', () => {
  assert.equal(nextRotation(350, 10), 370);
  assert.equal(nextRotation(10, 350), -10);
  assert.equal(nextRotation(720, 90), 810);
  assert.equal(nextRotation(45, null), 45);
  assert.equal(nextRotation(NaN, 200), 200);
});

test('vehicles: markers glide to each live position at constant speed', () => {
  let clock = 0;
  const frames = [];
  const motion = new MarkerMotion({ now: () => clock, raf: cb => { frames.push(cb); return frames.length; }, caf: () => {}, reducedMotion: false });
  const marker = { ll: { lat: 42.35, lng: -71.06 }, getLatLng() { return this.ll; }, setLatLng(p) { this.ll = { lat: p[0], lng: p[1] }; } };
  const run = t => { clock = t; frames.shift()(t); };
  assert.equal(motion.move(marker, [42.36, -71.06], 10000), 'glide');           // ~1.1 km
  run(5000);
  assert.ok(Math.abs(marker.ll.lat - 42.355) < 1e-9, 'halfway at half time');
  // A new update mid-glide starts from where the marker is drawn now: no jump back.
  assert.equal(motion.move(marker, [42.37, -71.06], 10000), 'glide');
  run(5040);
  assert.ok(marker.ll.lat > 42.355 && marker.ll.lat < 42.3551);
  run(15100);
  assert.deepEqual([marker.ll.lat, marker.ll.lng], [42.37, -71.06]);
  assert.equal(motion.tracks.size, 0);
  assert.equal(motion.move(marker, [42.37001, -71.06], 10000), 'none');          // GPS jitter
  assert.equal(motion.move(marker, [42.6, -71.06], 10000), 'snap');              // implausible jump
  assert.equal(marker.ll.lat, 42.6);
  assert.ok(Math.abs(metersBetween([42.35, -71.06], [42.36, -71.06]) - 1113) < 2);
  assert.deepEqual(lerpLatLng([0, 0], [10, 20], 2), [10, 20]);
});

test('vehicles: reduced motion snaps instead of gliding', () => {
  const motion = new MarkerMotion({ now: () => 0, raf: () => 1, caf: () => {}, reducedMotion: true });
  const marker = { ll: { lat: 42.35, lng: -71.06 }, getLatLng() { return this.ll; }, setLatLng(p) { this.ll = { lat: p[0], lng: p[1] }; } };
  assert.equal(motion.move(marker, [42.351, -71.06], 10000), 'snap');
  assert.equal(marker.ll.lat, 42.351);
});
