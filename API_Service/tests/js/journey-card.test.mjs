// Charlie AI journey cards — run with: node --test tests/js/*.test.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { journeyCardHtml, journeyAction, actionLabel, journeyEndpoints } from '../../static/js/ui/journey-card.js';
import { sortTrips } from '../../static/js/views/trip.js';

const T0 = Math.floor(Date.parse('2026-11-16T20:00:00Z') / 1000);
const route = (id, dep, arr, transfers, walk, alerts = []) => ({
  duration: (arr - dep), duration_text: `${arr - dep} min`, transfers, walk_time: walk,
  depart_time: `id-${id}-dep`, arrive_time: `id-${id}-arr`, depart_ts: T0 + dep * 60, arrive_ts: T0 + arr * 60,
  fare: '$2.40', summary: `route ${id}`, alerts,
  steps: [{ type: 'tram', name: 'Green Line B', short_name: 'B', color: '#00843D' }],
  detailed_steps: [{ type: 'transit', instruction: `Take Green Line B (${id})`, details: 'Board at Chiswick Road', time: '', duration: '10 min' }],
});
const station = (name, id) => ({ status: 'ok', kind: 'station', query: name.toLowerCase(), name, id, lat: 42.34, lon: -71.15, lines: ['Green-B'] });

function okJourney() {
  return {
    code: 'ok', provider: 'transitous', checked_at: '2026-11-16T20:00:00Z', sort: 'arrival', alerts_checked: true,
    origin: station('Chiswick Road', 'place-chswk'),
    destination: { ...station('Harvard', 'place-harsq'), similar: ['Harvard Ave'] },
    request: { origin: { name: 'chiswick rd' }, destination: { name: 'harvard' } },
    routes: [route('late', 5, 50, 0, 2), route('best', 2, 41, 1, 4, [{ step: 'Bus 66', scope: 'route', header: 'Bus 66 detour', effect: 'DETOUR' }])],
    notices: [], caveats: [{ kind: 'time', text: 'I can only plan trips leaving now.' }],
  };
}

test('journey card: best option first, alternatives collapsed, facts straight from the data', () => {
  const j = okJourney();
  const html = journeyCardHtml(j, 3);
  const order = sortTrips(j.routes, 'arrival').map(r => `Take Green Line B (${r.summary.split(' ')[1]})`);
  assert.deepEqual(order, ['Take Green Line B (best)', 'Take Green Line B (late)']);
  assert.ok(html.indexOf(order[0]) > 0 && html.indexOf(order[0]) < html.indexOf(order[1]), 'same ranking as the Trip Planner');
  const clock = ts => new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', hour: 'numeric', minute: '2-digit' }).format(new Date(ts * 1000));
  const best = j.routes[1];
  assert.ok(html.includes(clock(best.arrive_ts)), 'arrival estimate from the provider timestamp');
  assert.match(html, /1 more option/);
  assert.match(html, /Bus 66 detour/);                                // relevant alert on the option
  assert.match(html, /Transitous · checked/);
  assert.match(html, /I can only plan trips leaving now/);            // caveat shown
  assert.ok(!/<details class="trip-card" open/.test(html), 'steps start collapsed in chat');
  for (const a of ['refresh', 'reverse', 'planner', 'similar']) assert.match(html, new RegExp(`data-journey-action="${a}"`));
  assert.match(html, /Did you mean Harvard Ave\?/);
});

test('journey card: each non-ok outcome is distinct and invents no routes', () => {
  const base = { ...okJourney(), routes: [] };
  const outage = journeyCardHtml({ ...base, code: 'routing_unavailable' }, 0);
  assert.match(outage, /routing service didn’t answer/);
  assert.match(outage, /data-journey-action="refresh"/);
  assert.ok(!/trip-card/.test(outage));
  const empty = journeyCardHtml({ ...base, code: 'no_itineraries' }, 0);
  assert.match(empty, /data-code="no_itineraries"/);
  assert.ok(!/trip-card/.test(empty));
  const amb = journeyCardHtml({ ...base, code: 'ambiguous', ambiguous: ['origin'], candidates: { origin: ['Park Street', 'Science Park'] } }, 0,
    { stations: [{ name: 'Park Street', lines: ['Red'] }] });
  assert.match(amb, /data-journey-action="pick" data-field="origin" data-name="Park Street"/);
  assert.match(amb, /Science Park/);
  assert.match(journeyCardHtml({ ...base, code: 'unresolved', unresolved: ['origin'] }, 0), /Browse stations/);
  assert.match(journeyCardHtml({ code: 'needs_origin', request: { destination: { name: 'Hynes' } } }, 0), /Use my location/);
  assert.equal(journeyCardHtml(null, 0), '');
});

test('journey card: names are escaped', () => {
  const j = okJourney();
  j.origin = { ...j.origin, name: '<img src=x onerror=alert(1)>' };
  assert.ok(!journeyCardHtml(j, 0).includes('<img src=x'));
});

test('journey actions: exact resolved endpoints for refresh, reverse and picks', () => {
  const j = okJourney();
  const ends = journeyEndpoints(j);
  assert.deepEqual(ends.origin, { name: 'Chiswick Road', id: 'place-chswk', lat: 42.34, lon: -71.15 });
  assert.deepEqual(journeyAction(j, 'refresh'), { type: 'trip', kind: 'refresh', origin: ends.origin, destination: ends.destination, sort: undefined });
  const back = journeyAction(j, 'reverse');
  assert.equal(back.origin.name, 'Harvard');
  assert.equal(back.destination.name, 'Chiswick Road');
  assert.deepEqual(journeyAction(j, 'similar', { field: 'destination', name: 'Harvard Ave' }).destination, { name: 'Harvard Ave' });
  const amb = { code: 'ambiguous', origin: { status: 'ambiguous', query: 'park' }, destination: station('Harvard', 'place-harsq'), request: { origin: { name: 'park' }, destination: { name: 'harvard' } } };
  const pick = journeyAction(amb, 'pick', { field: 'origin', name: 'Science Park' });
  assert.deepEqual([pick.origin, pick.destination.id], [{ name: 'Science Park' }, 'place-harsq']);
  const loc = journeyAction({ code: 'needs_origin', request: { destination: { name: 'Hynes' } } }, 'locate', { location: { lat: 42.3, lon: -71.1 } });
  assert.deepEqual(loc.origin, { name: 'Your location', lat: 42.3, lon: -71.1 });
  assert.equal(actionLabel(journeyAction(j, 'refresh')), 'Refresh: Chiswick Road → Harvard');
  assert.equal(journeyAction(j, 'planner'), null);
});

test('chat layout: the message scroller contains absolutely positioned labels', async () => {
  // Without this, .sr-only spans in line badges stretched the page on phones and the composer drifted.
  const { readFileSync } = await import('node:fs');
  const css = readFileSync(new URL('../../static/css/app.css', import.meta.url), 'utf8');
  assert.match(css, /\.chat__scroll\s*\{[^}]*position:\s*relative/);
});
