"""
Journey planning shared by the Trip Planner (/api/trip-planner) and Charlie AI.

Both features call plan() with the same providers, so a trip asked in chat and the
same trip typed into the planner resolve the same places, try the same routing
services in the same order, carry the same alerts and return the same itineraries.

No Flask and no direct network calls: app.py injects the station table, the
geocoder, the routing providers and the alert matcher, which keeps this testable.

plan() result "code" is one of:
  ok                  itineraries found
  ambiguous           a typed name matches several stations; "candidates" lists them
  unresolved          a place couldn't be found in the Boston area; "unresolved" names the fields
  same_place          origin and destination are the same station
  no_itineraries      both places were found and a router answered, with no trips
  routing_unavailable every routing service failed (network/HTTP error), so nothing is known

A requested time (when, arrive_by) goes to providers that accept it. Trips that have
already left are dropped, and an "arrive by" that no trip can still make returns the
soonest arrivals with missed_deadline set.
"""

from datetime import datetime

import places


class RoutingError(RuntimeError):
    """A routing or geocoding service could not be reached or sent an unusable reply."""


FIELDS = ("origin", "destination")


def _coords(value):
    """(lat, lon) from 'lat,lon', else None."""
    try:
        lat, lon = (float(p) for p in str(value).split(","))
        return lat, lon
    except (TypeError, ValueError):
        return None


def resolve_endpoint(text, *, stop_id=None, coords=None, stations, aliases, station_by_id, landmark=None):
    """Local resolution only (no network): a station, explicit coordinates, a known
    landmark, a list of candidate stations, or 'needs geocoding'.

    landmark(text) -> {"name", "coords"} | None covers places riders name that the
    geocoder misses ("BU", "Fenway Park"); a station match still wins."""
    text = (text or "").strip()
    if stop_id:
        station = station_by_id(stop_id)
        if station:
            return {"status": "ok", "kind": "station", "query": text, **places.station_record(station["name"], stations)}
    point = _coords(coords)
    if point:
        return {"status": "ok", "kind": "place", "query": text, "name": text or "Selected location",
                "id": None, "lat": point[0], "lon": point[1], "lines": []}
    found = places.resolve_station(text, stations, aliases)
    if found["status"] == "ok":
        return {"status": "ok", "kind": "station", "query": text, "match": found["match"],
                "similar": found.get("similar", []), **places.station_record(found["name"], stations)}
    mark = landmark(text) if landmark and text else None
    point = _coords(mark["coords"]) if mark else None
    if point:
        return {"status": "ok", "kind": "place", "query": text, "name": mark["name"],
                "id": None, "lat": point[0], "lon": point[1], "lines": []}
    if found["status"] == "ambiguous":
        return {"status": "ambiguous", "query": text,
                "candidates": [places.station_record(n, stations) for n in found["candidates"]]}
    return {"status": "geocode", "query": text}


def _geocode(endpoint, geocode):
    """Fill in coordinates for free text. Unresolved -> status 'unresolved';
    geocoder down -> status 'unverified' (a provider may still understand the text)."""
    try:
        point = _coords(geocode(endpoint["query"]))
    except RoutingError as e:
        return {**endpoint, "status": "unverified", "kind": "place", "name": endpoint["query"],
                "id": None, "lat": None, "lon": None, "lines": [], "error": str(e)}
    if not point:
        return {**endpoint, "status": "unresolved"}
    return {**endpoint, "status": "ok", "kind": "place", "name": endpoint["query"],
            "id": None, "lat": point[0], "lon": point[1], "lines": []}


def public_endpoint(ep):
    keep = ("status", "kind", "query", "name", "id", "lat", "lon", "lines", "match", "similar", "candidates")
    return {k: ep[k] for k in keep if k in ep}


def plan(origin, destination, *, stations, aliases, station_by_id, geocode, providers, annotate,
         now=None, origin_id=None, dest_id=None, origin_coords=None, dest_coords=None,
         landmark=None, when=None, arrive_by=False):
    """Plan a journey. providers: [(name, fn(origin_ep, dest_ep, **time) -> routes)], tried
    in order; fn raises RoutingError when its service fails and returns [] when it found
    nothing. With a requested time, fn is called with when= and arrive_by=; the caller
    leaves out providers that can't plan for a time."""
    now = now or datetime.now()
    arrive_by = bool(arrive_by and when)
    resolve = lambda text, sid, xy: resolve_endpoint(text, stop_id=sid, coords=xy, stations=stations, aliases=aliases,
                                                     station_by_id=station_by_id, landmark=landmark)
    eps = {"origin": resolve(origin, origin_id, origin_coords), "destination": resolve(destination, dest_id, dest_coords)}
    result = {"origin": None, "destination": None, "routes": [], "notices": [], "alerts_checked": False,
              "provider": None, "providers_tried": [], "checked_at": now.isoformat(), "missed_deadline": False}

    def done(code, **extra):
        result.update({f: public_endpoint(eps[f]) for f in FIELDS})
        result.update(code=code, **extra)
        return result

    ambiguous = [f for f in FIELDS if eps[f]["status"] == "ambiguous"]
    if ambiguous:
        return done("ambiguous", ambiguous=ambiguous,
                    candidates={f: [c["name"] for c in eps[f]["candidates"]] for f in ambiguous})

    for f in FIELDS:
        if eps[f]["status"] == "geocode":
            eps[f] = _geocode(eps[f], geocode)
    unresolved = [f for f in FIELDS if eps[f]["status"] == "unresolved"]
    if unresolved:
        return done("unresolved", unresolved=unresolved)

    o, d = eps["origin"], eps["destination"]
    if o.get("id") and o.get("id") == d.get("id"):
        return done("same_place")

    cutoff = now.timestamp() - 60

    def route_with(time_kw):
        routes, answered, errors = [], [], []
        for name, fn in providers:
            if name not in result["providers_tried"]:
                result["providers_tried"].append(name)
            try:
                found = fn(o, d, **time_kw) or []
            except RoutingError as e:
                errors.append(str(e))
                continue
            answered.append(name)
            # Never offer a trip that has already left.
            found = [r for r in found if not isinstance(r.get("depart_ts"), (int, float)) or r["depart_ts"] >= cutoff]
            if found:
                routes, result["provider"] = found, name
                break
        return routes, answered, errors

    routes, answered, errors = route_with({"when": when, "arrive_by": arrive_by} if when else {})
    if arrive_by and answered and not routes:
        # No trip leaving now makes the deadline: say so and show the soonest arrivals.
        routes, answered, errors = route_with({})
        result["missed_deadline"] = bool(answered)

    if not answered:
        return done("routing_unavailable",
                    message=errors[-1] if errors else "No trip planning service is configured.")

    stations_used = [e if e.get("kind") == "station" else None for e in (o, d)]
    try:
        result["notices"] = annotate(routes, *stations_used) or []
        result["alerts_checked"] = True
    except Exception:  # alerts are supplementary; say they weren't checked rather than "none"
        result["notices"] = []
    for r in routes:
        r.pop("_legs", None)
    result["routes"] = routes
    return done("ok" if routes else "no_itineraries")
