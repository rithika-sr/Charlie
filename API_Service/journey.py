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


def resolve_endpoint(text, *, stop_id=None, coords=None, stations, aliases, station_by_id):
    """Local resolution only (no network): a station, explicit coordinates, a list of
    candidate stations, or 'needs geocoding'."""
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
         now=None, origin_id=None, dest_id=None, origin_coords=None, dest_coords=None):
    """Plan a journey. providers: [(name, fn(origin_ep, dest_ep) -> routes)], tried in order;
    fn raises RoutingError when its service fails and returns [] when it found nothing."""
    now = now or datetime.now()
    eps = {
        "origin": resolve_endpoint(origin, stop_id=origin_id, coords=origin_coords, stations=stations,
                                   aliases=aliases, station_by_id=station_by_id),
        "destination": resolve_endpoint(destination, stop_id=dest_id, coords=dest_coords, stations=stations,
                                        aliases=aliases, station_by_id=station_by_id),
    }
    result = {"origin": None, "destination": None, "routes": [], "notices": [], "alerts_checked": False,
              "provider": None, "providers_tried": [], "checked_at": now.isoformat()}

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

    routes, answered, errors = [], [], []
    for name, fn in providers:
        result["providers_tried"].append(name)
        try:
            found = fn(o, d) or []
        except RoutingError as e:
            errors.append(str(e))
            continue
        answered.append(name)
        if found:
            routes, result["provider"] = found, name
            break

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
