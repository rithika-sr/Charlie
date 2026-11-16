"""
Pure helpers for MBTA data: prediction timing/sorting and alert classification.

Kept free of Flask and network calls so they can be unit-tested directly.
"""

from datetime import timedelta
from dateutil import parser

# Effects that describe station accessibility rather than train/bus service.
ACCESSIBILITY_EFFECTS = {
    "ELEVATOR_CLOSURE", "ESCALATOR_CLOSURE", "ACCESS_ISSUE", "FACILITY_ISSUE",
    "PARKING_ISSUE", "PARKING_CLOSURE", "BIKE_ISSUE", "STATION_ISSUE",
}

# GTFS route_type -> mode key used by the frontend.
ROUTE_TYPE_MODES = {0: "subway", 1: "subway", 2: "commuter_rail", 3: "bus", 4: "ferry"}

# Predictions that are this far in the past are dropped from arrival boards.
DEPARTED_GRACE = timedelta(seconds=60)


def effective_time(attrs):
    """The time a prediction is displayed and sorted by: arrival, else departure."""
    return attrs.get("arrival_time") or attrs.get("departure_time")


def event_kind(attrs):
    """'arrival' when the train arrives, 'departure' when only a departure is known
    (e.g. trains leaving a terminal like Oak Grove)."""
    return "arrival" if attrs.get("arrival_time") else "departure"


def parse_time(value):
    try:
        return parser.isoparse(value) if value else None
    except (ValueError, TypeError):
        return None


def sort_by_time(items, key="time"):
    """Stable sort by an ISO timestamp field; items without a time go last."""
    def sort_key(item):
        dt = parse_time(item.get(key))
        return (0, dt.timestamp()) if dt else (1, 0)

    return sorted(items, key=sort_key)


def is_departed(dt, now):
    return dt is not None and dt < now - DEPARTED_GRACE


def is_cancelled(attrs):
    return (attrs.get("schedule_relationship") or "").upper() in {"CANCELLED", "SKIPPED", "NO_DATA"}


def alert_route_types(attrs):
    return sorted({e["route_type"] for e in attrs.get("informed_entity", []) if "route_type" in e})


def alert_modes(attrs):
    return sorted({ROUTE_TYPE_MODES[t] for t in alert_route_types(attrs) if t in ROUTE_TYPE_MODES})


def alert_is_active(attrs, now):
    """True when `now` falls inside one of the alert's active periods."""
    periods = attrs.get("active_period") or []
    if not periods:
        return True
    for period in periods:
        start = parse_time(period.get("start"))
        end = parse_time(period.get("end"))
        if (start is None or start <= now) and (end is None or now <= end):
            return True
    return False


def alert_payload(alert, now):
    """Normalize a raw MBTA alert into the shape the frontend uses."""
    attrs = alert.get("attributes", {})
    entities = attrs.get("informed_entity", [])
    routes = sorted({e["route"] for e in entities if e.get("route")})
    stops = sorted({e["stop"] for e in entities if e.get("stop")})
    effect = attrs.get("effect") or "UNKNOWN"
    return {
        "id": alert.get("id"),
        "effect": effect,
        "severity": attrs.get("severity", 0),
        "header": attrs.get("header", ""),
        "short_header": attrs.get("short_header", ""),
        "description": attrs.get("description") or "",
        "service_effect": attrs.get("service_effect") or "",
        "cause": attrs.get("cause") or "",
        "lifecycle": attrs.get("lifecycle") or "",
        "timeframe": attrs.get("timeframe") or "",
        "active_period": attrs.get("active_period") or [],
        "active_now": alert_is_active(attrs, now),
        "updated_at": attrs.get("updated_at"),
        "created_at": attrs.get("created_at"),
        "url": attrs.get("url") or "",
        "affected_routes": routes,
        "affected_stops": stops,
        "route_types": alert_route_types(attrs),
        "modes": alert_modes(attrs),
        "category": "accessibility" if effect in ACCESSIBILITY_EFFECTS else "service",
    }


def alert_priority(alert):
    """Sort key: active service alerts first, then by severity, then most recently updated."""
    updated = parse_time(alert.get("updated_at"))
    updated_ts = updated.timestamp() if updated else 0
    return (
        0 if alert.get("active_now") else 1,
        0 if alert.get("category") == "service" else 1,
        -(alert.get("severity") or 0),
        -updated_ts,
    )


# MBTA severity runs 0-10; 1-2 are informational notices (e.g. "premium parking is
# available at Braintree") that shouldn't mark a whole line as having a service alert.
STATUS_MIN_SEVERITY = 3


def line_status(alerts_for_line):
    """Summarize a line from its *distinct, currently active service* alerts."""
    active = [a for a in alerts_for_line
              if a.get("active_now") and a.get("category") == "service"
              and (a.get("severity") or 0) >= STATUS_MIN_SEVERITY]
    if any((a.get("severity") or 0) >= 7 for a in active):
        status = "alert"
    elif active:
        status = "warning"
    else:
        status = "normal"
    return status, active


# ==================== ARRIVAL ANSWERS ====================

def next_by_direction(predictions, per_direction=2):
    """Group time-sorted predictions by (route, direction) and keep the soonest few.

    Branch headsigns (e.g. Ashmont vs Braintree) stay on each item, and a direction's
    `headsigns` lists every branch seen so a rider can tell them apart."""
    groups = {}
    for p in predictions:
        key = (p.get("route_id"), p.get("direction_id"))
        g = groups.setdefault(key, {"route_id": key[0], "route_name": p.get("route_name", ""),
                                    "direction_id": key[1], "headsigns": [], "items": []})
        if p.get("headsign") and p["headsign"] not in g["headsigns"]:
            g["headsigns"].append(p["headsign"])
        if len(g["items"]) < per_direction:
            g["items"].append(p)
    return list(groups.values())


def filter_toward(predictions, toward, direction_id=None):
    """Predictions heading toward `toward` (matched on headsign), or in `direction_id`."""
    if direction_id is not None:
        return [p for p in predictions if p.get("direction_id") == direction_id]
    if not toward:
        return list(predictions)
    t = toward.lower().strip()

    def matches(p):
        h = (p.get("headsign") or "").lower()
        return bool(h) and (t in h or h in t)

    return [p for p in predictions if matches(p)]


# ==================== TRIP PLANNING ====================

# Alert effects that close or bypass a stop for riders.
CLOSURE_EFFECTS = {"STATION_CLOSURE", "STOP_CLOSURE", "SUSPENSION", "SHUTTLE", "NO_SERVICE", "DETOUR", "STOP_MOVED"}


def alert_active_between(attrs, start, end):
    """True when any active period overlaps [start, end] (alerts with no periods are always active)."""
    periods = attrs.get("active_period") or []
    if not periods:
        return True
    for period in periods:
        p_start = parse_time(period.get("start"))
        p_end = parse_time(period.get("end"))
        if (p_start is None or p_start <= end) and (p_end is None or start <= p_end):
            return True
    return False


def mbta_id(value):
    """Transitous prefixes MBTA GTFS ids with the feed name ('us-ma-MBTA_70012' -> '70012')."""
    value = str(value or "")
    return value.split("_", 1)[1] if value.startswith("us-ma-MBTA_") else value


# Effects that only matter to riders who board, alight or transfer at the stop. When
# trains simply pass a closed station, the rider's trip is not affected by them.
STOP_ONLY_EFFECTS = {"STATION_CLOSURE", "STOP_CLOSURE", "STOP_MOVED"}

SCOPE_RANK = {"stop": 0, "segment": 1, "direction": 2, "route": 3}


def match_leg_alerts(alerts, leg, min_severity=STATUS_MIN_SEVERITY):
    """Service alerts that affect one transit leg, labelled with the scope the feed supports.

    leg: {route_id, direction_id, stop_ids: stops the rider uses (board/alight),
          passed_stop_ids: stops the vehicle passes through, start, end}
    scope: 'stop'      — a stop the rider boards, alights or transfers at;
           'segment'   — disrupted service at a stop the vehicle passes (e.g. a suspension
                         or shuttle on the rider's segment), never a mere station closure;
           'direction' — this direction of the route (no stops named);
           'route'     — the feed names only the route."""
    matches = []
    used = set(leg.get("stop_ids") or [])
    passed = set(leg.get("passed_stop_ids") or []) - used
    for alert in alerts:
        attrs = alert.get("attributes", {})
        effect = attrs.get("effect") or ""
        if effect in ACCESSIBILITY_EFFECTS or (attrs.get("severity") or 0) < min_severity:
            continue
        if leg.get("start") and leg.get("end") and not alert_active_between(attrs, leg["start"], leg["end"]):
            continue
        best = None
        hit_stops = set()
        for ent in attrs.get("informed_entity", []):
            if ent.get("route") and ent["route"] != leg.get("route_id"):
                continue
            if ent.get("direction_id") is not None and leg.get("direction_id") is not None \
                    and ent["direction_id"] != leg["direction_id"]:
                continue
            stop = ent.get("stop")
            if stop:
                if stop in used:
                    scope = "stop"
                elif stop in passed and effect not in STOP_ONLY_EFFECTS:
                    scope = "segment"
                else:
                    continue
                hit_stops.add(stop)
            elif ent.get("route"):
                scope = "direction" if ent.get("direction_id") is not None else "route"
            else:
                continue
            if best is None or SCOPE_RANK[scope] < SCOPE_RANK[best]:
                best = scope
        if best is None:
            continue
        matches.append({
            "id": alert.get("id"),
            "effect": effect,
            "severity": attrs.get("severity") or 0,
            "header": attrs.get("header") or "",
            "service_effect": attrs.get("service_effect") or "",
            "scope": best,
            "stops": sorted(hit_stops),
        })
    matches.sort(key=lambda m: (SCOPE_RANK[m["scope"]], -m["severity"]))
    return matches


def station_closures(alerts, station_id, at_start, at_end):
    """Active closure-type service alerts naming a station (parent id like 'place-symcl')."""
    found = []
    for alert in alerts:
        attrs = alert.get("attributes", {})
        if attrs.get("effect") not in CLOSURE_EFFECTS:
            continue
        if not alert_active_between(attrs, at_start, at_end):
            continue
        ents = attrs.get("informed_entity", [])
        if not any(e.get("stop") == station_id for e in ents):
            continue
        found.append({
            "id": alert.get("id"),
            "effect": attrs.get("effect"),
            "severity": attrs.get("severity") or 0,
            "header": attrs.get("header") or "",
            "routes": sorted({e["route"] for e in ents if e.get("route")}),
            "direction_ids": sorted({e["direction_id"] for e in ents
                                     if e.get("stop") == station_id and e.get("direction_id") is not None}),
        })
    found.sort(key=lambda a: -a["severity"])
    return found
