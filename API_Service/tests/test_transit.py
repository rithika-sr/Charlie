"""Unit tests for prediction timing/sorting and alert classification."""

from datetime import datetime, timedelta

import pytz

import transit

BOSTON = pytz.timezone("America/New_York")
NOW = BOSTON.localize(datetime(2026, 11, 16, 8, 0, 0))


def iso(minutes):
    return (NOW + timedelta(minutes=minutes)).isoformat()


def test_effective_time_prefers_arrival_then_departure():
    assert transit.effective_time({"arrival_time": iso(5), "departure_time": iso(6)}) == iso(5)
    # Terminal stations (e.g. Oak Grove) only publish a departure time.
    assert transit.effective_time({"arrival_time": None, "departure_time": iso(7)}) == iso(7)
    assert transit.effective_time({}) is None


def test_event_kind():
    assert transit.event_kind({"arrival_time": iso(1)}) == "arrival"
    assert transit.event_kind({"arrival_time": None, "departure_time": iso(1)}) == "departure"


def test_sort_by_time_handles_departure_only_nulls_and_ties():
    items = [
        {"id": "a", "time": iso(49)},
        {"id": "b", "time": iso(24)},
        {"id": "nil", "time": None},
        {"id": "c", "time": iso(34)},
        {"id": "tie1", "time": iso(10)},
        {"id": "tie2", "time": iso(10)},
    ]
    assert [i["id"] for i in transit.sort_by_time(items)] == ["tie1", "tie2", "b", "c", "a", "nil"]


def test_sort_by_time_compares_across_utc_offsets():
    later_utc = {"id": "utc", "time": (NOW + timedelta(minutes=3)).astimezone(pytz.utc).isoformat()}
    sooner_local = {"id": "local", "time": iso(2)}
    assert [i["id"] for i in transit.sort_by_time([later_utc, sooner_local])] == ["local", "utc"]


def test_departed_and_cancelled():
    assert transit.is_departed(NOW - timedelta(minutes=2), NOW)
    assert not transit.is_departed(NOW - timedelta(seconds=30), NOW)
    assert transit.is_cancelled({"schedule_relationship": "CANCELLED"})
    assert transit.is_cancelled({"schedule_relationship": "SKIPPED"})
    assert not transit.is_cancelled({"schedule_relationship": None})


def raw_alert(**attrs):
    base = {"effect": "DELAY", "severity": 5, "header": "h", "informed_entity": [], "active_period": []}
    base.update(attrs)
    return {"id": attrs.pop("id", "1"), "attributes": base}


def test_alert_payload_classifies_and_dedupes_routes():
    alert = raw_alert(effect="ELEVATOR_CLOSURE", informed_entity=[
        {"route": "Red", "route_type": 1, "stop": "70075"},
        {"route": "Red", "route_type": 1, "stop": "70076"},
        {"route": "Green-B", "route_type": 0, "stop": "70196"},
    ])
    p = transit.alert_payload(alert, NOW)
    assert p["category"] == "accessibility"
    assert p["affected_routes"] == ["Green-B", "Red"]
    assert p["modes"] == ["subway"]
    assert p["affected_stops"] == ["70075", "70076", "70196"]


def test_alert_active_now_uses_active_periods():
    current = raw_alert(active_period=[{"start": iso(-60), "end": iso(60)}])
    future = raw_alert(active_period=[{"start": iso(60), "end": iso(120)}])
    open_ended = raw_alert(active_period=[{"start": iso(-60), "end": None}])
    assert transit.alert_payload(current, NOW)["active_now"]
    assert not transit.alert_payload(future, NOW)["active_now"]
    assert transit.alert_payload(open_ended, NOW)["active_now"]


def test_alert_priority_orders_active_service_by_severity():
    def a(id_, sev, active=True, category="service"):
        return {"id": id_, "severity": sev, "active_now": active, "category": category, "updated_at": iso(-5)}
    alerts = [a("access", 9, category="accessibility"), a("upcoming", 10, active=False), a("minor", 3), a("major", 8)]
    assert [x["id"] for x in sorted(alerts, key=transit.alert_priority)] == ["major", "minor", "access", "upcoming"]


def test_line_status_ignores_accessibility_and_inactive_alerts():
    access = {"severity": 9, "active_now": True, "category": "accessibility"}
    upcoming = {"severity": 9, "active_now": False, "category": "service"}
    assert transit.line_status([access, upcoming])[0] == "normal"
    assert transit.line_status([{"severity": 4, "active_now": True, "category": "service"}])[0] == "warning"
    assert transit.line_status([{"severity": 7, "active_now": True, "category": "service"}])[0] == "alert"
    # Informational notices (severity 1-2) don't change a line's status.
    assert transit.line_status([{"severity": 1, "active_now": True, "category": "service"}])[0] == "normal"
