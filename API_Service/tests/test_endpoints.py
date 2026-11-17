"""Endpoint tests with the MBTA API mocked out (no network)."""

from datetime import datetime, timedelta

import pytest

import app as charlie


class FakeResponse:
    def __init__(self, data, status=200):
        self._data = data
        self.status_code = status

    def json(self):
        return self._data


@pytest.fixture
def client(monkeypatch):
    charlie._mbta_cache.clear()
    charlie.app.config["TESTING"] = True
    return charlie.app.test_client()


def iso(minutes):
    return (charlie.get_now() + timedelta(minutes=minutes)).isoformat()


def route_mbta(monkeypatch, routes):
    """routes: {path_prefix: data or callable(params)}"""
    calls = []

    def fake_get(url, headers=None, params=None, timeout=None):
        path = url.replace(charlie.BASE_URL, "")
        calls.append((path, params))
        for prefix, data in routes.items():
            if path.startswith(prefix):
                value = data(params) if callable(data) else data
                return value if isinstance(value, FakeResponse) else FakeResponse(value)
        return FakeResponse({"data": []})

    monkeypatch.setattr(charlie.requests, "get", fake_get)
    return calls


def prediction(pid, arrival, departure, route="Orange", headsign="Forest Hills", direction=0, status=None):
    return {
        "id": pid,
        "attributes": {"arrival_time": arrival, "departure_time": departure, "direction_id": direction,
                       "status": status, "schedule_relationship": None},
        "relationships": {"route": {"data": {"id": route}}, "trip": {"data": {"id": f"t-{pid}"}},
                          "stop": {"data": {"id": "70036"}}},
    }


def test_station_sorts_terminal_departures_and_drops_departed(client, monkeypatch):
    preds = [
        prediction("p49", None, iso(49)),
        prediction("p34", None, iso(34)),
        prediction("p24", None, iso(24)),
        prediction("gone", None, iso(-5)),
        prediction("p2", iso(2), iso(3), status="Boarding"),
    ]
    included = [{"type": "route", "id": "Orange", "attributes": {"long_name": "Orange Line", "color": "ED8B00", "type": 1}}]
    included += [{"type": "trip", "id": f"t-{p['id']}", "attributes": {"headsign": "Forest Hills"}} for p in preds]
    route_mbta(monkeypatch, {
        "/stops/place-ogmnl": {"data": {"attributes": {"name": "Oak Grove", "wheelchair_boarding": 1}}},
        "/predictions": {"data": preds, "included": included},
        "/alerts": {"data": []},
    })
    data = client.get("/api/station/place-ogmnl").get_json()
    events = [p["event"] for p in data["predictions"]]
    times = [p["time"] for p in data["predictions"]]
    assert times == sorted(times, key=lambda t: datetime.fromisoformat(t))
    assert len(data["predictions"]) == 4            # departed train removed
    assert events == ["arrival", "departure", "departure", "departure"]
    assert data["predictions"][0]["status"] == "Boarding"
    assert data["station"]["wheelchair_boarding"] == 1
    assert data["station"]["lines"] == ["Orange"]


def test_dashboard_counts_each_alert_once_per_line(client, monkeypatch):
    entities = [{"route": "Red", "route_type": 1, "stop": str(s)} for s in range(30)]
    alerts = [
        {"id": "svc", "attributes": {"effect": "DELAY", "severity": 5, "header": "Delays", "informed_entity": entities, "active_period": []}},
        {"id": "elev", "attributes": {"effect": "ELEVATOR_CLOSURE", "severity": 3, "header": "Elevator", "informed_entity": entities[:2], "active_period": []}},
    ]
    vehicles = [{"id": f"v{i}", "relationships": {"route": {"data": {"id": "Red"}}}} for i in range(3)]
    calls = route_mbta(monkeypatch, {"/alerts": {"data": alerts}, "/vehicles": {"data": vehicles}})
    data = client.get("/api/dashboard").get_json()
    red = data["subway_status"]["Red"]
    assert red["alert_count"] == 2                   # not 32
    assert red["service_alert_count"] == 1
    assert red["accessibility_alert_count"] == 1
    assert red["status"] == "warning"
    assert red["vehicle_count"] == 3
    assert data["subway_status"]["Blue"]["status"] == "normal"
    assert sum(1 for path, _ in calls if path == "/vehicles") == 1   # one request, not eight


def test_mbta_errors_surface_as_errors_not_empty_lists(client, monkeypatch):
    route_mbta(monkeypatch, {"/vehicles": FakeResponse({"errors": [{"code": "rate_limited"}]}, status=429)})
    res = client.get("/api/vehicles?route=Red")
    assert res.status_code == 500
    assert "unavailable" in res.get_json()["error"]


def test_alerts_endpoint_shape_and_priority(client, monkeypatch):
    alerts = [
        {"id": "future", "attributes": {"effect": "SHUTTLE", "severity": 7, "header": "Later", "active_period": [{"start": iso(600), "end": iso(900)}],
                                        "informed_entity": [{"route": "Red", "route_type": 1}]}},
        {"id": "now", "attributes": {"effect": "DELAY", "severity": 4, "header": "Now", "active_period": [{"start": iso(-10), "end": None}],
                                     "url": "https://mbta.com/x", "informed_entity": [{"route": "66", "route_type": 3}]}},
    ]
    route_mbta(monkeypatch, {"/alerts": {"data": alerts}})
    data = client.get("/api/alerts").get_json()
    assert [a["id"] for a in data["alerts"]] == ["now", "future"]
    first = data["alerts"][0]
    assert first["modes"] == ["bus"] and first["url"] == "https://mbta.com/x" and first["active_now"] is True
    # Fields the previous frontend relied on are still present.
    for key in ("id", "effect", "severity", "header", "description", "affected_routes", "updated_at"):
        assert key in first


def test_line_sorts_predictions_and_builds_station_sequence(client, monkeypatch):
    preds = [prediction("late", iso(11), iso(12), route="Red", headsign="Braintree"),
             prediction("soon", iso(6), iso(7), route="Red", headsign="Braintree")]
    included = [{"type": "stop", "id": "70036", "attributes": {"name": "Park Street"},
                 "relationships": {"parent_station": {"data": {"id": "place-pktrm"}}}}]
    included += [{"type": "trip", "id": f"t-{p['id']}", "attributes": {"headsign": "Braintree"}} for p in preds]
    route_mbta(monkeypatch, {
        "/predictions": {"data": preds, "included": included},
        "/vehicles": {"data": []},
        "/alerts": {"data": []},
        "/stops": {"data": [{"id": "place-alfcl", "attributes": {"name": "Alewife"}}, {"id": "place-pktrm", "attributes": {"name": "Park Street"}}]},
        "/routes/Red": {"data": {"attributes": {"long_name": "Red Line", "color": "DA291C", "type": 1,
                                                 "direction_destinations": ["Ashmont/Braintree", "Alewife"]}}},
    })
    data = client.get("/api/line/Red").get_json()
    stop = data["predictions_by_stop"][0]["predictions"]
    assert [p["time"] for p in stop] == sorted(p["time"] for p in stop)
    assert [s["name"] for s in data["stations"]] == ["Alewife", "Park Street"]
    assert [p["time_away"] for p in data["stations"][1]["next"]["0"]][0].endswith("min")


def test_stations_list_covers_all_stations_and_is_cacheable(client):
    res = client.get("/api/stations")
    data = res.get_json()
    assert len(data) == 126
    assert {"place-ogmnl", "place-pktrm", "place-asmnl", "place-wondl"} <= {s["id"] for s in data}
    assert "s-maxage" in res.headers["Cache-Control"]


def test_chat_keeps_per_user_history_contract(client, monkeypatch):
    monkeypatch.setattr(charlie, "process_chat_message", lambda msg: {"reply": f"echo {msg}"})
    res = client.post("/api/chat", json={"message": "hi", "history": [{"role": "user", "content": "earlier"}, {"role": "system", "content": "x"}]})
    data = res.get_json()
    assert data["reply"] == "echo hi"
    assert data["history"] == [{"role": "user", "content": "earlier"}]  # system role stripped


def test_trip_planner_falls_back_to_transitous(client, monkeypatch):
    monkeypatch.setattr(charlie, "GOOGLE_MAPS_API_KEY", None)
    itinerary = {"itineraries": [{"duration": 900, "startTime": iso(1), "endTime": iso(16), "transfers": 0,
                                  "legs": [{"mode": "SUBWAY", "routeLongName": "Red Line", "routeColor": "DA291C", "headsign": "Alewife",
                                            "from": {"name": "Park Street"}, "to": {"name": "Harvard"}, "duration": 600,
                                            "startTime": iso(3), "endTime": iso(13), "intermediateStops": [{}, {}]}]}]}
    monkeypatch.setattr(charlie.requests, "get", lambda url, **kw: FakeResponse(itinerary))
    data = client.post("/api/trip-planner", json={"origin": "Park Street", "destination": "Harvard"}).get_json()
    assert len(data["routes"]) == 1
    assert data["routes"][0]["steps"][0]["name"] == "Red Line"
    assert data["routes"][0]["steps"][0]["color"] == "#DA291C"


def test_static_modules_are_versioned_and_never_served_stale(client):
    html = client.get("/").get_data(as_text=True)
    assert '<script type="importmap">' in html
    assert f'/static/js/core/api.js?v={charlie.ASSET_VERSION}' in html
    versioned = client.get(f"/static/js/main.js?v={charlie.ASSET_VERSION}")
    assert "immutable" in versioned.headers["Cache-Control"]
    plain = client.get("/static/js/main.js")
    assert plain.headers["Cache-Control"] == "no-store" and "ETag" not in plain.headers


def test_mode_shapes_returns_typical_bus_paths_with_route_labels(client, monkeypatch):
    routes = {"data": [{"id": "66", "attributes": {"short_name": "66", "long_name": "Harvard Square - Nubian Station", "color": "FFC72C"}}]}
    patterns = {
        "data": [
            {"id": "66-9-0", "attributes": {"typicality": 1, "direction_id": 0},
             "relationships": {"route": {"data": {"id": "66"}}, "representative_trip": {"data": {"id": "t1"}}}},
            {"id": "66-5-0", "attributes": {"typicality": 3, "direction_id": 0},
             "relationships": {"route": {"data": {"id": "66"}}, "representative_trip": {"data": {"id": "t2"}}}},
        ],
        "included": [
            {"type": "trip", "id": "t1", "attributes": {"headsign": "Nubian"}, "relationships": {"shape": {"data": {"id": "s1"}}}},
            {"type": "trip", "id": "t2", "attributes": {"headsign": "Dudley"}, "relationships": {"shape": {"data": {"id": "s2"}}}},
            {"type": "shape", "id": "s1", "attributes": {"polyline": "abc"}},
            {"type": "shape", "id": "s2", "attributes": {"polyline": "def"}},
        ],
    }
    route_mbta(monkeypatch, {"/routes": routes, "/route_patterns": patterns})
    data = client.get("/api/mode-shapes?mode=bus").get_json()
    assert data["count"] == 1                       # atypical pattern skipped
    shape = data["shapes"][0]
    assert shape["route_id"] == "66" and shape["short_name"] == "66" and shape["color"] == "#FFC72C"
    assert shape["polyline"] == "abc" and shape["headsign"] == "Nubian"
    assert client.get("/api/mode-shapes?mode=ferry").status_code == 400
