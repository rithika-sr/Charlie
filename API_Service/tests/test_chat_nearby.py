"""Charlie AI "near me" departures and Trip Planner depart/arrive times (MBTA mocked, no network)."""

from datetime import timedelta

import pytest

import app as charlie
import arrivals
import transit
import trips
from tests.test_endpoints import FakeResponse, client, iso, route_mbta  # noqa: F401  (fixture re-export)

STATIONS, ALIASES = charlie.STATION_DATA, charlie.STATION_ALIASES
HERE = {"lat": 42.3398, "lon": -71.0892, "station": "Northeastern University", "lines": ["Green-E"]}


def parse(text, context=None):
    return arrivals.parse_arrival_question(text, STATIONS, ALIASES, context or {})


# ---------- parsing ----------

@pytest.mark.parametrize("text,route,toward", [
    ("Next 39 bus number towards Back Bay from my current location", "39", "back bay"),
    ("When is the next 39 bus?", "39", None),
    ("bus 57 toward Watertown from my location", "57", "watertown"),
    ("next 39 near me", "39", None),
    ("Next SL1 near me", "741", None),
    ("when is the next red line train here", "Red", None),
    ("next green line b train near me", "Green-B", None),
    ("what's leaving near me right now?", None, None),
    ("what trains are near me", None, None),
    ("what can I take from here", None, None),
    ("route 1 bus near me inbound", "1", None),
])
def test_current_location_questions_use_the_rider_not_a_station(text, route, toward):
    got = parse(text)
    assert got and got["near_me"] is True and got["station_query"] is None, got
    assert got["route"] == route and got["toward"] == toward


def test_route_numbers_in_every_phrasing():
    for text, route in [("39 bus", "39"), ("bus number 39", "39"), ("bus #39", "39"), ("route 39", "39"),
                        ("the 39 at Copley", "39"), ("next 39 toward Forest Hills", "39"), ("CT2 bus", "747"),
                        ("next 5 min", None), ("Red Line", "Red")]:
        assert arrivals.parse_route(text) == route, text


def test_station_questions_still_name_the_station():
    assert parse("When is the next Red Line train at Harvard?")["station_query"] == "harvard"
    assert parse("next 39 at Copley")["station_query"] == "copley" and parse("next 39 at Copley")["route"] == "39"
    assert parse("next train from Park Street to Alewife")["toward"] == "alewife"
    assert parse("Are there delays on the Orange Line?") is None and parse("thanks") is None
    assert parse("restaurants near me") is None


def test_departure_questions_with_toward_are_not_trips():
    is_place = lambda s: charlie.find_station_data(s) is not None
    assert trips.parse_trip_question("bus 57 toward Watertown from my location", is_place) is None
    assert trips.parse_trip_question("Next 39 bus towards Back Bay from my current location", is_place) is None
    assert trips.parse_trip_question("What bus goes from Harvard to Nubian?", is_place)["destination"] == "Nubian"


def test_near_followup_keeps_the_spot_and_changes_direction():
    ctx = {"near_point": "42.33980,-71.08920", "route": "39"}
    got = parse("what about toward Forest Hills?", ctx)
    assert got["near_point"] == ctx["near_point"] and got["route"] == "39" and got["toward"] == "forest hills"


# ---------- grouping ----------

def test_nearest_stop_wins_per_route_and_direction():
    stops = {"far": {"name": "Far St", "distance": 400}, "near": {"name": "Near St", "distance": 60}}
    preds = [
        {"route_id": "39", "direction_id": 1, "stop_id": "far", "time": iso(2), "headsign": "Back Bay"},
        {"route_id": "39", "direction_id": 1, "stop_id": "near", "time": iso(5), "headsign": "Back Bay"},
        {"route_id": "39", "direction_id": 1, "stop_id": "near", "time": iso(15), "headsign": "Back Bay"},
        {"route_id": "39", "direction_id": 0, "stop_id": "far", "time": iso(3), "headsign": "Forest Hills"},
        {"route_id": "Orange", "direction_id": 0, "stop_id": "70001", "parent_id": "near", "time": iso(4), "headsign": "Forest Hills"},
    ]
    groups = transit.nearest_stop_departures(preds, stops)
    by_key = {(g["route_id"], g["direction_id"]): g for g in groups}
    assert by_key[("39", 1)]["stop_id"] == "near" and len(by_key[("39", 1)]["items"]) == 2
    assert by_key[("39", 0)]["stop_id"] == "far"                    # only stop with that direction
    assert by_key[("Orange", 0)]["stop_id"] == "near"               # platform mapped to its station
    assert groups[-1]["distance"] == 400                            # nearest stops first
    assert transit.walk_minutes(60) == 1 and transit.walk_minutes(400) == 7


# ---------- end to end through /api/chat ----------

def stop(sid, name, lat, lon):
    return {"id": sid, "attributes": {"name": name, "latitude": lat, "longitude": lon}}


def pred(pid, stop_id, minutes, route="39", direction=1):
    return {"id": pid, "attributes": {"arrival_time": iso(minutes), "departure_time": iso(minutes), "direction_id": direction,
                                      "status": None, "schedule_relationship": None},
            "relationships": {"route": {"data": {"id": route}}, "trip": {"data": {"id": f"t{pid}"}},
                              "stop": {"data": {"id": stop_id}}}}


def mock_route_39(monkeypatch):
    stops = {"data": [stop("1357", "360 Huntington Ave", 42.3402, -71.0890),
                      stop("1307", "Huntington Ave @ Opera Pl", 42.3397, -71.0886)]}
    preds = {"data": [pred("a", "1357", 0.5), pred("b", "1357", 14.5), pred("c", "1357", 22.5), pred("d", "1307", 12, direction=0)],
             "included": [{"type": "trip", "id": "ta", "attributes": {"headsign": "Back Bay"}},
                          {"type": "trip", "id": "tb", "attributes": {"headsign": "Back Bay"}},
                          {"type": "trip", "id": "tc", "attributes": {"headsign": "Back Bay"}},
                          {"type": "trip", "id": "td", "attributes": {"headsign": "Forest Hills"}},
                          {"type": "route", "id": "39", "attributes": {"short_name": "39", "type": 3}}]}
    meta = {"data": {"attributes": {"short_name": "39", "direction_destinations": ["Forest Hills Station", "Back Bay Station"]}}}
    # Bus stops are on the street: a station's own predictions don't include the 39.
    by_stop = lambda params: {"data": []} if str(params.get("filter[stop]", "")).startswith("place-") else preds
    return route_mbta(monkeypatch, {"/stops": stops, "/predictions": by_stop, "/routes/39": meta, "/alerts": {"data": []}})


def chat(client, message, **extra):
    return client.post("/api/chat", json={"message": message, **extra}).get_json()


def test_screenshot_question_answers_from_the_nearest_route_stop(client, monkeypatch):
    calls = mock_route_39(monkeypatch)
    data = chat(client, "Next 39 bus number towards Back Bay from my current location", location=HERE)
    a = data["arrivals"]
    assert a["kind"] == "near" and a["status"] == "ok" and a["toward"] == "Back Bay"
    assert [g["stop_name"] for g in a["groups"]] == ["360 Huntington Ave"]          # Back Bay direction only
    items = a["groups"][0]["items"]
    assert [i["countdown"] for i in items] == ["Arriving", "14 min", "22 min"]
    assert items[0]["catchable"] is False and items[1]["catchable"] is True
    assert "couldn’t find an MBTA station" not in data["reply"]
    assert "**Route 39 toward Back Bay near you**" in data["reply"] and "360 Huntington Ave" in data["reply"]
    stop_call = next(p for path, p in calls if path == "/stops")
    assert stop_call["filter[route]"] == "39" and stop_call["filter[latitude]"] == 42.3398
    assert data["context"]["near_point"] == "42.33980,-71.08920"


def test_without_location_charlie_asks_for_it(client, monkeypatch):
    mock_route_39(monkeypatch)
    data = chat(client, "Next 39 bus number towards Back Bay from my current location")
    assert data["request_location"] is True and data["arrivals"]["status"] == "need_location"
    assert "Share my location" in data["reply"] and "next 39 bus at Copley" in data["reply"]


def test_bus_at_a_station_uses_stops_outside_it(client, monkeypatch):
    calls = mock_route_39(monkeypatch)
    data = chat(client, "next 39 bus at Copley")
    assert data["arrivals"]["kind"] == "near" and data["arrivals"]["place"] == "Copley"
    copley = charlie.STATION_DATA["Copley"]
    assert any(path == "/stops" and p.get("filter[latitude]") == round(copley["lat"], 5) for path, p in calls)


def test_whats_leaving_near_me_covers_every_route(client, monkeypatch):
    calls = mock_route_39(monkeypatch)
    data = chat(client, "what's leaving near me right now?", location=HERE)
    a = data["arrivals"]
    assert a["status"] == "ok" and a["route"] is None and len(a["groups"]) == 2      # both directions of the 39
    stop_call = next(p for path, p in calls if path == "/stops")
    assert "filter[route]" not in stop_call and stop_call["filter[location_type]"] == "0"
    assert data["reply"].startswith("📍 **Departures near you**")


def test_no_stops_nearby_says_so(client, monkeypatch):
    route_mbta(monkeypatch, {"/stops": {"data": []}})
    data = chat(client, "next 39 near me", location={"lat": 42.6, "lon": -71.6})
    assert data["arrivals"]["status"] == "no_stops" and "couldn’t find a Route 39 stop within 0.7 mi" in data["reply"]


# ---------- Trip Planner depart / arrive times ----------

def plan(client, monkeypatch, **body):
    seen = []

    def fake_get(url, params=None, headers=None, timeout=None):
        if url.endswith("/v5/plan"):
            seen.append(params)
        return FakeResponse({"itineraries": []} if url.endswith("/v5/plan") else {"data": []})
    monkeypatch.setattr(charlie, "GOOGLE_MAPS_API_KEY", None)
    monkeypatch.setattr(charlie.requests, "get", fake_get)
    res = client.post("/api/trip-planner", json={"origin": "Park Street", "destination": "Harvard", **body})
    return res, seen


def test_planner_sends_a_depart_time_in_boston(client, monkeypatch):
    later = (charlie.get_now() + timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M")
    res, seen = plan(client, monkeypatch, when=later)
    assert res.status_code == 200 and seen[0]["arriveBy"] == "false"
    assert seen[0]["time"].startswith(later) and seen[0]["time"][-6:] in ("-05:00", "-04:00")
    assert res.get_json()["when"]["label"].startswith("leaving ")


def test_planner_arrive_by(client, monkeypatch):
    later = (charlie.get_now() + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M")
    res, seen = plan(client, monkeypatch, when=later, arrive_by=True)
    assert seen[0]["arriveBy"] == "true" and res.get_json()["when"]["label"].startswith("arriving by ")


def test_planner_rejects_past_and_bad_times(client, monkeypatch):
    past = (charlie.get_now() - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M")
    res, seen = plan(client, monkeypatch, when=past)
    assert res.status_code == 400 and res.get_json()["code"] == "past_time" and not seen
    res, _ = plan(client, monkeypatch, when="not a time")
    assert res.status_code == 400 and res.get_json()["code"] == "bad_time"


def test_planner_without_time_is_leave_now(client, monkeypatch):
    res, seen = plan(client, monkeypatch)
    assert res.status_code == 200 and "when" not in res.get_json() and "time" not in seen[0]
