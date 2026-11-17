"""Charlie AI arrivals and trip-planner regressions, with the MBTA and Transitous mocked."""

from datetime import timedelta

import pytest

import app as charlie
import arrivals
import transit
from tests.test_endpoints import FakeResponse, client, iso, route_mbta  # noqa: F401  (fixture re-export)


# ---------- fixtures: one Harvard prediction snapshot shared by board and assistant ----------

def harvard_snapshot():
    def pred(pid, minutes, headsign, direction, route="Red"):
        return {
            "id": pid,
            "attributes": {"arrival_time": iso(minutes + 0.5), "departure_time": iso(minutes + 1.5), "direction_id": direction,
                           "status": None, "schedule_relationship": None},
            "relationships": {"route": {"data": {"id": route}}, "trip": {"data": {"id": f"t-{pid}"}},
                              "stop": {"data": {"id": f"s-{direction}"}}},
        }
    preds = [pred("a1", 2, "Alewife", 1), pred("b1", 3, "Braintree", 0), pred("s1", 6, "Ashmont", 0),
             pred("a2", 9, "Alewife", 1), pred("b2", 12, "Braintree", 0), pred("bus", 4, "Watertown Square", 0, route="71")]
    included = [{"type": "trip", "id": f"t-{p['id']}", "attributes": {"headsign": h}}
                for p, h in zip(preds, ["Alewife", "Braintree", "Ashmont", "Alewife", "Braintree", "Watertown Square"])]
    included += [{"type": "route", "id": "Red", "attributes": {"long_name": "Red Line", "type": 1, "color": "DA291C"}},
                 {"type": "route", "id": "71", "attributes": {"short_name": "71", "long_name": "Watertown Square - Harvard", "type": 3}},
                 {"type": "stop", "id": "s-0", "attributes": {"platform_name": "Ashmont/Braintree"}},
                 {"type": "stop", "id": "s-1", "attributes": {"platform_name": "Alewife"}}]
    return {"data": preds, "included": included}


RED_ROUTE = {"data": {"id": "Red", "attributes": {"direction_destinations": ["Ashmont/Braintree", "Alewife"],
                                                 "direction_names": ["South", "North"]}}}


def mock_harvard(monkeypatch, predictions=None):
    return route_mbta(monkeypatch, {
        "/predictions": predictions if predictions is not None else harvard_snapshot(),
        "/routes/Red": RED_ROUTE,
        "/stops/place-harsq": {"data": {"attributes": {"name": "Harvard"}}},
        "/alerts": {"data": []},
    })


def chat(client, message, **extra):
    return client.post("/api/chat", json={"message": message, **extra}).get_json()


# ---------- Charlie AI arrivals ----------

def test_harvard_prompt_answers_both_directions_with_branch_headsigns(client, monkeypatch):
    mock_harvard(monkeypatch)
    data = chat(client, "When is the next Red Line train at Harvard?")
    a = data["arrivals"]
    assert a["status"] == "ok" and a["station"] == {"id": "place-harsq", "name": "Harvard"}
    labels = [g["label"] for g in a["groups"]]
    assert labels == ["Alewife", "Ashmont/Braintree"]          # sorted by soonest; route's own destinations
    south = a["groups"][1]["items"]
    assert [i["headsign"] for i in south] == ["Braintree", "Ashmont", "Braintree"]
    assert all(g["route_id"] == "Red" for g in a["groups"])    # bus 71 filtered out by "Red Line"
    assert "Toward Ashmont/Braintree" in data["reply"] and "Braintree — **3 min**" in data["reply"]
    assert a["updated_at"] and data["context"]["station_id"] == "place-harsq"


def test_assistant_and_station_board_agree_on_the_same_snapshot(client, monkeypatch):
    mock_harvard(monkeypatch)
    board = client.get("/api/station/place-harsq").get_json()
    answer = chat(client, "When is the next Red Line train at Harvard?")["arrivals"]
    board_red = sorted((p["headsign"], p["time"]) for p in board["predictions"] if p["route_id"] == "Red")
    chat_red = sorted((i["headsign"], i["time"]) for g in answer["groups"] for i in g["items"])
    assert set(chat_red) <= set(board_red)
    assert len(chat_red) == len(board_red)                       # 3 per direction covers all 5 here


def test_direction_specific_followup_uses_context(client, monkeypatch):
    mock_harvard(monkeypatch)
    first = chat(client, "When is the next Red Line train at Harvard?")
    follow = chat(client, "What about toward Ashmont?", history=first["history"], context=first["context"])
    items = [i for g in follow["arrivals"]["groups"] for i in g["items"]]
    assert follow["arrivals"]["toward"] == "Ashmont"
    assert [i["headsign"] for i in items] == ["Ashmont"]
    # A station along the line resolves to its direction even without a matching headsign.
    jfk = chat(client, "next train to JFK/UMass at Harvard")["arrivals"]
    assert {i["direction_id"] for g in jfk["groups"] for i in g["items"]} == {0}


def test_no_predictions_says_so_with_a_next_step(client, monkeypatch):
    mock_harvard(monkeypatch, predictions={"data": [], "included": []})
    data = chat(client, "When is the next Red Line train at Harvard?")
    assert data["arrivals"]["status"] == "none"
    assert "isn’t reporting any upcoming Red Line departures at **Harvard**" in data["reply"]
    assert "Service Alerts" in data["reply"]


def test_upstream_error_is_unavailable_not_empty(client, monkeypatch):
    mock_harvard(monkeypatch, predictions=FakeResponse({"errors": []}, status=503))
    data = chat(client, "When is the next Red Line train at Harvard?")
    assert data["arrivals"]["status"] == "unavailable"
    assert "couldn’t reach live MBTA predictions" in data["reply"]
    assert "**" not in data["reply"].split("**Harvard**", 1)[1]   # no invented times


def test_ambiguous_and_unknown_stations(client, monkeypatch):
    mock_harvard(monkeypatch)
    amb = chat(client, "next train at park")["arrivals"]
    assert amb["status"] == "ambiguous" and "Park Street" in amb["candidates"] and len(amb["candidates"]) > 1
    unk = chat(client, "when is the next train at Hogwarts")
    assert unk["arrivals"]["status"] == "unknown" and "couldn’t find an MBTA station" in unk["reply"]


def test_parser_leaves_non_arrival_questions_to_the_assistant():
    S, A = charlie.STATION_DATA, charlie.STATION_ALIASES
    assert arrivals.parse_arrival_question("Are there delays on the Orange Line?", S, A) is None
    assert arrivals.parse_arrival_question("How do I get from Fenway to Park Street?", S, A) is None
    p = arrivals.parse_arrival_question("southbound red line at park street", S, A)
    assert p["station_query"] == "park street" and p["direction"] == ("ns", 0) and p["route"] == "Red"


# ---------- trip planner: place resolution and errors ----------

def transitous(monkeypatch, plan=None, geocode=None, fail=None):
    def fake_get(url, params=None, headers=None, timeout=None):
        if fail:
            raise charlie.requests.ConnectionError("down")
        if url.endswith("/v1/geocode"):
            return FakeResponse(geocode(params["text"]) if callable(geocode) else (geocode or []))
        if url.endswith("/v5/plan"):
            return FakeResponse(plan or {"itineraries": []})
        if url.startswith(charlie.BASE_URL):
            return FakeResponse({"data": []})
        raise AssertionError(url)
    monkeypatch.setattr(charlie, "GOOGLE_MAPS_API_KEY", None)
    monkeypatch.setattr(charlie.requests, "get", fake_get)


def test_unresolved_origin_is_reported_per_field(client, monkeypatch):
    # The geocoder only knows far-away places for nonsense (it used to pick "New Boston, NH").
    transitous(monkeypatch, geocode=lambda text: [{"lat": 42.976, "lon": -71.69, "name": "New Boston"}])
    data = client.post("/api/trip-planner", json={"origin": "zzzzz", "destination": "Harvard"}).get_json()
    assert data["code"] == "unresolved" and data["unresolved"] == ["origin"]
    both = client.post("/api/trip-planner", json={"origin": "zzzzz", "destination": "qqqq"}).get_json()
    assert both["unresolved"] == ["origin", "destination"]


def test_address_resolves_inside_the_service_area(client, monkeypatch):
    seen = []
    def geocode(text):
        seen.append(text)
        return [{"lat": 42.339, "lon": -71.087, "name": "360 Huntington Avenue"}]
    transitous(monkeypatch, geocode=geocode, plan={"itineraries": []})
    data = client.post("/api/trip-planner", json={"origin": "360 Huntington Ave, Boston", "destination": "Harvard"}).get_json()
    assert data["code"] == "no_itineraries"                      # resolved, but the router had nothing
    assert seen == ["360 Huntington Ave, Boston"]


def test_routing_failure_is_distinct(client, monkeypatch):
    transitous(monkeypatch, fail=True)
    res = client.post("/api/trip-planner", json={"origin": "Park Street", "destination": "Harvard"})
    assert res.status_code == 502 and res.get_json()["code"] == "routing_unavailable"


def test_blank_input_is_still_rejected(client):
    assert client.post("/api/trip-planner", json={"origin": "", "destination": "Harvard"}).status_code == 400


# ---------- trip planner: closures where they affect the trip ----------

def leg(mode, route, frm, to, start, end, direction=0, stops=()):
    return {"mode": mode, "routeId": f"us-ma-MBTA_{route}", "routeLongName": f"{route} Line", "routeColor": "ED8B00",
            "directionId": str(direction), "headsign": "Forest Hills", "duration": (end - start) * 60,
            "from": {"name": frm[0], "stopId": f"us-ma-MBTA_{frm[1]}"}, "to": {"name": to[0], "stopId": f"us-ma-MBTA_{to[1]}"},
            "startTime": iso(start), "endTime": iso(end),
            "intermediateStops": [{"name": n, "stopId": f"us-ma-MBTA_{i}"} for n, i in stops]}


def walk(frm, to, minutes, distance=300):
    return {"mode": "WALK", "duration": minutes * 60, "distance": distance, "from": {"name": frm}, "to": {"name": to}}


def itin(start, end, legs, transfers=0):
    return {"duration": (end - start) * 60, "startTime": iso(start), "endTime": iso(end), "transfers": transfers, "legs": legs}


SYMPHONY_CLOSED = {"id": "1026060", "attributes": {
    "effect": "STATION_CLOSURE", "severity": 7, "header": "Green Line E Branch: Symphony station is closed.",
    "active_period": [{"start": "2026-06-06T03:00:00-04:00", "end": None}],
    "informed_entity": [{"route": "Green-E", "stop": "70241"}, {"route": "Green-E", "stop": "70242"},
                        {"route": "Green-E", "stop": "place-symcl"}]}}


def plan_with_alerts(monkeypatch, plan, alerts):
    def fake_get(url, params=None, headers=None, timeout=None):
        if url.endswith("/v5/plan"):
            return FakeResponse(plan)
        if url.startswith(charlie.BASE_URL + "/alerts"):
            return FakeResponse({"data": alerts})
        return FakeResponse([])
    monkeypatch.setattr(charlie, "GOOGLE_MAPS_API_KEY", None)
    monkeypatch.setattr(charlie.requests, "get", fake_get)


def test_destination_closure_notice_and_named_final_walk(client, monkeypatch):
    plan = {"itineraries": [
        itin(1, 17, [walk("START", "Downtown Crossing", 2),
                     leg("SUBWAY", "Orange", ("Downtown Crossing", "70020"), ("Massachusetts Avenue", "70012"), 3, 11),
                     walk("Massachusetts Avenue", "END", 6)]),
        itin(2, 20, [leg("SUBWAY", "Green-E", ("Park Street", "70200"), ("Prudential", "70240"), 2, 14,
                         stops=[("Copley", "70154")]), walk("Prudential", "END", 6)]),
    ]}
    plan_with_alerts(monkeypatch, plan, [SYMPHONY_CLOSED])
    data = client.post("/api/trip-planner", json={"origin": "Park Street", "destination": "Symphony", "dest_id": "place-symcl"}).get_json()
    [notice] = data["notices"]
    assert notice["role"] == "destination" and notice["station"] == "Symphony" and notice["effect"] == "STATION_CLOSURE"
    assert notice["explanation"] == "These options reach the area via Massachusetts Avenue or Prudential and a short walk."
    last = data["routes"][0]["detailed_steps"][-1]
    assert last["instruction"].endswith("to Symphony") and last["details"] == "From Massachusetts Avenue"
    assert all(r["alerts"] == [] for r in data["routes"])        # neither itinerary uses the closed stop


GREEN_E_DELAY = {"id": "delay", "attributes": {
    "effect": "DELAY", "severity": 5, "header": "Green Line E delays of about 10 minutes.",
    "active_period": [], "informed_entity": [{"route": "Green-E"}]}}


def test_passing_a_closed_station_is_not_flagged_but_route_wide_delays_are(client, monkeypatch):
    # 360 Huntington Ave → Harvard: Green Line E boards at Northeastern, passes Symphony,
    # gets off at Park Street. The rider never uses Symphony.
    plan = {"itineraries": [itin(0, 30, [
        walk("START", "Northeastern University", 1, distance=60),
        leg("SUBWAY", "Green-E", ("Northeastern University", "70244"), ("Park Street", "70200"), 1, 14, direction=1,
            stops=[("Symphony", "70242"), ("Prudential", "70240"), ("Copley", "70154")]),
        leg("SUBWAY", "Red", ("Park Street", "70076"), ("Harvard", "70068"), 16, 27, direction=1, stops=[("Charles/MGH", "70074")]),
        walk("Harvard", "END", 3)], transfers=1)]}
    plan_with_alerts(monkeypatch, plan, [SYMPHONY_CLOSED, GREEN_E_DELAY])
    data = client.post("/api/trip-planner", json={"origin": "360 Huntington Ave, Boston", "destination": "Harvard",
                                                   "origin_coords": "42.339,-71.087"}).get_json()
    alerts = data["routes"][0]["alerts"]
    assert [(a["id"], a["scope"]) for a in alerts] == [("delay", "route")]
    assert data["notices"] == []


def test_closure_at_a_stop_the_rider_uses_is_flagged(client, monkeypatch):
    plan = {"itineraries": [itin(0, 20, [leg("SUBWAY", "Green-E", ("Park Street", "70200"), ("Symphony", "70242"), 1, 14,
                                             direction=0, stops=[("Copley", "70155")])])]}
    plan_with_alerts(monkeypatch, plan, [SYMPHONY_CLOSED])
    data = client.post("/api/trip-planner", json={"origin": "Park Street", "destination": "Copley"}).get_json()
    [a] = data["routes"][0]["alerts"]
    assert a["scope"] == "stop" and a["stop_names"] == ["Symphony"]


def test_suspension_through_a_passed_stop_is_described_as_affecting_the_ride():
    suspended = {"id": "s", "attributes": {"effect": "SUSPENSION", "severity": 7, "header": "No E service at Symphony.",
                                           "active_period": [], "informed_entity": [{"route": "Green-E", "stop": "70242"}]}}
    now = charlie.get_now()
    leg_through = {"route_id": "Green-E", "direction_id": 1, "stop_ids": {"70244", "70200"}, "passed_stop_ids": {"70242", "70240"},
                   "start": now, "end": now + timedelta(minutes=12)}
    assert [(m["id"], m["scope"], m["stops"]) for m in transit.match_leg_alerts([suspended, SYMPHONY_CLOSED], leg_through)] == [("s", "segment", ["70242"])]
    elsewhere = {**leg_through, "passed_stop_ids": {"70154"}}
    assert transit.match_leg_alerts([suspended, SYMPHONY_CLOSED], elsewhere) == []


def test_direction_specific_bypass_and_route_wide_alerts():
    bypass = {"id": "b", "attributes": {"effect": "STOP_CLOSURE", "severity": 5, "header": "Northbound trains bypass State.",
                                        "active_period": [], "informed_entity": [{"route": "Orange", "stop": "70022", "direction_id": 1}]}}
    wide = {"id": "w", "attributes": {"effect": "DELAY", "severity": 5, "header": "Orange Line delays.",
                                      "active_period": [], "informed_entity": [{"route": "Orange"}]}}
    expired = {"id": "x", "attributes": {"effect": "SUSPENSION", "severity": 8, "header": "Old.",
                                         "active_period": [{"start": "2020-01-01T00:00:00Z", "end": "2020-01-02T00:00:00Z"}],
                                         "informed_entity": [{"route": "Orange"}]}}
    now = charlie.get_now()
    southbound = {"route_id": "Orange", "direction_id": 0, "stop_ids": {"70022", "70020"}, "passed_stop_ids": set(),
                  "start": now, "end": now + timedelta(minutes=9)}
    northbound = {**southbound, "direction_id": 1}
    unaffected = {**southbound, "route_id": "Red"}
    assert [m["id"] for m in transit.match_leg_alerts([bypass, wide, expired], southbound)] == ["w"]
    north = transit.match_leg_alerts([bypass, wide, expired], northbound)
    assert [(m["id"], m["scope"]) for m in north] == [("b", "stop"), ("w", "route")]
    assert transit.match_leg_alerts([bypass, wide, expired], unaffected) == []


# ---------- Charlie AI trip questions use the Trip Planner's routing ----------

def test_chat_plans_chiswick_rd_to_hynes_with_the_planner(client, monkeypatch):
    plan = {"itineraries": [
        itin(2, 40, [walk("START", "Chiswick Road", 2, distance=150),
                     leg("SUBWAY", "Green-B", ("Chiswick Road", "70122"), ("Hynes Convention Center", "70152"), 4, 38, direction=1)]),
        itin(1, 33, [walk("START", "Reservoir", 14, distance=950),
                     leg("SUBWAY", "Green-D", ("Reservoir", "70174"), ("Hynes Convention Center", "70152"), 15, 33, direction=1)]),
    ]}
    plan_with_alerts(monkeypatch, plan, [])
    data = client.post("/api/chat", json={"message": "How do I get from Chiswick Rd to Hynes Convention Center?"}).get_json()
    trip = data["trip"]
    assert trip["code"] == "ok" and trip["origin_name"] == "Chiswick Road" and trip["destination_name"] == "Hynes Convention Center"
    assert [r["summary"] for r in trip["routes"]] == ["Green-D Line", "Green-B Line"]     # earliest arrival first
    reply = data["reply"]
    assert reply.startswith("🗺️ **Chiswick Road → Hynes Convention Center** — 2 options")
    assert "**1. Green-D Line**" in reply and "**2. Green-B Line**" in reply      # every option is listed
    assert "Walk 0.6 mi to Reservoir" in reply                                  # first option's steps
    assert "couldn" not in reply


def test_chat_trip_errors_are_specific(client, monkeypatch):
    transitous(monkeypatch, geocode=[])                     # nothing resolves in the Boston area
    bad = client.post("/api/chat", json={"message": "How do I get from zzzzz to Harvard?"}).get_json()
    assert bad["trip"]["code"] == "unresolved" and "“zzzzz”" in bad["reply"]
    need = client.post("/api/chat", json={"message": "Take me to Harvard"}).get_json()
    assert need["trip"]["code"] == "need_origin" and need.get("request_location")
    near = client.post("/api/chat", json={"message": "Take me to Harvard",
                                          "location": {"station": "Park Street", "lat": 42.356, "lon": -71.062}}).get_json()
    assert near["trip"]["origin_name"] == "Park Street"
    transitous(monkeypatch, fail=True)
    down = client.post("/api/chat", json={"message": "from Park Street to Harvard"}).get_json()
    assert down["trip"]["code"] == "routing_unavailable" and "didn’t respond" in down["reply"]


def test_trip_parser_and_station_names():
    import trips
    is_place = lambda s: charlie.find_station_data(s) is not None
    cases = {
        "How do I get from Chiswick Rd to Hynes Convention Center?": ("Chiswick Rd", "Hynes Convention Center"),
        "Directions from South Station to Fenway": ("South Station", "Fenway"),
        "I'm at Ruggles, how can I reach Harvard?": ("Ruggles", "Harvard"),
        "How do I get to Harvard from 360 Huntington Ave, Boston?": ("360 Huntington Ave, Boston", "Harvard"),
        "Chiswick Rd to Hynes": ("Chiswick Rd", "Hynes"),
        "Take me to the Airport": (None, "Airport"),
    }
    for text, (o, d) in cases.items():
        assert trips.parse_trip_question(text, is_place) == {"origin": o, "destination": d, "when": None}, text
    assert trips.parse_trip_question("next train from Park Street to Alewife", is_place) is None
    assert trips.parse_trip_question("Are there delays on the Orange Line?", is_place) is None
    assert charlie.find_station_data("Chiswick Rd")["name"] == "Chiswick Road"
    assert charlie.find_station_data("hynes")["name"] == "Hynes Convention Center"
    assert charlie.find_station_data("360 Huntington Ave, Boston") is None
