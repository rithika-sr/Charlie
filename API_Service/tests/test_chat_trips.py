"""Charlie AI trip questions: phrasing, times, follow-ups and planner results (no network)."""

import re
from datetime import datetime

import pytest

import app as charlie
import trips
from tests.test_endpoints import FakeResponse, client, iso  # noqa: F401  (fixture re-export)

NOW = charlie.BOSTON_TZ.localize(datetime(2026, 11, 16, 17, 30))        # Mon 5:30 PM in Boston
IS_PLACE = lambda s: charlie.find_station_data(s) is not None or s.lower() in charlie.LANDMARKS

PHRASINGS = {
    "How do I get from Chiswick Rd to Hynes Convention Center?": ("Chiswick Rd", "Hynes Convention Center"),
    "from harvard to fenway": ("harvard", "fenway"),
    "Park Street to Logan Airport": ("Park Street", "Logan Airport"),
    "directions from South Station to Fenway Park": ("South Station", "Fenway Park"),
    "What's the best way to get to MIT from Back Bay?": ("Back Bay", "MIT"),
    "I'm at Ruggles. How can I reach Harvard Square?": ("Ruggles", "Harvard Square"),
    "I need to go from Downtown Crossing to the TD Garden": ("Downtown Crossing", "TD Garden"),
    "Can you plan a trip from Northeastern University to Boston Common?": ("Northeastern University", "Boston Common"),
    "how to go alewife to braintree": ("alewife", "braintree"),
    "route copley → wonderland": ("copley", "wonderland"),
    "Davis -> Quincy Center": ("Davis", "Quincy Center"),
    "get me from Jamaica Plain to Somerville": ("Jamaica Plain", "Somerville"),
    "How can I travel from 360 Huntington Ave, Boston to 1 Ashburton Pl, Boston?": ("360 Huntington Ave, Boston", "1 Ashburton Pl, Boston"),
    "how do i get from BU to Faneuil Hall": ("BU", "Faneuil Hall"),
    "how do I get to Worcester from South Station": ("South Station", "Worcester"),
    "from Kendall to the Museum of Fine Arts please": ("Kendall", "Museum of Fine Arts"),
    "Which train should I take from Back Bay to Riverside?": ("Back Bay", "Riverside"),
    "What bus goes from Harvard to Nubian?": ("Harvard", "Nubian"),
    "can I take the T from Brookline to Boston Children's Hospital": ("Brookline", "Boston Children's Hospital"),
    "I want to go to Hynes from Chiswick Road": ("Chiswick Road", "Hynes"),
    "Going from Lechmere to Union Square, what's the fastest way?": ("Lechmere", "Union Square"),
    "best route between Harvard and MIT": ("Harvard", "MIT"),
    "Chiswick Rd to Hynes": ("Chiswick Rd", "Hynes"),
    "Take me to the Airport": (None, "Airport"),
    "When should I leave Harvard to get to Fenway by 6?": ("Harvard", "Fenway"),
    "I'm in Cambridge near Central and I need to be at Fenway Park for a game, what should I take?": ("Central", "Fenway Park"),
    "yo whats the move to get from somerville to the seaport": ("somerville", "seaport"),
}


@pytest.mark.parametrize("text,expected", PHRASINGS.items())
def test_trip_phrasings_extract_origin_and_destination(text, expected):
    got = trips.parse_trip_question(text, IS_PLACE, now=NOW)
    assert got is not None, text
    assert (got["origin"], got["destination"]) == expected


@pytest.mark.parametrize("text", [
    "When is the next Red Line train at Harvard?", "next train from Park Street to Alewife",
    "Are there delays on the Orange Line?", "Show me service alerts", "hi", "thanks!",
])
def test_non_trip_questions_are_left_alone(text):
    assert trips.parse_trip_question(text, IS_PLACE, now=NOW) is None


@pytest.mark.parametrize("text,label,arrive_by,hhmm,day", [
    ("from park st to harvard tomorrow at 9am", "leaving tomorrow at 9:00 AM", False, (9, 0), 17),
    ("how do I get from Forest Hills to Oak Grove by 6 pm", "arriving by 6:00 PM today", True, (18, 0), 16),
    ("from park st to harvard at 8 tonight", "leaving today at 8:00 PM", False, (20, 0), 16),
    ("Chiswick Rd to Hynes leaving in 30 minutes", "leaving today at 6:00 PM", False, (18, 0), 16),
    ("from Davis to Copley at 7:15am", "leaving tomorrow at 7:15 AM", False, (7, 15), 17),   # 7:15 AM already passed today
    ("from Davis to Copley arriving at noon tomorrow", "arriving by 12:00 PM tomorrow", True, (12, 0), 17),
])
def test_times_are_parsed_and_removed_from_place_names(text, label, arrive_by, hhmm, day):
    got = trips.parse_trip_question(text, IS_PLACE, now=NOW)
    w = got["when"]
    assert w["label"] == label and w["arrive_by"] is arrive_by
    assert (w["time"].hour, w["time"].minute, w["time"].day) == (*hhmm, day)
    assert not any(x in got["destination"].lower() for x in ("tomorrow", "pm", "am", "minutes", "tonight", "leaving", "noon"))


def test_followups_fill_in_or_change_the_trip():
    ctx = {"trip_destination": "Fenway"}
    assert trips.parse_trip_followup("Park Street", ctx, IS_PLACE)["origin"] == "Park Street"
    assert trips.parse_trip_followup("from Ruggles", ctx, IS_PLACE)["origin"] == "Ruggles"
    ctx = {"trip_origin": "Harvard", "trip_destination": "Kendall"}
    assert trips.parse_trip_followup("what about to Copley?", ctx, IS_PLACE) == {"origin": "Harvard", "destination": "Copley", "when": None}
    assert trips.parse_trip_followup("from Davis instead", ctx, IS_PLACE) == {"origin": "Davis", "destination": "Kendall", "when": None}
    assert trips.parse_trip_followup("When is the next train?", ctx, IS_PLACE) is None
    assert trips.parse_trip_followup("Park Street", {}, IS_PLACE) is None


def test_landmarks_and_abbreviations_resolve():
    assert charlie.landmark("BU")["name"] == "Boston University"
    assert charlie.landmark("Fenway Park")["coords"] == "42.3467,-71.0972"
    assert charlie.find_station_data("Chiswick Rd")["name"] == "Chiswick Road"
    assert charlie.find_station_data("Logan Airport")["name"] == "Airport"


# ---------- end to end through /api/chat with the router mocked ----------

def leg(route, frm, to, start, end):
    return {"mode": "SUBWAY", "routeId": f"us-ma-MBTA_{route}", "routeLongName": f"{route} Line", "routeColor": "ED8B00",
            "directionId": "0", "headsign": "Somewhere", "duration": (end - start) * 60,
            "from": {"name": frm, "stopId": "us-ma-MBTA_1"}, "to": {"name": to, "stopId": "us-ma-MBTA_2"},
            "startTime": iso(start), "endTime": iso(end), "intermediateStops": []}


def itin(start, end, route="Orange"):
    return {"duration": (end - start) * 60, "startTime": iso(start), "endTime": iso(end), "transfers": 0,
            "legs": [leg(route, "A", "B", start, end)]}


def mock_router(monkeypatch, plans, seen=None):
    """plans: list of Transitous responses returned in order for successive /v5/plan calls."""
    queue = list(plans)

    def fake_get(url, params=None, headers=None, timeout=None):
        if url.endswith("/v5/plan"):
            if seen is not None:
                seen.append(params)
            return FakeResponse(queue.pop(0) if queue else {"itineraries": []})
        if url.endswith("/v1/geocode"):
            return FakeResponse([{"lat": 42.35, "lon": -71.06, "name": params["text"]}])
        return FakeResponse({"data": []})
    monkeypatch.setattr(charlie, "GOOGLE_MAPS_API_KEY", None)
    monkeypatch.setattr(charlie.requests, "get", fake_get)


def chat(client, message, **extra):
    return client.post("/api/chat", json={"message": message, **extra}).get_json()


def test_departed_options_are_never_shown(client, monkeypatch):
    mock_router(monkeypatch, [{"itineraries": [itin(-20, -5), itin(-3, 15), itin(4, 20)]}])
    data = chat(client, "from Park Street to Harvard")
    assert len(data["trip"]["routes"]) == 1                       # only the one leaving in 4 min
    assert "2 options" not in data["reply"] and "1 option" in data["reply"]


def test_requested_time_is_sent_to_the_router(client, monkeypatch):
    seen = []
    mock_router(monkeypatch, [{"itineraries": [itin(60, 76)]}], seen)
    data = chat(client, "from Park Street to Harvard in 1 hour")
    assert seen[0]["arriveBy"] == "false" and "T" in seen[0]["time"] and seen[0]["time"].endswith(":00-05:00")
    assert re.search(r"leaving (today|tomorrow) at", data["reply"])          # "tomorrow" after 11 PM


def test_arrive_by_lists_latest_departure_first(client, monkeypatch):
    mock_router(monkeypatch, [{"itineraries": [itin(5, 40), itin(15, 50), itin(10, 45)]}])
    data = chat(client, "how do I get from Forest Hills to Oak Grove by 11:59 pm")
    departs = [r["depart_ts"] for r in data["trip"]["routes"]]
    assert departs == sorted(departs, reverse=True)
    assert "latest departure first" in data["reply"]


def test_missed_deadline_says_so_and_shows_soonest_arrivals(client, monkeypatch):
    mock_router(monkeypatch, [{"itineraries": [itin(-30, -2)]}, {"itineraries": [itin(2, 30), itin(6, 34)]}])
    data = chat(client, "how do I get from Forest Hills to Oak Grove by 11:59 pm")
    assert data["trip"]["missed_deadline"] is True and len(data["trip"]["routes"]) == 2
    assert data["reply"].startswith("⏰ No trip leaving now gets you from **Forest Hills** to **Oak Grove**")


def test_conversation_followups_and_pronouns(client, monkeypatch):
    mock_router(monkeypatch, [{"itineraries": [itin(3, 20)]}] * 4)
    first = chat(client, "Take me to Fenway")
    assert first["trip"]["code"] == "need_origin" and first["context"] == {"trip_destination": "Fenway"}
    second = chat(client, "Park Street", history=first["history"], context=first["context"])
    assert second["trip"]["code"] == "ok"
    assert (second["trip"]["origin_name"], second["trip"]["destination_name"]) == ("Park Street", "Fenway")
    third = chat(client, "what about to Copley?", history=second["history"], context=second["context"])
    assert (third["trip"]["origin_name"], third["trip"]["destination_name"]) == ("Park Street", "Copley")
    here = chat(client, "from here to Kendall", location={"station": "Harvard", "lat": 42.37, "lon": -71.12})
    assert here["trip"]["origin_name"] == "Harvard"
    there = chat(client, "how do I get from there to Fenway", context={"station_name": "Ruggles"})
    assert there["trip"]["origin_name"] == "Ruggles"


def test_every_option_is_listed_with_times_and_first_option_steps(client, monkeypatch):
    mock_router(monkeypatch, [{"itineraries": [itin(2, 30), itin(5, 25, "Red"), itin(9, 40), itin(12, 33, "Blue")]}])
    reply = chat(client, "How do I get from Chiswick Rd to Hynes Convention Center?")["reply"]
    for n in range(1, 5):
        assert f"**{n}. " in reply
    assert reply.count("leave **") == 4 and reply.count("arrive **") == 4
    assert "• Take Red Line" in reply                             # earliest arrival (Red, +25) shows its steps
