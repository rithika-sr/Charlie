"""Charlie AI journeys: the reported failures, follow-ups, ambiguity, outages and
agreement with the Trip Planner. Transitous, Google, the MBTA and OpenAI are mocked."""

import time
from datetime import datetime

import pytest

import app as charlie
import assistant
from tests.test_endpoints import FakeResponse, client  # noqa: F401  (fixture re-export)
from tests.test_journey import b_line_plan

CHISWICK = charlie.STATION_DATA["Chiswick Road"]
HYNES = charlie.STATION_DATA["Hynes Convention Center"]
HARVARD = charlie.STATION_DATA["Harvard"]
NOW = charlie.BOSTON_TZ.localize(datetime(2026, 11, 16, 12, 0))      # Mon noon in Boston


@pytest.fixture
def services(monkeypatch):
    """Mocked routing/MBTA services; .plans records each router request, .set() changes behaviour."""
    state = {"plan": b_line_plan(), "geocode": [], "alerts": [], "alerts_status": 200}
    plans = []

    def fake_get(url, params=None, headers=None, timeout=None):
        if url.endswith("/v5/plan"):
            plans.append(params)
            plan = state["plan"]
            if isinstance(plan, Exception):
                raise plan
            return FakeResponse(plan)
        if url.endswith("/v1/geocode"):
            g = state["geocode"]
            return FakeResponse(g(params["text"]) if callable(g) else g)
        if url.startswith(charlie.BASE_URL + "/alerts"):
            return FakeResponse({"data": state["alerts"]}, status=state["alerts_status"])
        if "maps.googleapis.com" in url:
            raise AssertionError("Google is not configured in these tests")
        return FakeResponse({"data": []})

    monkeypatch.setattr(charlie.requests, "get", fake_get)
    monkeypatch.setattr(charlie, "GOOGLE_MAPS_API_KEY", None)
    monkeypatch.setattr(charlie, "client", None)            # no OpenAI key: deterministic paths only
    monkeypatch.setattr(charlie, "get_now", lambda: NOW)    # times like "by 5pm" don't depend on the clock

    class S:
        pass
    s = S()
    s.plans = plans
    s.set = lambda **kw: state.update(kw)
    return s


def ask(client, message, context=None, **extra):
    body = {"message": message, **extra}
    if context is not None:
        body["context"] = context
    return client.post("/api/chat", json=body).get_json()


def coords_of(station):
    return f"{station['lat']},{station['lon']}"


MISLEADING = ("check the station names", "Trip Planner tool", "couldn't find transit routes", "basic mode")


# ---------- the reported failures ----------

@pytest.mark.parametrize("message, dest", [("chiswick rd to harvard", HARVARD), ("chiswick rd to hynes", HYNES)])
def test_reported_prompts_return_journeys_from_provider_data(client, services, message, dest):
    data = ask(client, message)
    j = data["journey"]
    assert j["code"] == "ok" and j["origin"]["name"] == "Chiswick Road"
    assert services.plans[-1]["fromPlace"] == coords_of(CHISWICK) and services.plans[-1]["toPlace"] == coords_of(dest)
    best = j["routes"][0]
    assert f"leave **{best['depart_time']}**" in data["reply"] and f"arrive **{best['arrive_time']}**" in data["reply"]
    assert not any(m.lower() in data["reply"].lower() for m in MISLEADING)
    assert len(data["reply"]) < 300                                   # short; the card has the detail
    assert "$" not in data["reply"]                                    # no fare claims in the text


def test_harvard_reply_offers_harvard_ave_as_an_alternative(client, services):
    j = ask(client, "chiswick rd to harvard")["journey"]
    assert j["destination"]["name"] == "Harvard" and j["destination"]["similar"] == ["Harvard Ave"]


@pytest.mark.parametrize("message", [
    "How do I get from Chiswick Rd to Hynes?",
    "how do i get to hynes from chiswick road",
    "Chiswick Rd → Hynes Convention Ctr",
    "directions from chiswick rd. to hynes station please",
    "I'm at Chiswick Road, how can I get to Hynes?",
])
def test_ordinary_phrasings_reach_the_same_trip(client, services, message):
    j = ask(client, message)["journey"]
    assert j["code"] == "ok" and (j["origin"]["name"], j["destination"]["name"]) == ("Chiswick Road", "Hynes Convention Center")


def test_parsable_trips_never_ask_the_language_model(client, services, monkeypatch):
    monkeypatch.setattr(charlie, "get_smart_intent", lambda text: pytest.fail("LLM called for a parsable trip"))
    assert ask(client, "chiswick rd to hynes")["journey"]["code"] == "ok"


def test_llm_extracted_intent_uses_the_same_journey_service(client, services, monkeypatch):
    monkeypatch.setattr(charlie, "get_smart_intent",
                        lambda text: {"intent": "trip_planning", "origin": "Chiswick Road", "destination": "Hynes"})
    j = ask(client, "yo what's my move chiswick → convention center area")["journey"]
    assert j["code"] == "ok" and j["destination"]["name"] == "Hynes Convention Center"


# ---------- chat and planner agree ----------

@pytest.mark.parametrize("origin, dest", [("chiswick rd", "harvard"), ("Chiswick Road", "Hynes"), ("park street", "kenmore sq")])
def test_chat_and_planner_return_the_same_journey(client, services, origin, dest):
    planner = client.post("/api/trip-planner", json={"origin": origin, "destination": dest}).get_json()
    chat = ask(client, f"{origin} to {dest}")["journey"]
    assert chat["code"] == planner["code"] == "ok"
    assert chat["origin"] == planner["endpoints"]["origin"] and chat["destination"] == planner["endpoints"]["destination"]
    assert chat["routes"] == assistant.sort_routes(planner["routes"])     # same itineraries, same ranking
    assert chat["provider"] == planner["provider"] and chat["notices"] == planner["notices"]
    assert services.plans[0] == services.plans[1]                          # identical router request


@pytest.mark.parametrize("phrase, when, arrive_by, sort", [
    ("by 5pm", "2026-11-16T17:00", True, "departure"),
    ("tomorrow at 9am", "2026-11-17T09:00", False, "arrival"),
])
def test_chat_and_planner_agree_for_a_requested_time(client, services, phrase, when, arrive_by, sort):
    planner = client.post("/api/trip-planner", json={"origin": "Chiswick Road", "destination": "Harvard",
                                                     "when": when, "arrive_by": arrive_by}).get_json()
    chat = ask(client, f"chiswick rd to harvard {phrase}")["journey"]
    assert services.plans[0] == services.plans[1]                          # same time and arriveBy sent
    assert chat["routes"] == assistant.sort_routes(planner["routes"], sort)
    assert chat["requirements"]["when"]["label"] == planner["when"]["label"]


# ---------- follow-ups and per-user context ----------

def test_what_about_hynes_keeps_the_origin(client, services):
    first = ask(client, "chiswick rd to harvard")
    follow = ask(client, "what about Hynes?", first["context"])
    j = follow["journey"]
    assert (j["origin"]["name"], j["destination"]["name"]) == ("Chiswick Road", "Hynes Convention Center")
    assert services.plans[-1]["fromPlace"] == coords_of(CHISWICK)


@pytest.mark.parametrize("message, expected", [
    ("how about from Copley?", ("Copley", "Harvard")),
    ("how do I get back?", ("Harvard", "Chiswick Road")),
    ("and to Kenmore", ("Chiswick Road", "Kenmore")),
    ("refresh", ("Chiswick Road", "Harvard")),
    ("and to Copley instead", ("Chiswick Road", "Copley")),        # found in browser verification
    ("what about Kenmore then?", ("Chiswick Road", "Kenmore")),
    ("how about from Park Street instead?", ("Park Street", "Harvard")),
])
def test_follow_ups(client, services, message, expected):
    ctx = ask(client, "chiswick rd to harvard")["context"]
    j = ask(client, message, ctx)["journey"]
    assert (j["origin"]["name"], j["destination"]["name"]) == expected


def test_context_is_per_request_so_users_never_share_trips(client, services):
    ask(client, "chiswick rd to harvard")                      # user A; their context stays in their browser
    other = ask(client, "what about Hynes?")                   # user B sends no context
    assert "journey" not in other


def test_stale_context_is_ignored(client, services):
    ctx = ask(client, "chiswick rd to harvard")["context"]
    ctx["at"] = time.time() - assistant.CONTEXT_TTL_SECONDS - 60
    assert "journey" not in ask(client, "what about Hynes?", ctx)


def test_new_question_replaces_previous_context(client, services, monkeypatch):
    trip_ctx = ask(client, "chiswick rd to harvard")["context"]
    monkeypatch.setattr(charlie, "station_arrivals", lambda stop_id, now: [])
    arr = ask(client, "next train at Kenmore", trip_ctx)
    assert "arrivals" in arr and "trip" not in arr["context"]


def test_forged_context_is_sanitized():
    ctx = assistant.sanitize_context({"kind": "trip", "trip": {"origin": {"name": "X", "id": "javascript:1", "lat": "nope"}},
                                      "pending": {"field": "admin"}, "at": time.time(), "extra": "x" * 5000})
    assert ctx["trip"]["origin"] == {"name": "X"} and "pending" not in ctx and "extra" not in ctx


# ---------- clarification, unresolved, outages, empty results ----------

def test_ambiguous_station_asks_a_short_question_then_continues(client, services):
    data = ask(client, "park to harvard")
    j = data["journey"]
    assert j["code"] == "ambiguous" and {"Park Street", "Science Park"} <= set(j["candidates"]["origin"])
    assert data["reply"] == "Which “park” do you want to start from?"
    assert not services.plans                                   # nothing planned on a guess
    j2 = ask(client, "Science Park", data["context"])["journey"]
    assert (j2["origin"]["name"], j2["destination"]["name"]) == ("Science Park", "Harvard")


def test_unresolved_place_blames_only_that_place(client, services):
    services.set(geocode=[{"lat": 42.976, "lon": -71.69}])     # far outside the service area
    data = ask(client, "zzzzz to harvard")
    assert data["journey"]["code"] == "unresolved" and data["journey"]["unresolved"] == ["origin"]
    assert "“zzzzz”" in data["reply"] and "Harvard" not in data["reply"]


def test_router_outage_is_not_reported_as_bad_station_names(client, services):
    services.set(plan=charlie.requests.ConnectionError("down"))
    data = ask(client, "chiswick rd to hynes")
    assert data["journey"]["code"] == "routing_unavailable"
    assert "isn’t responding" in data["reply"] and "Nothing is wrong with the places" in data["reply"]
    assert not any(m.lower() in data["reply"].lower() for m in MISLEADING)
    assert data["context"]["trip"]["origin"]["name"] == "Chiswick Road"   # "refresh" can retry it


def test_no_itineraries_says_both_places_were_found(client, services):
    services.set(plan={"itineraries": []})
    data = ask(client, "chiswick rd to hynes")
    assert data["journey"]["code"] == "no_itineraries" and "Both places were found" in data["reply"]


def test_same_station(client, services):
    assert ask(client, "hynes to hynes convention center")["journey"]["code"] == "same_place"


def test_missing_origin_asks_then_uses_the_answer(client, services):
    data = ask(client, "take me to hynes")
    assert data["journey"]["code"] == "needs_origin" and data["reply"].startswith("Where are you starting from?")
    j = ask(client, "chiswick rd", data["context"])["journey"]
    assert (j["origin"]["name"], j["destination"]["name"]) == ("Chiswick Road", "Hynes Convention Center")


def test_pending_question_does_not_swallow_other_requests(client, services, monkeypatch):
    ctx = ask(client, "take me to hynes")["context"]
    monkeypatch.setattr(charlie, "station_arrivals", lambda stop_id, now: [])
    assert "arrivals" in ask(client, "next train at Kenmore", ctx)
    assert "journey" not in ask(client, "thanks", ctx)


def test_shared_location_is_used_as_the_origin(client, services):
    j = ask(client, "take me to hynes", location={"lat": 42.3401, "lon": -71.1507, "station": "Chiswick Road"})["journey"]
    assert j["code"] == "ok" and services.plans[-1]["fromPlace"] == "42.3401,-71.1507"
    assert any(c["kind"] == "location" for c in j["caveats"])


# ---------- times, and requirements the planner can't honor ----------

@pytest.mark.parametrize("message, kind, arrive_by", [
    ("chiswick rd to hynes by 5pm", "when", "true"),
    ("chiswick rd to hynes tomorrow morning", "when", "false"),
    ("chiswick rd to hynes next monday", "time", None),            # not a time we can read: stated, leaves now
    ("wheelchair accessible route from chiswick rd to hynes", "accessibility", None),
])
def test_times_are_applied_and_other_requirements_stated(client, services, message, kind, arrive_by):
    j = ask(client, message)["journey"]
    assert j["code"] == "ok" and j["destination"]["name"] == "Hynes Convention Center"
    assert [c["kind"] for c in j["caveats"]] == [kind]
    sent = services.plans[-1]
    assert sent.get("arriveBy") == arrive_by and ("time" in sent) == (arrive_by is not None)


def test_time_follow_up_replans_for_that_time(client, services):
    ctx = ask(client, "chiswick rd to harvard")["context"]
    assert "time" not in services.plans[-1]
    j = ask(client, "what about leaving at 6pm?", ctx)["journey"]
    assert j["code"] == "ok" and j["requirements"]["when"]["label"].endswith("at 6:00 PM")
    assert services.plans[-1]["time"][11:16] == "18:00" and services.plans[-1]["arriveBy"] == "false"
    assert j["caveats"][0]["kind"] == "when"


def test_followups_keep_the_requested_time(client, services):
    ctx = ask(client, "chiswick rd to harvard by 11:59 pm")["context"]
    j = ask(client, "what about Hynes?", ctx)["journey"]
    assert j["destination"]["name"] == "Hynes Convention Center"
    assert services.plans[-1]["time"][11:16] == "23:59" and services.plans[-1]["arriveBy"] == "true"


def test_preference_sorts_like_the_planner(client, services):
    j = ask(client, "fewest transfers from chiswick rd to harvard")["journey"]
    assert j["sort"] == "transfers" and [r["transfers"] for r in j["routes"]] == sorted(r["transfers"] for r in j["routes"])


def test_alerts_not_checked_is_said(client, services):
    services.set(alerts_status=500)
    j = ask(client, "chiswick rd to hynes")["journey"]
    assert j["alerts_checked"] is False and any(c["kind"] == "alerts" for c in j["caveats"])


# ---------- card actions ----------

def test_refresh_and_reverse_actions_use_exact_endpoints(client, services):
    j = ask(client, "chiswick rd to hynes")["journey"]
    o, d = j["request"]["origin"], j["request"]["destination"]
    again = client.post("/api/chat", json={"action": {"type": "trip", "kind": "refresh", "origin": o, "destination": d}}).get_json()
    back = client.post("/api/chat", json={"action": {"type": "trip", "kind": "reverse", "origin": d, "destination": o}}).get_json()
    assert again["journey"]["destination"]["name"] == "Hynes Convention Center"
    assert back["journey"]["origin"]["name"] == "Hynes Convention Center"


# ---------- alerts answers never claim normal service on failure ----------

def test_alerts_question_when_feed_is_down(client, services):
    services.set(alerts_status=500)
    data = ask(client, "Are there delays on the Orange Line?")
    assert data["alerts"]["status"] == "unavailable" and "couldn’t reach" in data["reply"]
    assert "good service" not in data["reply"].lower() and "no active" not in data["reply"].lower()


def test_alerts_question_with_empty_feed_is_phrased_as_what_the_feed_says(client, services):
    data = ask(client, "any delays on the orange line?")
    assert data["alerts"]["status"] == "none" and "isn’t reporting any active service alerts" in data["reply"]


def test_greeting_without_llm_is_short_help(client, services):
    reply = ask(client, "hello")["reply"]
    assert reply.startswith("Hi!") and "basic mode" not in reply


def test_requirements_survive_a_clarification(client, services):
    first = ask(client, "park to hynes by 5pm")
    assert first["journey"]["code"] == "ambiguous" and first["journey"]["requirements"]["when"]["arrive_by"] is True
    typed = ask(client, "Park Street", first["context"])["journey"]
    assert typed["code"] == "ok" and [c["kind"] for c in typed["caveats"]] == ["when"]
    assert services.plans[-1]["arriveBy"] == "true" and services.plans[-1]["time"][11:16] == "17:00"
    picked = client.post("/api/chat", json={"action": {"type": "trip", "kind": "pick", "origin": {"name": "Park Street"},
                                                       "destination": first["journey"]["request"]["destination"],
                                                       "requirements": first["journey"]["requirements"]}}).get_json()
    assert [c["kind"] for c in picked["journey"]["caveats"]] == ["when"]
    assert services.plans[-1]["arriveBy"] == "true"


def test_action_requirements_are_sanitized():
    r = assistant.sanitize_requirements({"time": "x" * 500, "accessibility": 5, "sort": "drop table"})
    assert len(r["time"]) == 60 and r["accessibility"] is None and r["sort"] is None
    assert assistant.sanitize_requirements({"when": {"time": "not a time"}})["when"] is None
    ok = assistant.sanitize_requirements({"when": {"time": "2026-11-16T18:00:00-05:00", "arrive_by": "yes", "label": "x"}})
    assert ok["when"] == {"time": "2026-11-16T18:00:00-05:00", "arrive_by": False, "label": "x"}
