"""The shared journey service: provider fallbacks, place resolution and outcomes.

Every routing/geocoding/MBTA call is mocked (no network, no credentials)."""

import pytest

import app as charlie
from tests.test_endpoints import FakeResponse, client  # noqa: F401  (fixture re-export)
from tests.test_arrivals_and_trips import itin, leg, walk


def b_line_plan():
    """Chiswick Road → Harvard: Green Line B to Kenmore, walk, bus 57/66-style transfer kept simple."""
    return {"itineraries": [
        itin(2, 41, [leg("TRAM", "Green-B", ("Chiswick Road", "70127"), ("Harvard Avenue", "70131"), 2, 12, direction=1),
                     walk("Harvard Avenue", "Brighton Ave @ Harvard Ave", 2),
                     leg("BUS", "66", ("Harvard Ave @ Brighton Ave", "1036"), ("Harvard", "2168"), 15, 38),
                     walk("Harvard", "END", 3)], transfers=1),
        itin(5, 48, [leg("TRAM", "Green-B", ("Chiswick Road", "70127"), ("Kenmore", "71151"), 5, 22, direction=1),
                     leg("SUBWAY", "Red", ("Park Street", "70075"), ("Harvard", "70068"), 30, 44, direction=1)], transfers=2),
    ]}


def fake_services(monkeypatch, *, plan=None, plan_status=200, geocode=None, alerts=None, alerts_status=200,
                  google=None, calls=None):
    calls = calls if calls is not None else []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(url)
        if "maps.googleapis.com" in url:
            if isinstance(google, Exception):
                raise google
            return FakeResponse(google or {"status": "ZERO_RESULTS", "routes": []})
        if url.endswith("/v1/geocode"):
            return FakeResponse(geocode(params["text"]) if callable(geocode) else (geocode or []))
        if url.endswith("/v5/plan"):
            if isinstance(plan, Exception):
                raise plan
            return FakeResponse(plan if plan is not None else {"itineraries": []}, status=plan_status)
        if url.startswith(charlie.BASE_URL + "/alerts"):
            return FakeResponse({"data": alerts or []}, status=alerts_status)
        return FakeResponse({"data": []})

    monkeypatch.setattr(charlie.requests, "get", fake_get)
    monkeypatch.setattr(charlie, "GOOGLE_MAPS_API_KEY", None)
    return calls


def planner(client, **body):
    return client.post("/api/trip-planner", json=body)


def test_reported_trip_resolves_abbreviation_locally_and_finds_routes(client, monkeypatch):
    calls = fake_services(monkeypatch, plan=b_line_plan())
    data = planner(client, origin="chiswick rd", destination="harvard").get_json()
    assert data["code"] == "ok" and len(data["routes"]) == 2
    assert data["endpoints"]["origin"]["name"] == "Chiswick Road"
    assert data["endpoints"]["destination"]["similar"] == ["Harvard Ave"]
    assert not any(u.endswith("/v1/geocode") for u in calls)      # both are stations: no geocoding guess


def test_google_failure_falls_back_to_transitous(client, monkeypatch):
    fake_services(monkeypatch, plan=b_line_plan(), google={"status": "REQUEST_DENIED", "error_message": "bad key"})
    monkeypatch.setattr(charlie, "GOOGLE_MAPS_API_KEY", "test-key")
    data = planner(client, origin="Chiswick Road", destination="Hynes").get_json()
    assert data["code"] == "ok" and data["provider"] == "transitous"
    assert data["providers_tried"] == ["google", "transitous"]


def test_google_zero_results_still_tries_transitous(client, monkeypatch):
    fake_services(monkeypatch, plan=b_line_plan())
    monkeypatch.setattr(charlie, "GOOGLE_MAPS_API_KEY", "test-key")
    data = planner(client, origin="Chiswick Road", destination="Hynes").get_json()
    assert data["code"] == "ok" and data["provider"] == "transitous"


def test_every_provider_down_is_an_outage_not_no_routes(client, monkeypatch):
    fake_services(monkeypatch, plan=charlie.requests.ConnectionError("down"), google=charlie.requests.ConnectionError("down"))
    monkeypatch.setattr(charlie, "GOOGLE_MAPS_API_KEY", "test-key")
    res = planner(client, origin="Chiswick Road", destination="Harvard")
    data = res.get_json()
    assert res.status_code == 502 and data["code"] == "routing_unavailable"
    assert data["endpoints"]["origin"]["status"] == "ok"           # the places were fine


def test_http_error_from_router_is_an_outage(client, monkeypatch):
    fake_services(monkeypatch, plan={"error": "x"}, plan_status=503)
    assert planner(client, origin="Chiswick Road", destination="Harvard").get_json()["code"] == "routing_unavailable"


def test_ambiguous_station_asks_instead_of_guessing(client, monkeypatch):
    calls = fake_services(monkeypatch, plan=b_line_plan())
    data = planner(client, origin="park", destination="Harvard").get_json()
    assert data["code"] == "ambiguous" and data["ambiguous"] == ["origin"]
    assert {"Park Street", "Science Park"} <= set(data["candidates"]["origin"])
    assert not any(u.endswith("/v5/plan") for u in calls)


def test_same_station_is_reported(client, monkeypatch):
    fake_services(monkeypatch, plan=b_line_plan())
    assert planner(client, origin="Hynes", destination="Hynes Convention Center").get_json()["code"] == "same_place"


def test_alert_lookup_failure_is_flagged_not_reported_as_clear(client, monkeypatch):
    fake_services(monkeypatch, plan=b_line_plan(), alerts_status=500)
    data = planner(client, origin="Chiswick Road", destination="Harvard").get_json()
    assert data["code"] == "ok" and data["alerts_checked"] is False


def test_alert_lookup_success_is_flagged(client, monkeypatch):
    fake_services(monkeypatch, plan=b_line_plan())
    assert planner(client, origin="Chiswick Road", destination="Harvard").get_json()["alerts_checked"] is True
