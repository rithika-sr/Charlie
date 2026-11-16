"""Station resolution shared by Charlie AI and the Trip Planner (pure, no network)."""

import pytest

import app as charlie
import places

S, A = charlie.STATION_DATA, charlie.STATION_ALIASES


def resolve(q):
    return places.resolve_station(q, S, A)


@pytest.mark.parametrize("typed, expected", [
    ("chiswick rd", "Chiswick Road"),            # reported failure: street abbreviation
    ("Chiswick Rd.", "Chiswick Road"),
    ("chiswik road", "Chiswick Road"),           # misspelling
    ("hynes", "Hynes Convention Center"),        # reported failure: short name
    ("Hynes Convention Ctr", "Hynes Convention Center"),
    ("harvard sq station", "Harvard"),           # suffix words riders add
    ("North Station", "North Station"),          # ...but not when they are the name
    ("south station", "South Station"),
    ("mass ave", "Mass Ave"),
    ("gov ctr", "Government Center"),
    ("kenmore sq", "Kenmore"),
    ("the pru", "Prudential"),
    ("Packard's Corner", "Packards Corner"),
    ("saint marys", "St Marys Street"),
])
def test_ordinary_names_and_abbreviations_resolve(typed, expected):
    r = resolve(typed)
    assert r["status"] == "ok" and r["name"] == expected, r


def test_exact_short_name_wins_but_similar_station_is_reported():
    r = resolve("harvard")
    assert r["name"] == "Harvard" and r["match"] == "exact"
    assert r["similar"] == ["Harvard Ave"]          # the Green Line B stop a rider might mean


@pytest.mark.parametrize("typed, includes", [
    ("park", {"Park Street", "Science Park"}),
    ("St. Paul St", {"St Paul Street B", "St Paul Street C"}),   # same name on two branches
    ("north", {"North Station", "North Quincy"}),
    ("bu", {"BU East", "BU Central", "BU West"}),
])
def test_ambiguous_names_return_candidates(typed, includes):
    r = resolve(typed)
    assert r["status"] == "ambiguous" and includes <= set(r["candidates"]), r


@pytest.mark.parametrize("typed", ["360 Huntington Ave", "Fenway Park", "boston", "zzzz", ""])
def test_addresses_places_and_nonsense_are_not_snapped_to_stations(typed):
    assert resolve(typed)["status"] == "unknown"
