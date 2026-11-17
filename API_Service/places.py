"""
Station-name resolution shared by Charlie AI and the Trip Planner.

Pure functions (no Flask, no network). Riders type "chiswick rd", "mass ave",
"harvard sq station" or "hynes"; both features must agree on what that means, so
both call resolve_station() with the app's station table and aliases.

resolve_station() returns one of:
  {"status": "ok", "name", "id", "match": "exact"|"prefix"|"fuzzy", "similar": [names]}
  {"status": "ambiguous", "query", "candidates": [names]}
  {"status": "unknown", "query"}
"""

import difflib
import re

# Street-suffix and common abbreviations, applied to whole words on both sides of a
# comparison (the query and every station name), so "St Paul St" and "Saint Paul
# Street" normalize the same way without guessing which "st" means saint.
ABBREVIATIONS = {
    "rd": "road", "st": "street", "str": "street", "sq": "square", "ave": "avenue", "av": "avenue",
    "ctr": "center", "cntr": "center", "centre": "center", "cir": "circle", "hts": "heights",
    "jct": "junction", "pk": "park", "pl": "place", "blvd": "boulevard", "hwy": "highway",
    "govt": "government", "gov": "government", "mass": "massachusetts", "univ": "university",
    "saint": "street", "mt": "mount", "sta": "station", "stn": "station",
}

# Trailing words riders add that are not part of a station's name.
_SUFFIX = re.compile(r"(?:\s+(?:t\s+)?(?:station|stop|platform))+$")
_PREFIX = re.compile(r"^(?:the)\s+")
_HAS_STREET_NUMBER = re.compile(r"^\d+[a-z]?\s+\w")


def normalize(text, strip_suffix=True):
    """Lowercase, drop punctuation, expand abbreviations: 'Chiswick Rd.' -> 'chiswick road'.

    strip_suffix drops a trailing "station"/"stop" a rider added; station names keep
    theirs ("North Station" must not become "north")."""
    t = str(text or "").lower().replace("&", " and ").replace("/", " ").replace("’", "'")
    t = re.sub(r"[^a-z0-9' -]", " ", t).replace("'", "").replace("-", " ")
    words = [ABBREVIATIONS.get(w, w) for w in t.split()]
    t = " ".join(words)
    if strip_suffix:
        t = _SUFFIX.sub("", t)
    t = _PREFIX.sub("", t)
    return t.strip()


def looks_like_address(text):
    """'360 Huntington Ave' — a street number means this is an address, not a station."""
    return bool(_HAS_STREET_NUMBER.match(str(text or "").strip().lower()))


def _index(stations, aliases):
    """(canonical name per id, {normalized key: canonical name}).

    The first table entry for a stop id is its canonical name (the same one /api/stations
    shows); later entries with the same id and every alias point at it."""
    canonical, keys = {}, {}
    for name, data in stations.items():
        canonical.setdefault(data["id"], name)
    for name, data in stations.items():
        keys.setdefault(normalize(name, strip_suffix=False), canonical[data["id"]])
    for alias, name in aliases.items():
        if name in stations:
            keys.setdefault(normalize(alias, strip_suffix=False), canonical[stations[name]["id"]])
    return canonical, keys


def _result(name, stations, match, similar=()):
    return {"status": "ok", "name": name, "id": stations[name]["id"], "match": match, "similar": list(similar)}


# Words that are a place on their own (the city, a compass point), so a bare "boston"
# is geocoded instead of being read as the start of "Boston College".
_NOT_A_PREFIX = {"boston", "street", "square", "road", "avenue", "center", "station"}
# Generic trailing words tried away as a last resort: "kenmore sq" -> "kenmore".
_GENERIC_TAIL = {"square", "center", "stop", "t"}


def resolve_station(query, stations, aliases=None):
    """Resolve a typed place to one MBTA station, a short list to choose from, or nothing."""
    aliases = aliases or {}
    full = normalize(query, strip_suffix=False)
    q = normalize(query)
    if not q:
        return {"status": "unknown", "query": query}
    canonical, keys = _index(stations, aliases)

    def ids_of(names):
        out, seen = [], set()
        for n in names:
            sid = stations[n]["id"]
            if sid not in seen:
                seen.add(sid)
                out.append(canonical[sid])
        return out

    # 1. Exact (after normalization) — a full station name or a known alias.
    for key in (full, q):
        if key not in keys:
            continue
        name = keys[key]
        others = [n for n in ids_of(k_name for k, k_name in keys.items() if k.startswith(key + " "))
                  if stations[n]["id"] != stations[name]["id"]]
        # Same name on two Green Line branches ("St Paul Street" B and C): ask.
        branches = [n for n in others if re.fullmatch(re.escape(key) + r" [a-e]", normalize(n, False))]
        if branches:
            return {"status": "ambiguous", "query": query, "candidates": ids_of([name] + branches)}
        # "harvard" is the Red Line station, but "Harvard Ave" exists too: say so.
        return _result(name, stations, "exact", others[:2])

    if looks_like_address(query):
        return {"status": "unknown", "query": query}

    def by_prefix(text):
        words = text.split()
        if len(words) == 1 and words[0] in _NOT_A_PREFIX:
            return []
        found = [name for key, name in sorted(keys.items(), key=lambda kv: len(kv[0]))
                 if all(any(w.startswith(qw) for w in key.split()) for qw in words)]
        return ids_of(found)

    # 2. Every query word starts a word of the name: "chis rd", "hynes conv", "kenm".
    tail = q.split()
    attempts = [q] + ([" ".join(tail[:-1])] if len(tail) > 1 and tail[-1] in _GENERIC_TAIL else [])
    for text in attempts:
        prefix = by_prefix(text)
        if len(prefix) == 1:
            return _result(prefix[0], stations, "prefix")
        if prefix:
            return {"status": "ambiguous", "query": query, "candidates": prefix[:6]}

    # 3. Close spelling ("chiswik road", "symphony stn") — strict, so places such as
    #    "Fenway Park" are geocoded as places rather than snapped to a station.
    if q in _NOT_A_PREFIX or len(q) < 5:
        return {"status": "unknown", "query": query}
    close = difflib.get_close_matches(q, list(keys), n=4, cutoff=0.84)
    close = ids_of(keys[k] for k in close)
    if len(close) == 1:
        return _result(close[0], stations, "fuzzy")
    if close:
        return {"status": "ambiguous", "query": query, "candidates": close[:6]}
    return {"status": "unknown", "query": query}


def station_record(name, stations):
    """Station dict for a canonical name: {name, id, lat, lon, lines}."""
    data = stations[name]
    return {"name": name, "id": data["id"], "lat": data["lat"], "lon": data["lon"], "lines": list(data.get("lines", []))}
