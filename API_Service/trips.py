"""
Charlie AI trip answers: pull origin and destination out of a question and phrase a
reply from the itineraries the Trip Planner returned. Pure functions (no Flask, no
network). Every time, line and step in a reply comes from the routing data.
"""

import math
import re

TRIP_WORDS = re.compile(
    r"\b(how (do|can|should|would) i (get|go|reach|travel)|how to (get|go|reach)|directions?|route from|best way|"
    r"way to (get|go)|take me|get me|get from|go from|travel from|reach|commute|navigate|plan (a|my) (trip|journey)|trip from|journey from)\b", re.I)
ARRIVAL_WORDS = re.compile(r"\b(next|when|arriv\w*|due|countdown|predictions?)\b", re.I)

# Words that end a place phrase.
TAIL = r"(?:\s+(?:please|now|today|tonight|right now|by (?:transit|train|bus|subway|t)|on the t|asap))*\s*[?.!]*\s*$"


def _strip(place):
    p = re.sub(r"^(?:the|my)\s+", "", place.strip(" ,.?!"), flags=re.I)
    return p.strip(" ,.?!")


def parse_trip_question(text, is_place=None):
    """Return {"origin": str|None, "destination": str} for a trip question, else None.

    `is_place(name)` (optional) says whether a phrase resolves to a known station; it
    lets a bare "Chiswick Rd to Hynes" count as a trip without trip words."""
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    if not t:
        return None
    asks_trip = bool(TRIP_WORDS.search(t))
    if ARRIVAL_WORDS.search(t) and not asks_trip:
        return None                      # "next train from Park Street to Alewife" is an arrivals question

    patterns = [
        r"\bfrom\s+(?P<o>.+?)\s+(?:to|into|towards?)\s+(?P<d>.+?)" + TAIL,
        r"\b(?:to|reach|towards?)\s+(?P<d>.+?)\s+from\s+(?P<o>.+?)" + TAIL,
        r"\b(?:i'?m|i am|i'?m currently|currently)\s+(?:at|in|near)\s+(?P<o>.+?)[,.;]?\s+(?:and\s+)?(?:how|what|which|can|i want|i need|i'?d like).*?\b(?:to|reach)\s+(?P<d>.+?)" + TAIL,
    ]
    for pat in patterns:
        m = re.search(pat, t, re.I)
        if m:
            o, d = _strip(m.group("o")), _strip(m.group("d"))
            if o and d:
                return {"origin": o, "destination": d}

    if asks_trip:
        m = re.search(r"\b(?:to|reach)\s+(?P<d>.+?)" + TAIL, t, re.I)
        if m and _strip(m.group("d")):
            return {"origin": None, "destination": _strip(m.group("d"))}

    # Bare "Chiswick Rd to Hynes Convention Center".
    m = re.fullmatch(r"(?P<o>[^?]+?)\s+to\s+(?P<d>[^?]+?)" + TAIL, t, re.I)
    if m and is_place and is_place(_strip(m.group("o"))) and is_place(_strip(m.group("d"))):
        return {"origin": _strip(m.group("o")), "destination": _strip(m.group("d"))}
    return None


def _num(v, missing=math.inf):
    return v if isinstance(v, (int, float)) and math.isfinite(v) else missing


def sort_routes(routes):
    """Earliest arrival by displayed minute, then fewer transfers, less walking, exact
    time, duration and the router's order — the Trip Planner's default sort."""
    def key(item):
        i, r = item
        ts = _num(r.get("arrive_ts"))
        return (ts // 60 if math.isfinite(ts) else math.inf, _num(r.get("transfers")), _num(r.get("walk_time"), 0),
                ts, _num(r.get("duration")), i)
    return [r for _, r in sorted(enumerate(routes), key=key)]


def _transfers(n):
    return "no transfers" if not n else f"{n} transfer{'s' if n > 1 else ''}"


def build_trip_reply(result, planned_at=""):
    """Markdown reply for a plan_trip_core() result."""
    code = result.get("code")
    o = result.get("origin_name") or result.get("origin") or "your starting point"
    d = result.get("destination_name") or result.get("destination") or "your destination"

    if code == "need_origin":
        return (f"Where are you starting from? Tell me a station or address (for example "
                f"“from **Park Street** to **{d}**”), or share your location with the 📍 button.")
    if code == "unresolved":
        bad = result.get("unresolved") or []
        names = [f"“{result.get('origin')}”" if f == "origin" else f"“{result.get('destination')}”" for f in bad]
        which = " or ".join(names)
        return (f"I couldn’t find {which} in the Boston area. Check the spelling, use a station name "
                f"(like **Hynes Convention Center**) or a full street address with the town.")
    if code == "routing_unavailable":
        return (f"The trip planning service didn’t respond, so I can’t plan **{o} → {d}** right now. "
                "Try again in a moment, or open the Trip Planner.")
    if code == "no_itineraries" or not result.get("routes"):
        return (f"I found both **{o}** and **{d}**, but the routing service has no transit itineraries "
                "for this trip right now. Service may have ended for the night — try a nearby station or "
                "open the Trip Planner.")

    routes = result["routes"]
    lines = [f"🗺️ **{o} → {d}** — {len(routes)} option{'s' if len(routes) != 1 else ''}"
             + (f", planned at **{planned_at}**" if planned_at else "") + ", earliest arrival first:"]
    for n in result.get("notices") or []:
        lines.append("")
        lines.append(f"⚠️ **{n['station']} ({'your destination' if n['role'] == 'destination' else 'your starting point'}):** "
                     f"{n.get('header', '')}" + (f" {n['explanation']}" if n.get("explanation") else ""))
    for i, r in enumerate(routes, 1):
        lines.append("")
        facts = [f"**{r.get('duration_text') or str(r.get('duration')) + ' min'}**"]
        if r.get("depart_time") and r.get("arrive_time"):
            facts.append(f"leave **{r['depart_time']}** → arrive **{r['arrive_time']}**")
        facts.append(_transfers(r.get("transfers")))
        if r.get("walk_time"):
            facts.append(f"{r['walk_time']} min walk")
        if r.get("fare"):
            facts.append(f"{r['fare']} est.")
        lines.append(f"**{i}. {r.get('summary') or 'Transit'}** · " + " · ".join(facts))
        if i == 1:
            for s in r.get("detailed_steps") or []:
                when = " · ".join(x for x in (s.get("time"), s.get("duration")) if x)
                lines.append(f"   • {s.get('instruction', '')}" + (f" — {s['details']}" if s.get("details") else "")
                             + (f" ({when})" if when else ""))
        for a in (r.get("alerts") or [])[:2]:
            lines.append(f"   ⚠️ {a.get('step')}: {a.get('service_effect') or a.get('header')}")
    return "\n".join(lines)
