"""
Charlie AI trip requests: understand "chiswick rd to harvard", "what about Hynes?" or
"how do I get back?", and phrase short replies from a journey.plan() result.

Pure functions (no Flask, no network). Every journey fact in a reply (times, lines,
transfers, alerts) is read from the structured result; nothing is generated.
Requested departure/arrival times are read with trips.parse_when and sent to the planner.
"""

import re
import time
from datetime import datetime

import trips

# ---------------------------------------------------------------- requirements and preferences
# A time trips.parse_when can read is applied; any other time phrase ("next Monday") and
# accessibility needs are stated as not applied.

TIME_REQUEST = re.compile(
    r"\b(?:(?:arriv\w*|get there|be there|leav\w*|depart\w*|go)\s+(?:by|at|before|after|around)"
    r"|by|at|before|after|around)\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)?|noon|midnight)(?=\s|$)"
    r"|\b(?:tomorrow(?: (?:morning|afternoon|evening|night))?|tonight|this (?:morning|afternoon|evening)|later today|next (?:week|monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"|on (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|in \d+ (?:min\w*|hours?))\b", re.I)
ACCESSIBILITY = re.compile(
    r"\b(?:wheelchair[- ]?(?:accessible|friendly)?|accessib\w*|step[- ]free|no stairs|without stairs|elevators?"
    r"|stroller[- ]?(?:friendly)?|mobility)\b", re.I)
SORT_WORDS = [
    (re.compile(r"\b(?:fewest|least|fewer|no|minimum|min)\s+(?:transfers?|changes?)\b", re.I), "transfers"),
    (re.compile(r"\b(?:least|less|minimal|minimum|no|little)\s+walk\w*\b", re.I), "walking"),
    (re.compile(r"\b(?:fastest|quickest|shortest)\b", re.I), "duration"),
]
_NOW = re.compile(r"\b(?:right now|now|asap|today)\b", re.I)
_FILLER = re.compile(
    r"^(?:(?:hey|hi|hello|ok|okay|so|um|charlie)[, ]+)*"
    r"(?:please\s+)?(?:can|could|would) you\s+(?:please\s+)?(?:tell me|show me|help me|find)?\s*|^(?:please\s+)?(?:tell me|show me|find|help me)\s+",
    re.I)
_TAIL = re.compile(r"\s*(?:,?\s*(?:please|thanks|thank you|pls|thx))+$|[?.!]+$", re.I)

# ---------------------------------------------------------------- trip phrasing

_LEAD = (r"(?:how (?:do|can|should|would|could) (?:i|we) (?:get|go|travel|ride|commute|reach)|how (?:to|i) (?:get|reach)|how to go"
         r"|(?:give me |get |show )?(?:directions|a route|route|routes|a trip|trip|itinerary)"
         r"|get me|take me|bring me|i(?:'m| am) (?:going|heading|trying to get)|i (?:want|need) to (?:go|get)"
         r"|(?:best|fastest|quickest) way(?: to get)?|going|heading|travel|commute|navigate)")
_SEP = r"\s+(?:to|->|→|till|until|toward|towards)\s+|\s*(?:->|→)\s*"
PATTERNS = [
    # "how do I get to Harvard from Chiswick Rd"
    re.compile(rf"^(?:{_LEAD}\s+)?(?:to\s+)(?P<d>.+?)\s+from\s+(?P<o>.+)$", re.I),
    # "directions from Chiswick Rd to Harvard", "from chiswick rd to harvard"
    re.compile(rf"^(?:{_LEAD}\s+)?from\s+(?P<o>.+?)(?:{_SEP})(?P<d>.+)$", re.I),
    # "how do I get from X" (destination asked next)
    re.compile(rf"^(?:{_LEAD}\s+)from\s+(?P<o>.+)$", re.I),
    # "how do I get to Harvard", "take me to Hynes"
    re.compile(rf"^(?:{_LEAD})\s+(?:to\s+)?(?P<d>.+)$", re.I),
    # "I'm at Ruggles, how do I get to Harvard?"
    re.compile(rf"^i(?:'m| am) (?:at|near|in)\s+(?P<o>.+?)[,;]?\s+(?:{_LEAD})\s+(?:to\s+)?(?P<d>.+)$", re.I),
]
# "chiswick rd to harvard": accepted only if a side is a known station or an address.
BARE = re.compile(rf"^(?P<o>[^,;]+?)(?:{_SEP})(?P<d>[^,;]+)$", re.I)
NOT_A_PLACE = re.compile(
    r"\b(?:next|when|train|trains|bus|buses|trolley|arriv\w*|depart\w*|delays?|alerts?|status|talk|speak|listen"
    r"|you|me|it|this|that|hello|reply|answer|subscribe|sign|log)\b", re.I)

FOLLOW_DEST = re.compile(
    r"^(?:(?:and|ok|okay|so|then)\s+)?(?:what|how)\s+about\s+(?:going\s+)?(?:to\s+)?(?P<d>(?!from\b).+)$"
    r"|^(?:and|or)\s+(?:to\s+)?(?P<d2>(?!from\b).+)$"
    r"|^(?:to\s+)?(?P<d3>.+?)\s+instead$|^instead\s+(?:to\s+)?(?P<d4>.+)$", re.I)
FOLLOW_ORIGIN = re.compile(
    r"^(?:(?:and|ok|okay|so|then)\s+)?(?:(?:what|how)\s+about|and|or)?\s*(?:starting\s+)?from\s+(?P<o>.+?)(?:\s+instead)?$", re.I)
FOLLOW_REVERSE = re.compile(
    r"\b(?:way back|trip back|return trip|ride back|reverse(?: it| that| the trip)?|go(?:ing)? back|get back|get home|back home|the other way)\b", re.I)
FOLLOW_REFRESH = re.compile(r"^(?:refresh|update|again|check again|try again|retry|any updates?|reload|same trip)$", re.I)

LINE_NAME = re.compile(r"^(?:the\s+)?(?:red|orange|blue|green|silver|mattapan)(?:\s+line)?(?:\s+[b-e])?(?:\s+(?:line|branch|train))?$", re.I)
_FOLLOW_WORDS = re.compile(r"\b(?:what|how|about|and|then|so|ok|okay|instead|leaving|leave|departing|depart|arriving|arrive|going|if i|i|we)\b", re.I)

_NOT_AN_ANSWER = re.compile(r"^(?:thanks?|thank you|ok|okay|yes|yeah|no|nope|cool|great|hi|hello|hey|never ?mind|cancel|stop|help)\b", re.I)

CONTEXT_TTL_SECONDS = 30 * 60


def _clean(text):
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    t = _FILLER.sub("", t)
    t = _TAIL.sub("", t)
    return t.strip(" ,")


def _place(text):
    t = re.sub(r"^(?:the|a|instead)\s+", "", str(text or "").strip(" ,.?!"), flags=re.I)
    # Words that qualify the request, not the place: "Copley instead", "Kenmore then".
    t = re.sub(r"(?:\s+(?:please|now|right now|asap|today|instead|then|too|as well|leaving|departing|arriving|starting))+$", "", t, flags=re.I)
    return t.strip(" ,") or None


def extract_requirements(text):
    """Pull out the parts of a request the planner can't honor or treats as a preference.
    Returns (remaining text, {"time": phrase|None, "accessibility": phrase|None, "sort": key|None})."""
    found = {"time": None, "accessibility": None, "sort": None}
    t = f" {text} "
    m = TIME_REQUEST.search(t)
    if m:
        found["time"] = m.group(0).strip()
        t = t[:m.start()] + " " + t[m.end():]
    m = ACCESSIBILITY.search(t)
    if m:
        found["accessibility"] = m.group(0).strip()
        t = re.sub(r"\b(?:an?\s+|with\s+|using\s+)?" + re.escape(m.group(0)) + r"(?:\s+(?:route|trip|way|access|options?))?", " ", t, flags=re.I)
    for pattern, key in SORT_WORDS:
        m = pattern.search(t)
        if m:
            found["sort"] = key
            t = t[:m.start()] + " " + t[m.end():]
            break
    t = _NOW.sub(" ", t)
    t = re.sub(r"\s+(?:with|using|on|by|via|and)\s*$", "", re.sub(r"\s+", " ", t).strip())
    t = re.sub(r"^(?:with|using)\s+", "", t)
    return t.strip(" ,"), found


def when_requirement(when):
    """trips.parse_when() result -> the JSON-safe form kept in requirements."""
    return {"time": when["time"].isoformat(), "arrive_by": bool(when["arrive_by"]), "label": when["label"]}


def parse_trip_request(text, context=None, is_place=lambda s: False, now=None):
    """Decide whether `text` asks for a journey. Returns None, or
    {"origin", "destination", "followup", "reverse", "refresh", "requirements"}.

    `context` is the sanitized chat context (see sanitize_context); `is_place(text)` says
    whether a phrase is a known station or an address, used for the bare "X to Y" form.
    `now` (Boston time) turns "by 6 pm" or "tomorrow at 9am" into requirements["when"]."""
    context = context or {}
    trip = context.get("trip") or {}
    pending = context.get("pending") or {}
    raw = _clean(text)
    stripped, when = trips.parse_when(raw, now) if now is not None else (raw, None)
    t, req = extract_requirements(stripped)
    req["when"] = when_requirement(when) if when else None
    if when:
        req["time"] = None
    out = lambda o=None, d=None, **kw: {"origin": _place(o), "destination": _place(d), "followup": False,
                                       "reverse": False, "refresh": False, "requirements": req, **kw}

    # A reply to "Which Park did you mean?" / "Where are you starting from?"
    if pending.get("field") in ("origin", "destination") and t and not re.search(r"\b(?:to|from)\b", t, re.I):
        if is_place(t) or (len(t.split()) <= 5 and not NOT_A_PLACE.search(t) and not _NOT_AN_ANSWER.match(t)):
            field = pending["field"]
            return out(t if field == "origin" else None, t if field == "destination" else None,
                       followup=True, fill=field)

    if trip.get("origin") and trip.get("destination"):
        only_requirements = (req["time"] or req["when"] or req["accessibility"] or req["sort"]) and not _FOLLOW_WORDS.sub("", t).strip(" ,")
        if FOLLOW_REFRESH.match(t) or only_requirements:
            return out(None, None, followup=True, refresh=True)
        if FOLLOW_REVERSE.search(t) and not re.search(r"\bto\s+\w", re.sub(FOLLOW_REVERSE, "", t)):
            return out(None, None, followup=True, reverse=True)
        m = FOLLOW_ORIGIN.match(t)
        if m:
            return out(m.group("o"), None, followup=True)
        m = FOLLOW_DEST.match(t)
        if m:
            d = next(g for g in m.groups() if g)
            if not NOT_A_PLACE.search(d) and not LINE_NAME.match(d.strip()):
                return out(None, d, followup=True)

    for pattern in PATTERNS:
        m = pattern.match(t)
        if m:
            g = m.groupdict()
            o, d = g.get("o"), g.get("d")
            if d and re.match(r"^(?:there|home|work)$", d.strip(), re.I) and not trip:
                return out(o, d)
            return out(o, d)

    # Broader phrasing ("what's the best way to get to MIT from Back Bay?", "between X and Y",
    # "I'm in Cambridge near Central and need to be at Fenway Park").
    found = trips.parse_trip_question(t, is_place=is_place, now=now)
    if found:
        return out(found["origin"], found["destination"])

    m = BARE.match(t)
    if m and len(m.group("o").split()) <= 6 and len(m.group("d").split()) <= 6 \
            and not NOT_A_PLACE.search(m.group("o")) and not NOT_A_PLACE.search(m.group("d")) \
            and (is_place(m.group("o")) or is_place(m.group("d"))):
        return out(m.group("o"), m.group("d"))
    return None


# ---------------------------------------------------------------- per-user chat context

def _endpoint(value):
    if not isinstance(value, dict) or not isinstance(value.get("name"), str):
        return None
    ep = {"name": value["name"][:120]}
    if isinstance(value.get("id"), str) and value["id"].startswith("place-"):
        ep["id"] = value["id"][:40]
    for k in ("lat", "lon"):
        if isinstance(value.get(k), (int, float)) and not isinstance(value.get(k), bool):
            ep[k] = float(value[k])
    return ep


def sanitize_requirements(raw):
    """Requirements carried by a card action or a pending question: short phrases only."""
    raw = raw if isinstance(raw, dict) else {}
    out = {"time": None, "accessibility": None, "sort": None, "when": None}
    for k in ("time", "accessibility"):
        if isinstance(raw.get(k), str) and raw[k].strip():
            out[k] = raw[k].strip()[:60]
    if raw.get("sort") in SORT_CHAINS:
        out["sort"] = raw["sort"]
    when = raw.get("when")
    if isinstance(when, dict) and isinstance(when.get("time"), str):
        try:
            datetime.fromisoformat(when["time"][:40])
            out["when"] = {"time": when["time"][:40], "arrive_by": when.get("arrive_by") is True,
                           "label": str(when.get("label") or "")[:60]}
        except ValueError:
            pass
    return out


REQUIREMENT_KEYS = ("time", "accessibility", "sort", "when")


def merge_requirements(new, old):
    """A follow-up keeps earlier requirements unless it states its own. A new time of
    either kind (readable or not) replaces the old one."""
    old, new = old or {}, new or {}
    merged = {k: new.get(k) or old.get(k) for k in REQUIREMENT_KEYS}
    if new.get("when") or new.get("time"):
        merged["when"], merged["time"] = new.get("when"), new.get("time")
    return merged


def sanitize_context(raw, now=None):
    """The browser keeps each user's context and sends it back; accept only known
    shapes, and drop it entirely once it is older than CONTEXT_TTL_SECONDS."""
    now = now if now is not None else time.time()
    if not isinstance(raw, dict):
        return {}
    at = raw.get("at")
    if isinstance(at, (int, float)) and not isinstance(at, bool):
        if now - at > CONTEXT_TTL_SECONDS or at - now > 300:
            return {}
    ctx = {k: str(raw[k])[:80] for k in ("station_name", "station_id", "route", "near_place") if raw.get(k)}
    if isinstance(raw.get("near_point"), str) and re.fullmatch(r"-?\d{1,3}\.\d+,-?\d{1,3}\.\d+", raw["near_point"][:40]):
        ctx["near_point"] = raw["near_point"]
    if raw.get("kind") in ("arrivals", "trip"):
        ctx["kind"] = raw["kind"]
    trip = raw.get("trip")
    if isinstance(trip, dict):
        o, d = _endpoint(trip.get("origin")), _endpoint(trip.get("destination"))
        if o or d:
            ctx["trip"] = {"origin": o, "destination": d}
            if isinstance(trip.get("requirements"), dict):
                ctx["trip"]["requirements"] = sanitize_requirements(trip["requirements"])
    pending = raw.get("pending")
    if isinstance(pending, dict) and pending.get("field") in ("origin", "destination"):
        ctx["pending"] = {"field": pending["field"], "other": _endpoint(pending.get("other")),
                          "requirements": sanitize_requirements(pending.get("requirements"))}
    if isinstance(at, (int, float)) and not isinstance(at, bool):
        ctx["at"] = at
    return ctx


# ---------------------------------------------------------------- ranking (mirrors trip.js sortTrips)

def _num(v, missing):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else missing


INF = float("inf")
SORT_CHAINS = {
    "arrival": lambda r: (_num(r.get("arrive_ts"), INF) // 60 if _num(r.get("arrive_ts"), None) is not None else INF,
                          _num(r.get("transfers"), INF), _num(r.get("walk_time"), 0), _num(r.get("arrive_ts"), INF),
                          _num(r.get("duration"), INF)),
    "duration": lambda r: (_num(r.get("duration"), INF), _num(r.get("arrive_ts"), INF), _num(r.get("transfers"), INF),
                           _num(r.get("walk_time"), 0)),
    "transfers": lambda r: (_num(r.get("transfers"), INF), _num(r.get("arrive_ts"), INF), _num(r.get("duration"), INF),
                            _num(r.get("walk_time"), 0)),
    "walking": lambda r: (_num(r.get("walk_time"), 0), _num(r.get("arrive_ts"), INF), _num(r.get("duration"), INF),
                          _num(r.get("transfers"), INF)),
}


SORT_CHAINS["departure"] = lambda r: (-_num(r.get("depart_ts"), -INF), _num(r.get("transfers"), INF),
                                      _num(r.get("walk_time"), 0))     # "arrive by": leave as late as possible


def sort_routes(routes, by="arrival"):
    """Same order as the Trip Planner's sortTrips (stable, so the router's order breaks ties)."""
    return sorted(routes, key=SORT_CHAINS.get(by) or SORT_CHAINS["arrival"])


# ---------------------------------------------------------------- replies

SORT_LABEL = {"arrival": "earliest arrival", "duration": "shortest trip", "transfers": "fewest transfers",
              "walking": "least walking", "departure": "latest departure"}


def place_label(ep):
    return (ep or {}).get("name") or (ep or {}).get("query") or "that place"


def caveats_for(result, requirements, location_origin=None):
    """Short notes about what the answer does and doesn't cover, from the request and result."""
    notes = []
    when = requirements.get("when")
    if result.get("missed_deadline") and when:
        notes.append({"kind": "time", "text": f"No trip gets you there {when['label'].replace('arriving ', '')}. "
                                              "These are the soonest arrivals if you leave now."})
    elif when:
        notes.append({"kind": "when", "text": when["label"][:1].upper() + when["label"][1:] + "."})
    elif requirements.get("time"):
        notes.append({"kind": "time", "text": f"I couldn’t read “{requirements['time']}” as a time, so these trips leave now."})
    if requirements.get("accessibility"):
        notes.append({"kind": "accessibility",
                      "text": "I can’t filter for step-free routes yet. Check elevator and escalator status in Alerts before you go."})
    if location_origin:
        notes.append({"kind": "location", "text": f"Starting from your shared location{f' near {location_origin}' if location_origin != True else ''}."})
    for field in ("origin", "destination"):
        ep = result.get(field) or {}
        if ep.get("match") == "fuzzy":
            notes.append({"kind": "match", "text": f"Showing {ep['name']} for “{ep['query']}”."})
    if result.get("code") == "ok" and not result.get("alerts_checked"):
        notes.append({"kind": "alerts", "text": "Service alerts couldn’t be checked for these routes."})
    return notes


def journey_reply(result, requirements=None, sort="arrival"):
    """One or two sentences for the chat bubble; the journey card carries the detail."""
    requirements = requirements or {}
    code = result.get("code")
    o, d = place_label(result.get("origin")), place_label(result.get("destination"))
    trip = f"**{o}** → **{d}**"
    asked = requirements.get("when")
    if code == "ok" and result.get("missed_deadline") and asked:
        best = result["routes"][0]
        soonest = f" The soonest arrival is **{best['arrive_time']}**." if best.get("arrive_time") else ""
        return f"⏰ No trip leaving now gets you from **{o}** to **{d}** {asked['label'].replace('arriving ', '')}.{soonest}"
    if code == "ok":
        best = result["routes"][0]
        when = []
        if best.get("depart_time"):
            when.append(f"leave **{best['depart_time']}**")
        if best.get("arrive_time"):
            when.append(f"arrive **{best['arrive_time']}**")
        transfers = best.get("transfers")
        facts = [best.get("duration_text") or f"{best.get('duration')} min"]
        if isinstance(transfers, int):
            facts.append("no transfers" if transfers == 0 else f"{transfers} transfer{'s' if transfers > 1 else ''}")
        lines = " → ".join(s.get("name", "") for s in best.get("steps", []) if s.get("name"))
        label = SORT_LABEL.get(sort, SORT_LABEL["arrival"])
        if asked:
            trip += f" ({asked['label']})"
        msg = f"{trip}: {', '.join(when) or 'best option'} ({', '.join(facts)})" + (f" on {lines}" if lines else "") + "."
        others = len(result["routes"]) - 1
        if sort != "arrival":
            msg += f" Sorted by {label}."
        if others:
            msg += f" {others} more option{'s' if others > 1 else ''} below."
        affected = len(best.get("alerts") or [])
        if affected:
            msg += f"\n⚠️ {affected} service alert{'s' if affected > 1 else ''} affect{'s' if affected == 1 else ''} this option."
        return msg
    if code == "ambiguous":
        field = result["ambiguous"][0]
        query = (result.get(field) or {}).get("query") or ""
        role = "start from" if field == "origin" else "go to"
        return f"Which “{query}” do you want to {role}?"
    if code == "unresolved":
        bad = [(result.get(f) or {}).get("query") for f in result.get("unresolved", [])]
        names = " or ".join(f"“{b}”" for b in bad if b)
        return f"I couldn’t find {names} as an MBTA station or a Boston-area address. Check the spelling, or add a street or town."
    if code == "same_place":
        return f"**{o}** is both the start and the destination. Where are you heading?"
    if code == "no_itineraries":
        return (f"Both places were found, but the routing service returned no trips "
                f"{asked['label'] if asked else 'leaving now'} for {trip}. "
                "Try a nearby station, or check alerts for closures.")
    if code == "routing_unavailable":
        return (f"The trip-planning service isn’t responding right now, so I can’t show routes for {trip}. "
                "Nothing is wrong with the places you entered — try again in a moment.")
    return "I couldn’t plan that trip."
