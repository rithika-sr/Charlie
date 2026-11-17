"""
Trip phrasing and departure/arrival times for Charlie AI: pull origin, destination
and a requested time out of free text. Pure functions (no Flask, no network).

assistant.parse_trip_request uses this as its broad phrasing fallback and for times;
journeys themselves come from journey.plan(), shared with the Trip Planner.
"""

import re

from datetime import datetime, timedelta

TRIP_WORDS = re.compile(
    r"\b(how (do|can|should|would|could) (i|we|you) (get|go|reach|travel|commute|ride)|how to (get|go|reach|travel)|"
    r"directions?|route|routes|best way|fastest way|quickest way|easiest way|way to (get|go)|take me|get me|"
    r"get from|go from|going from|travel(ling)? from|reach|commute|navigate|plan|trip|journey|between|"
    r"which (train|bus|line|subway|t)|what (train|bus|line)|take the (t|train|bus|subway))\b", re.I)
ARRIVAL_WORDS = re.compile(r"\b(next|when|due|countdown|predictions?|schedule)\b", re.I)

# Trailing filler and follow-on clauses that are never part of a place name.
TAIL = (r"(?:\s+(?:please|pls|now|today|right now|asap|leaving|departing|arriving|starting|by (?:transit|train|bus|subway|the t|t)|on the t|using the t|"
        r"with the t|via (?:the )?mbta))*\s*[?.!]*\s*$")
LEAD_FILLER = re.compile(r"^(?:(?:get|go|going|travel|head|commute|ride|walk|be|reach|arrive)\s+)?(?:to|at|into)?\s*(?:the\s+)?", re.I)
CLAUSE_BREAK = re.compile(r"\s*[,;]\s*(?=(?:what|which|how|is|are|can|could|should|please|thanks|thank|and|so|i)\b).*$", re.I)


def _strip(place):
    p = CLAUSE_BREAK.sub("", place.strip())
    p = re.sub(TAIL, "", p, flags=re.I)
    prev = None
    while prev != p:                                  # "get to MIT" -> "MIT", "go to the Hynes" -> "Hynes"
        prev = p
        p = LEAD_FILLER.sub("", p, count=1).strip()
    p = re.sub(r"^(?:my|our)\s+", "", p, flags=re.I)
    # Purpose clauses aren't part of the place: "Fenway Park for a game", "MGH for an appointment".
    p = re.sub(r"\s+for\s+(?:a|an|the|my|our|some)\s+\w+(?:\s+\w+)?$", "", p, flags=re.I)
    p = re.sub(r"\s+(?:and|so|but)\s+(?:i|we)\b.*$", "", p, flags=re.I)
    return p.strip(" ,.?!\"'")


# ---------- departure / arrival times ----------

TIME_RE = re.compile(
    r"\b(?:(?P<verb>leave|leaving|depart(?:ing)?|arrive|arriving|get there|be there|reach (?:it|there))\s+)?"
    r"(?P<prep>at|by|around|before|after|for)\s+"
    r"(?:(?P<word>noon|midnight)|(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<ap>a\.?\s?m\.?|p\.?\s?m\.?)?)(?![\w:])", re.I)
DAY_RE = re.compile(r"\b(tomorrow(?: morning| afternoon| evening| night)?|tonight|this (?:morning|afternoon|evening))\b", re.I)
IN_RE = re.compile(r"\bin\s+(?P<n>\d{1,3})\s*(?P<u>min(?:ute)?s?|hours?|hrs?)\b", re.I)


def parse_when(text, now):
    """Pull a departure/arrival time out of the text.

    Returns (text_without_time, when) where `when` is None or
    {"time": datetime, "arrive_by": bool, "label": "tomorrow 9:00 AM"}."""
    t = text
    day = DAY_RE.search(t)
    m = TIME_RE.search(t)
    rel = IN_RE.search(t)
    if not (day or m or rel):
        return text, None
    when_dt, arrive_by = None, False
    if rel:
        n = int(rel.group("n"))
        when_dt = (now + (timedelta(hours=n) if rel.group("u").lower().startswith("h") else timedelta(minutes=n))).replace(second=0, microsecond=0)
        t = t[:rel.start()] + t[rel.end():]
    else:
        base = now
        dword = day.group(1).lower() if day else ""
        if dword.startswith("tomorrow"):
            base = now + timedelta(days=1)
        if m:
            if m.group("word"):
                hour, minute = (12, 0) if m.group("word").lower() == "noon" else (0, 0)
            else:
                hour, minute = int(m.group("h")), int(m.group("m") or 0)
                if hour > 23 or minute > 59:
                    return text, None
                ap = (m.group("ap") or "").lower().replace(".", "").replace(" ", "")
                if ap == "pm" and hour < 12:
                    hour += 12
                elif ap == "am" and hour == 12:
                    hour = 0
                elif not ap and hour < 12 and (hour <= 6 or "night" in dword or "evening" in dword or "afternoon" in dword):
                    hour += 12                       # "at 6" usually means 6 PM
            prep = m.group("prep").lower()
            verb = (m.group("verb") or "").lower()
            arrive_by = prep in ("by", "before") or verb.startswith(("arriv", "get there", "be there", "reach"))
        else:
            hour, minute = {"morning": (8, 0), "afternoon": (13, 0), "evening": (18, 0), "night": (20, 0)}.get(
                (dword.split() + [""])[-1] if dword != "tomorrow" else "", (now.hour, now.minute))
            if dword in ("tonight",):
                hour, minute = 20, 0
        when_dt = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if not day and when_dt < now - timedelta(minutes=10):
            when_dt += timedelta(days=1)             # "at 7am" asked in the evening means tomorrow
        if m:
            t = t[:m.start()] + t[m.end():]
        if day:
            t = DAY_RE.sub("", t, count=1)
    return re.sub(r"\s+", " ", t).strip(), {"time": when_dt, "arrive_by": arrive_by, "label": when_label(when_dt, arrive_by, now)}


def when_label(when_dt, arrive_by, now):
    """'leaving today at 6:10 PM' / 'arriving by 9:00 AM tomorrow'."""
    label_day = "today" if when_dt.date() == now.date() else "tomorrow" if when_dt.date() == (now + timedelta(days=1)).date() else when_dt.strftime("%a %b %-d")
    clock = when_dt.strftime('%-I:%M %p')
    return f"arriving by {clock} {label_day}" if arrive_by else f"leaving {label_day} at {clock}"


# ---------- origin / destination ----------

PREFIX = re.compile(
    r"^(?:hey|hi|hello|ok|okay|charlie|please|so)?[\s,]*"
    r"(?:(?:how|what)(?:'s| is| are)?\s+(?:do|can|should|would|could|will)?\s*(?:i|we|you|one)?\s*"
    r"(?:the\s+)?(?:best|fastest|quickest|easiest|shortest)?\s*(?:way|route|routes|option|options|directions)?\s*"
    r"(?:to\s+)?(?:get|go|travel|commute|ride|reach)?|"
    r"(?:show|give|find|get|tell)\s+(?:me\s+)?(?:the\s+)?(?:best\s+|fastest\s+)?(?:route|routes|directions|way|trip|options)?(?:\s+for)?|"
    r"(?:plan|planning)\s+(?:a|my|the)?\s*(?:trip|journey|route)?|"
    r"(?:i\s+(?:want|need|have|would like|'d like|wanna)\s+to\s+(?:go|get|travel|head)|i'?m\s+(?:going|heading|traveling|travelling))|"
    r"(?:which|what)\s+(?:train|bus|line|subway|t|route)s?\s+(?:should\s+i\s+take|do\s+i\s+take|goes|go|runs|to\s+take)?|"
    r"(?:can|could)\s+i\s+(?:take|ride|get|go)(?:\s+the\s+(?:t|train|bus|subway))?|"
    r"(?:route|routes|directions|trip|journey|take me|get me|go|travel|navigate|commute)"
    r")\s*", re.I)


def _near(place, is_place):
    """"Cambridge near Central" -> "Central" when the part after "near" is a known place."""
    m = re.match(r"^(?P<area>.+?)\s+(?:near|by|close to|next to)\s+(?P<spot>.+)$", place, re.I)
    if m and is_place and is_place(m.group("spot")):
        return m.group("spot")
    return place


def _pair(o, d, is_place=None):
    o, d = _near(_strip(o), is_place), _near(_strip(d), is_place)
    return {"origin": o, "destination": d} if o and d and o.lower() != d.lower() else None


def parse_trip_question(text, is_place=None, now=None):
    """Return {"origin": str|None, "destination": str, "when": dict|None} for a trip
    question, else None.

    `is_place(name)` (optional) says whether a phrase resolves to a known station; it
    lets a bare "Chiswick Rd to Hynes" count as a trip without trip words."""
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    if not t:
        return None
    t = t.replace("→", " to ").replace("->", " to ").replace("—>", " to ").replace(" > ", " to ")
    t = re.sub(r"\s+", " ", t)
    asks_trip = bool(TRIP_WORDS.search(t))
    if ARRIVAL_WORDS.search(t) and not re.search(r"\b(how|directions?|route|way|plan|take me|get me|trip|journey|get to|leave|should i)\b", t, re.I):
        return None                      # "next train from Park Street to Alewife" is an arrivals question
    if not asks_trip and re.search(r"\btowards?\b", t, re.I) and \
            re.search(r"\b(?:bus|route|rt)\s*(?:number|no\.?)?\s*#?\s*\d|\b\d{1,3}[a-z]?\s+bus\b|\b(?:train|bus|trolley)\b", t, re.I):
        return None                      # "bus 57 toward Watertown from here" asks for departures, not a trip
    t, when = parse_when(t, now or datetime.now())

    result = None
    patterns = [
        r"\bfrom\s+(?P<o>.+?)\s+(?:to|into|towards?|till|until)\s+(?P<d>.+)$",
        r"\bbetween\s+(?P<o>.+?)\s+and\s+(?P<d>.+)$",
        r"\b(?:leave|leaving|depart|departing)\s+(?P<o>.+?)\s+(?:to get to|to reach|to go to|to make it to|to be at|to)\s+(?P<d>.+)$",
        r"\b(?:to|reach|towards?)\s+(?P<d>.+?)\s+(?:from|starting (?:at|from)|leaving)\s+(?P<o>.+)$",
        r"\b(?:i'?m|i am|i'?m currently|currently|we'?re|we are)\s+(?:at|in|near|by)\s+(?P<o>.+?)[,.;]?\s+(?:and\s+)?(?:how|what|which|can|could|i want|i need|i'?d like|need|want|i have|trying)\b.*?\b(?:to|reach)\s+(?P<d>.+)$",
    ]
    for pat in patterns:
        m = re.search(pat, t, re.I)
        if m:
            result = _pair(m.group("o"), m.group("d"), is_place)
            if result:
                break

    if not result:
        body = PREFIX.sub("", t, count=1) if asks_trip else t
        m = re.fullmatch(r"(?P<o>.+?)\s+to\s+(?P<d>.+)", body.strip(" ?.!"), re.I)
        if m and (asks_trip or (is_place and is_place(_strip(m.group("o"))) and is_place(_strip(m.group("d"))))):
            result = _pair(m.group("o"), m.group("d"), is_place)

    if not result and asks_trip:
        m = re.search(r"\b(?:to|reach|towards?)\s+(?P<d>.+)$", t, re.I)
        if m and _strip(m.group("d")):
            result = {"origin": None, "destination": _strip(m.group("d"))}

    if not result:
        return None
    result["when"] = when
    return result


def parse_trip_followup(text, context, is_place=None):
    """Follow-ups after a trip answer: the missing origin ("Park Street", "from Ruggles"),
    a new destination ("what about to Kendall?") or a new origin ("from Davis instead")."""
    if not context or not context.get("trip_destination"):
        return None
    t = re.sub(r"\s+", " ", str(text or "")).strip(" ?.!")
    if not t or len(t.split()) > 9 or ARRIVAL_WORDS.search(t):
        return None
    dest, origin = context["trip_destination"], context.get("trip_origin")
    m = re.match(r"^(?:(?:what|how) about|and|ok|okay|actually|instead|or)?[\s,]*(?:going\s+)?(?:to|toward|towards)\s+(?P<d>.+?)(?:\s+instead)?$", t, re.I)
    if m and origin:
        return {"origin": origin, "destination": _strip(m.group("d")), "when": None}
    m = re.match(r"^(?:(?:what|how) about|and|ok|okay|actually|instead|or)?[\s,]*(?:from|starting (?:at|from)|leaving|i'?m (?:at|in|near)|i am (?:at|in|near))\s+(?P<o>.+?)(?:\s+instead)?$", t, re.I)
    if m:
        return {"origin": _strip(m.group("o")), "destination": dest, "when": None}
    if not origin and (is_place is None or is_place(_strip(t))):
        return {"origin": _strip(t), "destination": dest, "when": None}   # answer to "Where are you starting from?"
    return None
