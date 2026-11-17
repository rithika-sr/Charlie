"""
MBTA Real-Time Transit Dashboard with AI Chatbot + Location Integration
A comprehensive transit information system for Boston's public transportation.
Supports: Subway (Red, Orange, Blue, Green), Bus, Commuter Rail, and Ferry.
Includes AI-powered chatbot for natural language queries with Context Memory & Route Logic.
NEW: Location-based station detection and map integration.
UPDATED: Bold timing in chatbot responses for better readability.
"""

import os
import re
import json
import math
import requests
from flask import Flask, jsonify, request, render_template, g
from dotenv import load_dotenv
from dateutil import parser
from datetime import datetime, timedelta
import pytz
from openai import OpenAI

import transit
import arrivals
import trips

# Load Environment Variables
load_dotenv()
MBTA_API_KEY = os.getenv("MBTA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)


def cached(response, seconds):
    """Let Vercel's CDN cache public, non-personal responses briefly."""
    response.headers["Cache-Control"] = f"public, max-age=0, s-maxage={seconds}, stale-while-revalidate={seconds * 2}"
    return response

# ==================== MBTA REQUESTS ====================
# Short-lived in-process cache: identical MBTA calls within a few seconds share one
# response (fewer requests, kinder to rate limits). Non-200 responses raise, so the
# frontend keeps its last good data instead of rendering an empty result.

class MBTAError(RuntimeError):
    pass


_MBTA_TTL = {"/vehicles": 5, "/predictions": 8, "/alerts": 20, "/stops": 3600, "/shapes": 86400, "/routes": 3600, "/route_patterns": 86400}
_mbta_cache = {}


class _CachedResponse:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


def mbta_request(url, headers=None, params=None, timeout=15):
    path = url.replace(BASE_URL, "")
    ttl = next((t for prefix, t in _MBTA_TTL.items() if path.startswith(prefix)), 5)
    key = (path, tuple(sorted((params or {}).items())))
    now_ts = datetime.now().timestamp()
    hit = _mbta_cache.get(key)
    if hit and now_ts - hit[0] < ttl:
        return _CachedResponse(hit[1])
    response = requests.get(url, headers=headers or HEADERS, params=params, timeout=timeout)
    if response.status_code != 200:
        raise MBTAError(f"The MBTA API is unavailable right now (HTTP {response.status_code}).")
    data = response.json()
    if len(_mbta_cache) > 500:
        _mbta_cache.clear()
    _mbta_cache[key] = (now_ts, data)
    return _CachedResponse(data)


# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Configuration
BASE_URL = "https://api-v3.mbta.com"
HEADERS = {"x-api-key": MBTA_API_KEY} if MBTA_API_KEY else {}
BOSTON_TZ = pytz.timezone('America/New_York')


# Per-user state (chat history + location) is kept by the browser and sent with
# each request, so users never share state and it survives serverless restarts.
EMPTY_LOCATION = {"lat": None, "lon": None, "station": None, "stop_id": None, "lines": []}
MAX_HISTORY = 10


@app.before_request
def load_client_state():
    """Load this user's chat history and location from the request body."""
    data = request.get_json(silent=True) if request.is_json else None
    data = data if isinstance(data, dict) else {}

    history = data.get('history') if isinstance(data.get('history'), list) else []
    g.history = [
        {"role": m["role"], "content": m["content"][:2000]}
        for m in history
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
    ][-MAX_HISTORY:]

    location = data.get('location')
    g.location = {**EMPTY_LOCATION, **location} if isinstance(location, dict) else dict(EMPTY_LOCATION)

    # What the last arrivals answer was about, so "what about toward Ashmont?" works.
    context = data.get('context')
    g.chat_context = {k: str(v)[:80] for k, v in context.items()
                      if k in ("station_name", "station_id", "route", "trip_origin", "trip_destination", "near_point", "near_place") and v} if isinstance(context, dict) else {}

# Line Colors for UI
LINE_COLORS = {
    "Red": "#DA291C", "Orange": "#ED8B00", "Blue": "#003DA5",
    "Green-B": "#00843D", "Green-C": "#00843D", "Green-D": "#00843D", "Green-E": "#00843D",
    "Green": "#00843D", "Mattapan": "#DA291C", "Silver": "#7C878E",
    "Bus": "#FFC72C", "CR": "#80276C", "Ferry": "#008EAA"
}

# All Subway Routes
SUBWAY_ROUTES = ["Red", "Orange", "Blue", "Green-B", "Green-C", "Green-D", "Green-E", "Mattapan"]

# Commuter Rail Lines
COMMUTER_RAIL_LINES = [
    "CR-Fairmount", "CR-Fitchburg", "CR-Worcester", "CR-Franklin", "CR-Greenbush",
    "CR-Haverhill", "CR-Kingston", "CR-Lowell", "CR-Middleborough", "CR-Needham",
    "CR-Newburyport", "CR-Providence", "CR-Foxboro"
]

# ==================== COMPREHENSIVE STATION DATA WITH COORDINATES ====================
STATION_DATA = {
    # RED LINE
    "Alewife": {"id": "place-alfcl", "lat": 42.395428, "lon": -71.142483, "lines": ["Red"]},
    "Davis": {"id": "place-davis", "lat": 42.39674, "lon": -71.121815, "lines": ["Red"]},
    "Davis Square": {"id": "place-davis", "lat": 42.39674, "lon": -71.121815, "lines": ["Red"]},
    "Porter": {"id": "place-portr", "lat": 42.3884, "lon": -71.119149, "lines": ["Red"]},
    "Porter Square": {"id": "place-portr", "lat": 42.3884, "lon": -71.119149, "lines": ["Red"]},
    "Harvard": {"id": "place-harsq", "lat": 42.373362, "lon": -71.118956, "lines": ["Red"]},
    "Harvard Square": {"id": "place-harsq", "lat": 42.373362, "lon": -71.118956, "lines": ["Red"]},
    "Central": {"id": "place-cntsq", "lat": 42.365486, "lon": -71.103802, "lines": ["Red"]},
    "Central Square": {"id": "place-cntsq", "lat": 42.365486, "lon": -71.103802, "lines": ["Red"]},
    "Kendall/MIT": {"id": "place-knncl", "lat": 42.362491, "lon": -71.086176, "lines": ["Red"]},
    "Kendall": {"id": "place-knncl", "lat": 42.362491, "lon": -71.086176, "lines": ["Red"]},
    "MIT": {"id": "place-knncl", "lat": 42.362491, "lon": -71.086176, "lines": ["Red"]},
    "Charles/MGH": {"id": "place-chmnl", "lat": 42.361166, "lon": -71.070628, "lines": ["Red"]},
    "Park Street": {"id": "place-pktrm", "lat": 42.356395, "lon": -71.062424, "lines": ["Red", "Green-B", "Green-C", "Green-D", "Green-E"]},
    "Downtown Crossing": {"id": "place-dwnxg", "lat": 42.355518, "lon": -71.060225, "lines": ["Red", "Orange"]},
    "South Station": {"id": "place-sstat", "lat": 42.352271, "lon": -71.055242, "lines": ["Red"]},
    "Broadway": {"id": "place-brdwy", "lat": 42.342622, "lon": -71.056967, "lines": ["Red"]},
    "Andrew": {"id": "place-andrw", "lat": 42.330154, "lon": -71.057655, "lines": ["Red"]},
    "JFK/UMass": {"id": "place-jfk", "lat": 42.320685, "lon": -71.052391, "lines": ["Red"]},
    "JFK": {"id": "place-jfk", "lat": 42.320685, "lon": -71.052391, "lines": ["Red"]},
    "Savin Hill": {"id": "place-shmnl", "lat": 42.31129, "lon": -71.053331, "lines": ["Red"]},
    "Fields Corner": {"id": "place-fldcr", "lat": 42.300093, "lon": -71.061667, "lines": ["Red"]},
    "Shawmut": {"id": "place-smmnl", "lat": 42.29312, "lon": -71.065738, "lines": ["Red"]},
    "Ashmont": {"id": "place-asmnl", "lat": 42.284652, "lon": -71.064489, "lines": ["Red"]},
    "North Quincy": {"id": "place-nqncy", "lat": 42.275275, "lon": -71.029583, "lines": ["Red"]},
    "Wollaston": {"id": "place-wlsta", "lat": 42.2665, "lon": -71.0203, "lines": ["Red"]},
    "Quincy Center": {"id": "place-qnctr", "lat": 42.251809, "lon": -71.005409, "lines": ["Red"]},
    "Quincy Adams": {"id": "place-qamnl", "lat": 42.233391, "lon": -71.007153, "lines": ["Red"]},
    "Braintree": {"id": "place-brntn", "lat": 42.2078, "lon": -71.0011, "lines": ["Red"]},
    
    # ORANGE LINE
    "Oak Grove": {"id": "place-ogmnl", "lat": 42.43668, "lon": -71.071097, "lines": ["Orange"]},
    "Malden Center": {"id": "place-mlmnl", "lat": 42.426632, "lon": -71.07411, "lines": ["Orange"]},
    "Malden": {"id": "place-mlmnl", "lat": 42.426632, "lon": -71.07411, "lines": ["Orange"]},
    "Wellington": {"id": "place-welln", "lat": 42.40237, "lon": -71.077082, "lines": ["Orange"]},
    "Assembly": {"id": "place-astao", "lat": 42.392811, "lon": -71.077257, "lines": ["Orange"]},
    "Assembly Row": {"id": "place-astao", "lat": 42.392811, "lon": -71.077257, "lines": ["Orange"]},
    "Sullivan Square": {"id": "place-sull", "lat": 42.383975, "lon": -71.076994, "lines": ["Orange"]},
    "Sullivan": {"id": "place-sull", "lat": 42.383975, "lon": -71.076994, "lines": ["Orange"]},
    "Community College": {"id": "place-ccmnl", "lat": 42.373622, "lon": -71.069533, "lines": ["Orange"]},
    "North Station": {"id": "place-north", "lat": 42.365577, "lon": -71.06129, "lines": ["Orange", "Green-C", "Green-E"]},
    "Haymarket": {"id": "place-haecl", "lat": 42.363021, "lon": -71.05829, "lines": ["Orange", "Green-C", "Green-E"]},
    "State": {"id": "place-state", "lat": 42.358978, "lon": -71.057598, "lines": ["Orange", "Blue"]},
    "State Street": {"id": "place-state", "lat": 42.358978, "lon": -71.057598, "lines": ["Orange", "Blue"]},
    "Chinatown": {"id": "place-chncl", "lat": 42.352547, "lon": -71.062752, "lines": ["Orange"]},
    "Tufts Medical Center": {"id": "place-tumnl", "lat": 42.349662, "lon": -71.063917, "lines": ["Orange"]},
    "Tufts Medical": {"id": "place-tumnl", "lat": 42.349662, "lon": -71.063917, "lines": ["Orange"]},
    "Back Bay": {"id": "place-bbsta", "lat": 42.34735, "lon": -71.075727, "lines": ["Orange"]},
    "Mass Ave": {"id": "place-masta", "lat": 42.341512, "lon": -71.083423, "lines": ["Orange"]},
    "Massachusetts Avenue": {"id": "place-masta", "lat": 42.341512, "lon": -71.083423, "lines": ["Orange"]},
    "Ruggles": {"id": "place-rugg", "lat": 42.336377, "lon": -71.088961, "lines": ["Orange"]},
    "Roxbury Crossing": {"id": "place-rcmnl", "lat": 42.331397, "lon": -71.095451, "lines": ["Orange"]},
    "Jackson Square": {"id": "place-jaksn", "lat": 42.323132, "lon": -71.099592, "lines": ["Orange"]},
    "Stony Brook": {"id": "place-sbmnl", "lat": 42.317062, "lon": -71.104248, "lines": ["Orange"]},
    "Green Street": {"id": "place-grnst", "lat": 42.310525, "lon": -71.107414, "lines": ["Orange"]},
    "Forest Hills": {"id": "place-forhl", "lat": 42.300523, "lon": -71.113686, "lines": ["Orange"]},
    
    # BLUE LINE
    "Wonderland": {"id": "place-wondl", "lat": 42.41342, "lon": -70.991648, "lines": ["Blue"]},
    "Revere Beach": {"id": "place-rbmnl", "lat": 42.40784, "lon": -70.992533, "lines": ["Blue"]},
    "Beachmont": {"id": "place-bmmnl", "lat": 42.39754, "lon": -70.992319, "lines": ["Blue"]},
    "Suffolk Downs": {"id": "place-sdmnl", "lat": 42.39050, "lon": -70.997123, "lines": ["Blue"]},
    "Orient Heights": {"id": "place-orhte", "lat": 42.386867, "lon": -71.004736, "lines": ["Blue"]},
    "Wood Island": {"id": "place-wimnl", "lat": 42.379542, "lon": -71.022865, "lines": ["Blue"]},
    "Airport": {"id": "place-apts", "lat": 42.374262, "lon": -71.030395, "lines": ["Blue"]},
    "Logan Airport": {"id": "place-apts", "lat": 42.374262, "lon": -71.030395, "lines": ["Blue"]},
    "Maverick": {"id": "place-mvbcl", "lat": 42.36911, "lon": -71.03953, "lines": ["Blue"]},
    "Aquarium": {"id": "place-aqucl", "lat": 42.359784, "lon": -71.051652, "lines": ["Blue"]},
    "Government Center": {"id": "place-gover", "lat": 42.359705, "lon": -71.059215, "lines": ["Blue", "Green-B", "Green-C", "Green-D", "Green-E"]},
    "Govt Center": {"id": "place-gover", "lat": 42.359705, "lon": -71.059215, "lines": ["Blue", "Green-B", "Green-C", "Green-D", "Green-E"]},
    "Bowdoin": {"id": "place-bomnl", "lat": 42.361365, "lon": -71.062037, "lines": ["Blue"]},
    
    # GREEN LINE - COMMON
    "Lechmere": {"id": "place-lech", "lat": 42.370772, "lon": -71.076536, "lines": ["Green-D", "Green-E"]},
    "Science Park": {"id": "place-spmnl", "lat": 42.366664, "lon": -71.067666, "lines": ["Green-C", "Green-E"]},
    "Boylston": {"id": "place-boyls", "lat": 42.35302, "lon": -71.06459, "lines": ["Green-B", "Green-C", "Green-D", "Green-E"]},
    "Arlington": {"id": "place-arln", "lat": 42.351902, "lon": -71.070893, "lines": ["Green-B", "Green-C", "Green-D", "Green-E"]},
    "Copley": {"id": "place-coecl", "lat": 42.349974, "lon": -71.077447, "lines": ["Green-B", "Green-C", "Green-D", "Green-E"]},
    "Hynes Convention Center": {"id": "place-hymnl", "lat": 42.347888, "lon": -71.087903, "lines": ["Green-B", "Green-C", "Green-D"]},
    "Hynes": {"id": "place-hymnl", "lat": 42.347888, "lon": -71.087903, "lines": ["Green-B", "Green-C", "Green-D"]},
    "Kenmore": {"id": "place-kencl", "lat": 42.348949, "lon": -71.095169, "lines": ["Green-B", "Green-C", "Green-D"]},
    "Prudential": {"id": "place-prmnl", "lat": 42.345917, "lon": -71.081696, "lines": ["Green-E"]},
    "Pru": {"id": "place-prmnl", "lat": 42.345917, "lon": -71.081696, "lines": ["Green-E"]},
    "Symphony": {"id": "place-symcl", "lat": 42.342687, "lon": -71.085056, "lines": ["Green-E"]},
    
    # GREEN LINE - B BRANCH
    "Boston College": {"id": "place-lake", "lat": 42.340081, "lon": -71.166769, "lines": ["Green-B"]},
    "BC": {"id": "place-lake", "lat": 42.340081, "lon": -71.166769, "lines": ["Green-B"]},
    "South Street": {"id": "place-sougr", "lat": 42.3396, "lon": -71.1572, "lines": ["Green-B"]},
    "Chestnut Hill Ave": {"id": "place-chill", "lat": 42.338169, "lon": -71.15316, "lines": ["Green-B"]},
    "Chiswick Road": {"id": "place-chswk", "lat": 42.340302, "lon": -71.150711, "lines": ["Green-B"]},
    "Sutherland Road": {"id": "place-sthld", "lat": 42.341614, "lon": -71.146202, "lines": ["Green-B"]},
    "Washington Street": {"id": "place-wascm", "lat": 42.343864, "lon": -71.142853, "lines": ["Green-B"]},
    "Warren Street": {"id": "place-wrnst", "lat": 42.348343, "lon": -71.140457, "lines": ["Green-B"]},
    "Allston Street": {"id": "place-alsgr", "lat": 42.348701, "lon": -71.137955, "lines": ["Green-B"]},
    "Griggs Street": {"id": "place-grigg", "lat": 42.348545, "lon": -71.134949, "lines": ["Green-B"]},
    "Harvard Ave": {"id": "place-harvd", "lat": 42.350243, "lon": -71.131355, "lines": ["Green-B"]},
    "Harvard Avenue": {"id": "place-harvd", "lat": 42.350243, "lon": -71.131355, "lines": ["Green-B"]},
    "Packards Corner": {"id": "place-packr", "lat": 42.351967, "lon": -71.125031, "lines": ["Green-B"]},
    "Babcock Street": {"id": "place-babck", "lat": 42.35182, "lon": -71.12165, "lines": ["Green-B"]},
    "Pleasant Street": {"id": "place-plsgr", "lat": 42.351521, "lon": -71.117738, "lines": ["Green-B"]},
    "St Paul Street B": {"id": "place-stplb", "lat": 42.351340, "lon": -71.114685, "lines": ["Green-B"]},
    "BU West": {"id": "place-buwst", "lat": 42.350941, "lon": -71.113876, "lines": ["Green-B"]},
    "BU Central": {"id": "place-bucen", "lat": 42.350082, "lon": -71.106865, "lines": ["Green-B"]},
    "BU East": {"id": "place-buest", "lat": 42.349735, "lon": -71.103889, "lines": ["Green-B"]},
    "Blandford Street": {"id": "place-bland", "lat": 42.349293, "lon": -71.100258, "lines": ["Green-B"]},
    "Reservoir": {"id": "place-rsmnl", "lat": 42.335088, "lon": -71.148758, "lines": ["Green-B", "Green-C", "Green-D"]},

    # GREEN LINE - C BRANCH
    "Cleveland Circle": {"id": "place-clmnl", "lat": 42.336142, "lon": -71.149326, "lines": ["Green-C"]},
    "Englewood Ave": {"id": "place-engav", "lat": 42.336971, "lon": -71.145876, "lines": ["Green-C"]},
    "Dean Road": {"id": "place-denrd", "lat": 42.337807, "lon": -71.141853, "lines": ["Green-C"]},
    "Tappan Street": {"id": "place-tapst", "lat": 42.338459, "lon": -71.138702, "lines": ["Green-C"]},
    "Washington Square": {"id": "place-bcnwa", "lat": 42.339394, "lon": -71.13533, "lines": ["Green-C"]},
    "Fairbanks Street": {"id": "place-fbkst", "lat": 42.339725, "lon": -71.131073, "lines": ["Green-C"]},
    "Brandon Hall": {"id": "place-bndhl", "lat": 42.340023, "lon": -71.129082, "lines": ["Green-C"]},
    "Summit Ave": {"id": "place-sumav", "lat": 42.341002, "lon": -71.12561, "lines": ["Green-C"]},
    "Coolidge Corner": {"id": "place-cool", "lat": 42.342116, "lon": -71.121263, "lines": ["Green-C"]},
    "St Paul Street C": {"id": "place-stpul", "lat": 42.343327, "lon": -71.116997, "lines": ["Green-C"]},
    "St Paul Street": {"id": "place-stpul", "lat": 42.343327, "lon": -71.116997, "lines": ["Green-C"]},
    "Kent Street": {"id": "place-kntst", "lat": 42.344074, "lon": -71.114064, "lines": ["Green-C"]},
    "Hawes Street": {"id": "place-hwsst", "lat": 42.344906, "lon": -71.111145, "lines": ["Green-C"]},
    "St Marys Street": {"id": "place-smary", "lat": 42.345974, "lon": -71.107353, "lines": ["Green-C"]},

    # GREEN LINE - D BRANCH
    "Riverside": {"id": "place-river", "lat": 42.337352, "lon": -71.252685, "lines": ["Green-D"]},
    "Woodland": {"id": "place-woodl", "lat": 42.3328, "lon": -71.24305, "lines": ["Green-D"]},
    "Waban": {"id": "place-waban", "lat": 42.325845, "lon": -71.230609, "lines": ["Green-D"]},
    "Eliot": {"id": "place-eliot", "lat": 42.319023, "lon": -71.216713, "lines": ["Green-D"]},
    "Newton Highlands": {"id": "place-newtn", "lat": 42.322381, "lon": -71.205509, "lines": ["Green-D"]},
    "Newton Centre": {"id": "place-newto", "lat": 42.329443, "lon": -71.192413, "lines": ["Green-D"]},
    "Chestnut Hill": {"id": "place-chhil", "lat": 42.326653, "lon": -71.164699, "lines": ["Green-D"]},
    "Beaconsfield": {"id": "place-bcnfd", "lat": 42.335765, "lon": -71.140455, "lines": ["Green-D"]},
    "Brookline Hills": {"id": "place-brkhl", "lat": 42.331333, "lon": -71.126999, "lines": ["Green-D"]},
    "Brookline Village": {"id": "place-bvmnl", "lat": 42.332917, "lon": -71.11679, "lines": ["Green-D"]},
    "Longwood": {"id": "place-lngmd", "lat": 42.341145, "lon": -71.110451, "lines": ["Green-D"]},
    "Fenway": {"id": "place-fenwy", "lat": 42.345394, "lon": -71.104187, "lines": ["Green-D"]},

    # GREEN LINE - E BRANCH
    "Heath Street": {"id": "place-hsmnl", "lat": 42.328316, "lon": -71.110252, "lines": ["Green-E"]},
    "Back of the Hill": {"id": "place-bckhl", "lat": 42.330139, "lon": -71.111313, "lines": ["Green-E"]},
    "Riverway": {"id": "place-rvrwy", "lat": 42.331684, "lon": -71.111931, "lines": ["Green-E"]},
    "Mission Park": {"id": "place-mispk", "lat": 42.333195, "lon": -71.109756, "lines": ["Green-E"]},
    "Fenwood Road": {"id": "place-fenwd", "lat": 42.334182, "lon": -71.105667, "lines": ["Green-E"]},
    "Brigham Circle": {"id": "place-brmnl", "lat": 42.334377, "lon": -71.104167, "lines": ["Green-E"]},
    "Longwood Medical Area": {"id": "place-lngmd", "lat": 42.33596, "lon": -71.100052, "lines": ["Green-E"]},
    "MFA": {"id": "place-mfa", "lat": 42.337711, "lon": -71.095512, "lines": ["Green-E"]},
    "Museum of Fine Arts": {"id": "place-mfa", "lat": 42.337711, "lon": -71.095512, "lines": ["Green-E"]},
    "Northeastern": {"id": "place-nuniv", "lat": 42.340401, "lon": -71.088806, "lines": ["Green-E"]},
    "Northeastern University": {"id": "place-nuniv", "lat": 42.340401, "lon": -71.088806, "lines": ["Green-E"]},
    "NEU": {"id": "place-nuniv", "lat": 42.340401, "lon": -71.088806, "lines": ["Green-E"]},
    "Union Square": {"id": "place-unsqu", "lat": 42.377359, "lon": -71.094761, "lines": ["Green-D"]},
    "East Somerville": {"id": "place-esomr", "lat": 42.379482, "lon": -71.086625, "lines": ["Green-D"]},
    "Gilman Square": {"id": "place-gilmn", "lat": 42.387867, "lon": -71.096766, "lines": ["Green-D"]},
    "Magoun Square": {"id": "place-mgngl", "lat": 42.393171, "lon": -71.106046, "lines": ["Green-D"]},
    "Ball Square": {"id": "place-balsq", "lat": 42.399622, "lon": -71.110721, "lines": ["Green-D"]},
    "Medford/Tufts": {"id": "place-mdftf", "lat": 42.407975, "lon": -71.116865, "lines": ["Green-D"]},
    "Medford": {"id": "place-mdftf", "lat": 42.407975, "lon": -71.116865, "lines": ["Green-D"]},
    "Tufts": {"id": "place-mdftf", "lat": 42.407975, "lon": -71.116865, "lines": ["Green-D"]},
    
    # MATTAPAN TROLLEY
    "Mattapan": {"id": "place-matt", "lat": 42.267563, "lon": -71.092526, "lines": ["Mattapan"]},
    "Cedar Grove": {"id": "place-cedgr", "lat": 42.279629, "lon": -71.060394, "lines": ["Mattapan"]},
    "Butler": {"id": "place-butlr", "lat": 42.272429, "lon": -71.062519, "lines": ["Mattapan"]},
    "Milton": {"id": "place-miltt", "lat": 42.270349, "lon": -71.067266, "lines": ["Mattapan"]},
    "Central Avenue": {"id": "place-cenav", "lat": 42.270027, "lon": -71.073448, "lines": ["Mattapan"]},
    "Valley Road": {"id": "place-valrd", "lat": 42.268347, "lon": -71.081343, "lines": ["Mattapan"]},
    "Capen Street": {"id": "place-capst", "lat": 42.267563, "lon": -71.087338, "lines": ["Mattapan"]},
}

# Create STATION_TO_ID for backward compatibility
STATION_TO_ID = {name: data["id"] for name, data in STATION_DATA.items()}

VALID_STATIONS = list(STATION_DATA.keys())

# Station name aliases for typo correction
STATION_ALIASES = {
    "harverd": "Harvard", "harward": "Harvard", "havard": "Harvard",
    "alewhife": "Alewife", "alewif": "Alewife", "alwife": "Alewife",
    "dowtown": "Downtown Crossing", "dtx": "Downtown Crossing",
    "parkstreet": "Park Street", "park st": "Park Street",
    "southstation": "South Station", "south sta": "South Station",
    "northstation": "North Station", "north sta": "North Station",
    "foresthills": "Forest Hills", "forest hill": "Forest Hills",
    "oakgrove": "Oak Grove", "oak": "Oak Grove",
    "backbay": "Back Bay", "back": "Back Bay",
    "govtcenter": "Government Center", "gov center": "Government Center", "govcenter": "Government Center",
    "bc": "Boston College", "boston collge": "Boston College",
    "clevland": "Cleveland Circle", "cleveland": "Cleveland Circle",
    "riverside": "Riverside", "river side": "Riverside",
    "heath": "Heath Street", "heathstreet": "Heath Street",
    "harvard ave": "Harvard Avenue", "harvardave": "Harvard Avenue", "harvard av": "Harvard Avenue",
    "kendal": "Kendall/MIT", "kendall mit": "Kendall/MIT",
    "neu": "Northeastern", "northeastrn": "Northeastern",
    "pru": "Prudential", "prudentail": "Prudential",
    "hines": "Hynes Convention Center", "hynes": "Hynes Convention Center",
    "coolidge": "Coolidge Corner", "cooldige": "Coolidge Corner",
    "sullivan sq": "Sullivan Square", "sully": "Sullivan Square",
    "logan": "Airport", "logan airport": "Airport",
    "mfa": "MFA", "museum": "MFA",
    "tufts": "Tufts Medical Center", "tuftsmedical": "Tufts Medical Center",
    "assembly row": "Assembly", "assemblyrow": "Assembly",
    "union sq": "Union Square", "unionsquare": "Union Square",
    "medford tufts": "Medford/Tufts", "medford": "Medford/Tufts",
}

# Default routes based on destination (for simple lookups)
DEFAULT_ROUTES = {
    "Alewife": {"route": "Red", "dir": 1}, "Ashmont": {"route": "Red", "dir": 0},
    "Braintree": {"route": "Red", "dir": 0}, "JFK/UMass": {"route": "Red", "dir": 0},
    "Oak Grove": {"route": "Orange", "dir": 1}, "Forest Hills": {"route": "Orange", "dir": 0},
    "Wonderland": {"route": "Blue", "dir": 1}, "Bowdoin": {"route": "Blue", "dir": 0},
    "Boston College": {"route": "Green-B", "dir": 0}, "BC": {"route": "Green-B", "dir": 0},
    "Cleveland Circle": {"route": "Green-C", "dir": 0},
    "Riverside": {"route": "Green-D", "dir": 0}, "Union Square": {"route": "Green-D", "dir": 1},
    "Heath Street": {"route": "Green-E", "dir": 0}, "Medford/Tufts": {"route": "Green-E", "dir": 1},
    "Government Center": {"route": "Green-D", "dir": 1}, "Park Street": {"route": "Green-B", "dir": 1},
    "North Station": {"route": "Green-E", "dir": 1}, "Copley": {"route": "Green-B", "dir": 1},
    "Kenmore": {"route": "Green-B", "dir": 1}, "Lechmere": {"route": "Green-E", "dir": 1},
    "Downtown Crossing": {"route": "Red", "dir": 0}, "South Station": {"route": "Red", "dir": 0},
    "Harvard": {"route": "Red", "dir": 1}, "Haymarket": {"route": "Orange", "dir": 0},
    "State": {"route": "Orange", "dir": 0}, "Aquarium": {"route": "Blue", "dir": 0},
    "Airport": {"route": "Blue", "dir": 1}, "Mattapan": {"route": "Mattapan", "dir": 0},
}

# Route aliases
ROUTE_ALIASES = {
    "red": "Red", "red line": "Red", "redline": "Red",
    "orange": "Orange", "orange line": "Orange", "orangeline": "Orange",
    "blue": "Blue", "blue line": "Blue", "blueline": "Blue",
    "green": "Green-B", "green line": "Green-B", "greenline": "Green-B",
    "green b": "Green-B", "green-b": "Green-B", "greenb": "Green-B", "b line": "Green-B", "b train": "Green-B",
    "green c": "Green-C", "green-c": "Green-C", "greenc": "Green-C", "c line": "Green-C", "c train": "Green-C",
    "green d": "Green-D", "green-d": "Green-D", "greend": "Green-D", "d line": "Green-D", "d train": "Green-D",
    "green e": "Green-E", "green-e": "Green-E", "greene": "Green-E", "e line": "Green-E", "e train": "Green-E",
    "mattapan": "Mattapan", "mattapan trolley": "Mattapan", "mattapan line": "Mattapan",
    "silver": "741", "silver line": "741", "sl": "741", "sl1": "741", "sl2": "742", "sl3": "743",
}

# ==================== LOCATION FUNCTIONS ====================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS coordinates in meters."""
    R = 6371000  # Earth's radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c


def find_nearest_station(lat, lon, max_distance=2000):
    """Find the nearest MBTA station within max_distance meters."""
    nearest = None
    min_distance = float('inf')
    
    # Track unique stations by ID to avoid duplicates
    seen_ids = set()
    
    for name, data in STATION_DATA.items():
        if data["id"] in seen_ids:
            continue
        seen_ids.add(data["id"])
        
        distance = haversine_distance(lat, lon, data["lat"], data["lon"])
        if distance < min_distance and distance <= max_distance:
            min_distance = distance
            nearest = {
                "name": name,
                "id": data["id"],
                "distance": round(distance),
                "lines": data["lines"],
                "lat": data["lat"],
                "lon": data["lon"]
            }
    
    return nearest


# ==================== UTILITY FUNCTIONS ====================

def get_now():
    return datetime.now(BOSTON_TZ)

def format_time_diff(arrival_dt):
    now = get_now()
    if arrival_dt.tzinfo is None:
        arrival_dt = BOSTON_TZ.localize(arrival_dt)
    diff = (arrival_dt - now).total_seconds()
    minutes = int(diff / 60)
    if minutes < 0: return "Departed"
    elif minutes == 0: return "Arriving"
    elif minutes == 1: return "1 min"
    elif minutes < 60: return f"{minutes} min"
    else:
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours}h {mins}m" if mins else f"{hours}h"

def format_time(dt_str):
    if not dt_str: return None
    try:
        dt = parser.isoparse(dt_str)
        return dt.strftime("%-I:%M %p")
    except: return None

def normalize_station_name(name):
    if not name: return None
    lower_name = name.lower().strip()
    if lower_name in STATION_ALIASES:
        return STATION_ALIASES[lower_name]
    for station in VALID_STATIONS:
        if station.lower() == lower_name:
            return station
    for station in VALID_STATIONS:
        if lower_name in station.lower() or station.lower() in lower_name:
            return station
    return name.title()

def resolve_route(route_name):
    if not route_name: return None
    normalized = route_name.lower().strip()
    if normalized in ROUTE_ALIASES:
        return ROUTE_ALIASES[normalized]
    if route_name in SUBWAY_ROUTES:
        return route_name
    return None

def find_common_route(origin_name, dest_name):
    if not origin_name or not dest_name: return None
    
    origin_data = STATION_DATA.get(origin_name)
    dest_data = STATION_DATA.get(dest_name)
    
    if origin_data and dest_data:
        origin_lines = set(origin_data.get("lines", []))
        dest_lines = set(dest_data.get("lines", []))
        common = origin_lines.intersection(dest_lines)
        
        if len(common) == 1:
            return list(common)[0]
        elif len(common) > 1:
            for preferred in ["Red", "Orange", "Blue", "Green-D", "Green-E", "Green-B", "Green-C"]:
                if preferred in common:
                    return preferred
    
    return None


# ==================== CHATBOT FUNCTIONS ====================

def get_smart_intent(user_text):
    """Step 1: Understand what the user wants with Memory Logic and Location Awareness"""
    
    current_message = {"role": "user", "content": user_text}
    g.history.append(current_message)
    if len(g.history) > MAX_HISTORY:
        g.history.pop(0)

    location_context = ""
    if g.location.get("station"):
        location_context = f"""
    USER'S CURRENT LOCATION: The user is currently at or near '{g.location['station']}' station.
    If the user asks about "next train" or "going to [destination]" WITHOUT specifying origin, 
    you should set origin to "USE_CURRENT_LOCATION" to indicate we should use their GPS location.
    """

    system_prompt = f"""You are an expert MBTA Transit Assistant. 
    
    CRITICAL RULE - MEMORY MERGE:
    The user might reply with just one word (e.g., "Orange").
    You MUST look at the previous messages.
    If the previous message had an Origin and Destination, YOU MUST COPY THEM to the new output.
    DO NOT output null for origin/destination if they were mentioned in the last 3 turns.
    
    LOCATION AWARENESS:
    {location_context}
    
    TRIP PLANNING DETECTION:
    If user asks HOW TO GET from one place to another, or asks for DIRECTIONS, ROUTE, or WAY TO GO,
    use intent "trip_planning". Examples:
    - "How do I get from Harvard to Airport?"
    - "What's the best way to go from Park Street to Forest Hills?"
    - "I'm at Ruggles, how can I reach Harvard?"
    - "Directions from South Station to Fenway"
    - "How to travel from Downtown to Assembly?"
    - "Route from Copley to Wonderland"
    - "I want to go from Central to Back Bay"
    - "Take me from Davis to Airport"
    
    OUTPUT JSON:
    {{
        "intent": "prediction" | "greeting" | "thanks" | "alerts" | "bus" | "commuter_rail" | "trip_planning" | "help" | "general",
        "origin": "Station Name (normalized) or USE_CURRENT_LOCATION if user didn't specify and we should use GPS",
        "destination": "Station Name (normalized)",
        "specific_route": "Green-B" (if mentioned OR inferred),
        "time_offset": 0,
        "bus_route": "66" (if bus),
        "commuter_line": "Worcester" (if commuter rail),
        "original_station_mentioned": "what user typed"
    }}
    
    VALID STATIONS: {json.dumps(VALID_STATIONS[:80])}
    """

    messages = [{"role": "system", "content": system_prompt}] + g.history

    try:
        if not client: return None
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.1
        )
        result = json.loads(response.choices[0].message.content)
        print(f"AI Intent: {result}")
        return result
    except Exception as e:
        print(f"OpenAI Error: {e}")
        return None

def generate_natural_response(user_text, system_data):
    """Step 3: Generate human response with BOLD TIMING for easy reading"""
    if not client: 
        return "I found the data, but I'm in basic mode. Check the JSON response."

    system_prompt = """
    You are a friendly, local Boston MBTA transit assistant. 
    I will provide you with specific REAL-TIME data found by the system.
    
    Your Goal: Answer the user's question naturally using the provided data.
    
    ═══════════════════════════════════════════════════════════════
    CRITICAL FORMATTING RULES - MAKE TIMING STAND OUT:
    ═══════════════════════════════════════════════════════════════
    
    1. ALWAYS make timing information BOLD using **double asterisks**:
       - Minutes away: **2 min**, **5 min**, **12 min**, **Arriving**
       - Clock times: **10:45 AM**, **3:30 PM**, **11:02 AM**
       - Duration: **25 min total**, **8 min walk**
    
    2. RESPONSE STRUCTURE for train predictions:
       - Start with brief context (location if auto-detected)
       - Show the NEXT train prominently with both time formats
       - List additional trains in a clear format
       - Keep it scannable and easy to read
    
    3. EXAMPLE FORMAT for predictions:
       
       🚇 From Harvard to Boston College on the Green-B:
       
       🔜 **Next train: 3 min** (at **10:42 AM**)
       
       Following trains:
       • **8 min** — **10:47 AM**
       • **15 min** — **10:54 AM**
       • **22 min** — **11:01 AM**
    
    4. If location was auto-detected, mention it:
       "📍 Since you're near **Harvard**..."
    
    ═══════════════════════════════════════════════════════════════
    TRIP PLANNING RESPONSE FORMAT (VERY IMPORTANT):
    ═══════════════════════════════════════════════════════════════
    
    When intent is "trip_planning" and routes are provided, you MUST show ALL routes.
    Format EXACTLY like this:
    
    🗺️ **Directions from [Origin] to [Destination]**
    
    I found **[X] routes** for you:
    
    ---
    
    **🥇 Route 1 (Recommended):** **[duration]** | [transfers] | 💰 [fare]
    📍 [depart_time] — [arrive_time]
    
    Steps:
    1. 🚶 Walk to [station] (**X min**)
    2. 🚇 Take **[Line Name]** toward [direction] (**X min**, X stops)
       → Board at **[station]**, exit at **[station]**
    3. 🔄 Transfer to **[Line Name]**
    4. 🚌 Take **Bus [number]** (**X min**, X stops)
    5. 🚶 Walk to destination (**X min**)
    
    ---
    
    **🥈 Route 2:** **[duration]** | [transfers] | 💰 [fare]
    📍 [time range]
    • [Line 1] → [Line 2] → ...
    
    ---
    
    **🥉 Route 3:** **[duration]** | [transfers] | 💰 [fare]
    📍 [time range]
    • [Line 1] → [Line 2] → ...
    
    (Continue for all routes provided)
    
    ---
    
    CRITICAL RULES FOR TRIP PLANNING:
    - Show ALL routes from the data (usually 3-6 routes)
    - Make DURATION bold and prominent: **42 mins**
    - Make LINE NAMES bold: **Green Line B**, **Bus 66**, **Orange Line**
    - Show the step_summary for quick overview: Green-B → Bus 66 → Orange Line
    - Include fare, transfers, and walk time for each route
    - The FIRST route should show detailed steps
    - Other routes can show summary (line sequence)
    - Use medals: 🥇 🥈 🥉 for top 3, then numbers for rest
    
    ═══════════════════════════════════════════════════════════════
    
    Other Guidelines:
    - Talk like a helpful human, not a robot
    - Be concise but conversational  
    - Use emojis: 🚇 subway, 🚌 bus, 🚂 commuter rail, 🚶 walk, 🔄 transfer, ⛴️ ferry
    - If no predictions, apologize and suggest alternatives
    - Do NOT invent data - only use what is provided
    
    THE MOST IMPORTANT THING: Show ALL routes and make timing/line names **BOLD**!
    """
    
    user_prompt = f"""
    User Input: "{user_text}"
    System Data Found: {json.dumps(system_data)}
    
    Write the response showing ALL routes with timing and line names in **bold**.
    For trip_planning, show every route from the data.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Generation Error: {e}")
        return "I found the data, but I'm having trouble phrasing it right now."

def get_bus_predictions(route_id, stop_query=None, dest_query=None):
    try:
        stops_resp = requests.get(f"{BASE_URL}/stops", headers=HEADERS,
            params={"filter[route]": route_id, "page[limit]": 100})
        stops = stops_resp.json().get('data', [])
        
        target_stop = None
        if stop_query:
            query_lower = stop_query.lower()
            for stop in stops:
                if query_lower in stop['attributes'].get('name', '').lower():
                    target_stop = stop
                    break
        
        if not target_stop:
            return [], None

        pred_resp = requests.get(f"{BASE_URL}/predictions", headers=HEADERS,
            params={
                "filter[stop]": target_stop['id'], 
                "filter[route]": route_id,
                "include": "trip",
                "sort": "arrival_time", 
                "page[limit]": 10
            })
        
        data = pred_resp.json()
        predictions = []
        
        included_trips = {item['id']: item for item in data.get('included', []) if item['type'] == 'trip'}
        
        for pred in data.get('data', []):
            trip_id = pred['relationships']['trip']['data']['id']
            trip_info = included_trips.get(trip_id, {})
            headsign = trip_info.get('attributes', {}).get('headsign', 'Unknown')
            
            if dest_query:
                if dest_query.lower() not in headsign.lower():
                    continue 

            arrival = pred['attributes'].get('arrival_time') or pred['attributes'].get('departure_time')
            if arrival:
                arrival_dt = parser.isoparse(arrival)
                predictions.append({
                    "time_away": format_time_diff(arrival_dt),
                    "arrival_time": format_time(arrival),
                    "headsign": headsign
                })
                
        return predictions, target_stop['attributes'].get('name', '')
    except Exception as e:
        print(f"Bus prediction error: {e}")
        return [], None


def get_commuter_rail_predictions(line_name, stop_query=None, dest_query=None):
    try:
        route_id = None
        for cr_line in COMMUTER_RAIL_LINES:
            if line_name.lower() in cr_line.lower():
                route_id = cr_line
                break
        if not route_id: return [], None
        
        stops_resp = requests.get(f"{BASE_URL}/stops", headers=HEADERS,
            params={"filter[route]": route_id, "page[limit]": 100})
        stops = stops_resp.json().get('data', [])
        
        target_stop = None
        if stop_query:
            query_lower = stop_query.lower()
            for stop in stops:
                if query_lower in stop['attributes'].get('name', '').lower():
                    target_stop = stop
                    break
        if not target_stop and stops: target_stop = stops[0]
        
        if target_stop:
            pred_resp = requests.get(f"{BASE_URL}/predictions", headers=HEADERS,
                params={
                    "filter[stop]": target_stop['id'], 
                    "filter[route]": route_id,
                    "include": "trip",
                    "sort": "arrival_time", 
                    "page[limit]": 10
                })
            
            data = pred_resp.json()
            predictions = []
            included_trips = {item['id']: item for item in data.get('included', []) if item['type'] == 'trip'}

            for pred in data.get('data', []):
                trip_id = pred['relationships']['trip']['data']['id']
                trip_info = included_trips.get(trip_id, {})
                headsign = trip_info.get('attributes', {}).get('headsign', 'Unknown')
                
                if dest_query and dest_query.lower() not in headsign.lower():
                    continue

                arrival = pred['attributes'].get('arrival_time') or pred['attributes'].get('departure_time')
                if arrival:
                    arrival_dt = parser.isoparse(arrival)
                    predictions.append({
                        "time_away": format_time_diff(arrival_dt),
                        "arrival_time": format_time(arrival),
                        "headsign": headsign
                    })
            return predictions, target_stop['attributes'].get('name', '')
        return [], None
    except Exception as e:
        print(f"Commuter rail error: {e}")
        return [], None


def get_alerts_for_route(route_id=None, limit=5):
    params = {"sort": "-severity", "page[limit]": limit}
    if route_id: params["filter[route]"] = route_id
    
    try:
        response = requests.get(f"{BASE_URL}/alerts", headers=HEADERS, params=params)
        data = response.json()
        alerts = []
        for alert in data.get('data', []):
            attrs = alert['attributes']
            alerts.append({
                "header": attrs.get('header', ''),
                "effect": attrs.get('effect', ''),
                "severity": attrs.get('severity', 0)
            })
        return alerts
    except Exception as e:
        print(f"Alert error: {e}")
        return []


def route_meta(route_id):
    """Route attributes (direction names/destinations), cached for an hour."""
    try:
        return mbta_request(f"{BASE_URL}/routes/{route_id}").json().get('data', {}).get('attributes', {})
    except Exception:
        return {}


def answer_arrivals(parsed):
    """Answer an arrivals question from the same live predictions the station board shows."""
    now = get_now()
    route_filter = parsed.get("route")
    answer = {"status": "ok", "route": route_filter, "route_label": arrivals.route_label(route_filter, []),
              "toward": None, "groups": [], "alerts": []}

    if parsed.get("near_point"):
        lat, lon = (float(x) for x in parsed["near_point"].split(","))
        return answer_near(parsed, lat, lon, place=parsed.get("near_place"))
    if parsed.get("near_me") or not parsed.get("station_query"):
        return answer_near(parsed)
    is_bus = bool(route_filter) and (route_filter[:1].isdigit() or route_filter == "SILVER")

    resolved = arrivals.resolve_station(parsed["station_query"], STATION_DATA, STATION_ALIASES)
    if resolved["status"] == "unknown" and is_bus:
        # A bus stop or street, not a station ("next 39 at Huntington Ave and Opera Pl").
        try:
            point = _resolve_place(parsed["station_query"])
        except RoutingError:
            point = None
        if point:
            lat, lon = (float(x) for x in point.split(","))
            return answer_near(parsed, lat, lon, place=parsed["station_query"].title())
    if resolved["status"] != "ok":
        answer.update(resolved)
        return answer
    station = {"id": resolved["id"], "name": resolved["name"]}
    answer["station"] = station

    try:
        preds = station_arrivals(station["id"], now)
    except Exception as e:
        print(f"Arrivals unavailable: {e}")
        answer["status"] = "unavailable"
        return answer
    answer["updated_at"] = now.isoformat()
    answer["checked_text"] = now.strftime("%-I:%M %p")

    preds = [p for p in preds if arrivals.route_matches(route_filter, p["route_id"])]
    data = STATION_DATA.get(station["name"]) or {}
    if not preds and is_bus and data.get("lat") is not None:
        # Buses often stop on the street outside a station, not at the station itself.
        return answer_near(parsed, data["lat"], data["lon"], place=station["name"])

    # Direction: a headsign/destination ("toward Ashmont") or a compass word ("southbound").
    toward = parsed.get("toward")
    filtered = preds
    if toward:
        toward_station = arrivals.resolve_station(toward, STATION_DATA, STATION_ALIASES)
        matched = transit.filter_toward(preds, toward)
        if not matched and toward_station["status"] == "ok":
            matched = transit.filter_toward(preds, toward_station["name"])
            if not matched:
                # A station along the line (e.g. JFK/UMass): use its known direction.
                default = DEFAULT_ROUTES.get(toward_station["name"])
                if default and arrivals.route_matches(route_filter or default["route"], default["route"]):
                    matched = [p for p in preds if p["route_id"] == default["route"] and p["direction_id"] == default["dir"]]
        if matched or toward_station["status"] == "ok":
            filtered = matched
            answer["toward"] = toward_station.get("name") if toward_station["status"] == "ok" else toward.title()
        # Otherwise the phrase wasn't a destination ("how long to wait") — ignore it.
    elif parsed.get("direction") is not None:
        filtered = [p for p in preds if p["direction_id"] == parsed["direction"][1]]
        answer["toward"] = parsed.get("direction_word")

    metas = {}
    def meta(rid):
        if rid not in metas:
            metas[rid] = route_meta(rid)
        return metas[rid]

    groups = transit.next_by_direction(filtered, per_direction=3)
    # Subway first, then Commuter Rail, then buses; each by its soonest prediction.
    mode_rank = lambda g: 0 if g["route_id"] in SUBWAY_ROUTES else 1 if g["route_id"].startswith("CR-") else 2
    groups.sort(key=mode_rank)
    multi_route = len({g["route_id"] for g in groups}) > 1
    for g in groups[:6]:
        m = meta(g["route_id"])
        answer["groups"].append({
            "route_id": g["route_id"],
            "direction_id": g["direction_id"],
            "label": arrivals.direction_label(g, m),
            "route_name": (arrivals.route_label(g["route_id"], []) if g["route_id"] in SUBWAY_ROUTES
                           else (f"Route {m.get('short_name') or g['route_id']}" if g["route_id"][:1].isdigit() else m.get("long_name") or g["route_id"]))
                          if multi_route else None,
            "items": [{
                "headsign": p["headsign"], "time": p["time"], "event": p["event"], "status": p["status"],
                "platform_name": p["platform_name"], "route_id": p["route_id"], "direction_id": p["direction_id"],
                "countdown": arrivals.countdown_text(p, now), "clock": format_time(p["time"]) or "",
            } for p in g["items"]],
        })

    if not answer["groups"]:
        answer["status"] = "none"
        if filtered is not preds and preds:
            answer["other_directions"] = [
                {"label": arrivals.direction_label(g, meta(g["route_id"])), "next": arrivals.countdown_text(g["items"][0], now)}
                for g in transit.next_by_direction(preds, 1)[:3]]

    # Active service alerts at this station on the lines asked about.
    try:
        raw = mbta_request(f"{BASE_URL}/alerts", params={"filter[stop]": station["id"]}).json().get('data', [])
        for a in sorted((transit.alert_payload(x, now) for x in raw), key=transit.alert_priority):
            if a["active_now"] and a["category"] == "service" and a["severity"] >= transit.STATUS_MIN_SEVERITY \
                    and (not route_filter or any(arrivals.route_matches(route_filter, r) for r in a["affected_routes"])):
                answer["alerts"].append(a["header"])
    except Exception:
        pass
    return answer


SILVER_ROUTES = ["741", "742", "743", "746", "749", "751"]
NEAR_ROUTE_RADIUS_M = 1200     # how far to look for a stop on a named route
NEAR_ANY_RADIUS_M = 700        # "what's leaving near me" — stops within an easy walk


def route_ids_for(route_filter):
    """MBTA route ids behind a parsed route filter (None means every route)."""
    if not route_filter:
        return None
    if route_filter == "Green":
        return ["Green-B", "Green-C", "Green-D", "Green-E"]
    if route_filter == "SILVER":
        return list(SILVER_ROUTES)
    if route_filter == "CR":
        return list(COMMUTER_RAIL_LINES)
    return [route_filter]


def route_display_name(route_id, meta):
    if route_id in SUBWAY_ROUTES:
        return arrivals.route_label(route_id, [])
    if route_id[:1].isdigit():
        return f"Route {meta.get('short_name') or route_id}" if route_id not in SILVER_ROUTES else meta.get("short_name") or route_id
    return meta.get("long_name") or meta.get("short_name") or route_id


def stops_near(lat, lon, route_ids=None, radius_m=NEAR_ROUTE_RADIUS_M):
    """Stops within radius_m of a point (optionally only on some routes), nearest first."""
    params = {"filter[latitude]": round(float(lat), 5), "filter[longitude]": round(float(lon), 5),
              "filter[radius]": round(radius_m / 111000, 5), "sort": "distance", "page[limit]": 40}
    if route_ids:
        params["filter[route]"] = ",".join(route_ids)
    else:
        params["filter[location_type]"] = "0"          # boarding stops and platforms
    data = mbta_request(f"{BASE_URL}/stops", params=params).json().get('data', [])
    stops = {}
    for s in data:
        a = s.get('attributes', {})
        if a.get('latitude') is None:
            continue
        d = haversine_distance(float(lat), float(lon), a['latitude'], a['longitude'])
        if d <= radius_m * 1.05:
            stops[s['id']] = {"id": s['id'], "name": a.get('name', ''), "distance": round(d),
                              "lat": a['latitude'], "lon": a['longitude']}
    return dict(sorted(stops.items(), key=lambda kv: kv[1]["distance"]))


def direction_filter(preds, toward, direction, meta):
    """Predictions heading toward a place (headsign, then the route's direction names)."""
    if direction is not None:
        return [p for p in preds if p["direction_id"] == direction[1]]
    if not toward:
        return preds
    matched = transit.filter_toward(preds, toward)
    if matched:
        return matched
    t = toward.lower()
    keep = []
    for p in preds:
        dests = meta(p["route_id"]).get("direction_destinations") or []
        d = p.get("direction_id")
        if isinstance(d, int) and d < len(dests) and dests[d] and (t in dests[d].lower() or dests[d].lower() in t):
            keep.append(p)
    if keep:
        return keep
    station = arrivals.resolve_station(toward, STATION_DATA, STATION_ALIASES)
    if station["status"] == "ok":
        return transit.filter_toward(preds, station["name"])
    return None                                         # not a destination we can match ("how long")


def answer_near(parsed, lat=None, lon=None, place=None):
    """Next departures at the nearest stops to a point: a named route ("next 39 toward Back
    Bay near me") or every route ("what's leaving near me?"). Defaults to the rider's location."""
    now = get_now()
    route_filter = parsed.get("route")
    answer = {"status": "ok", "kind": "near", "route": route_filter,
              "route_label": arrivals.route_label(route_filter, []), "toward": None, "groups": [], "alerts": [],
              "place": place}
    if lat is None or lon is None:
        lat, lon = g.location.get("lat"), g.location.get("lon")
    if lat is None or lon is None:
        answer["status"] = "need_location"
        return answer
    answer["point"] = f"{float(lat):.5f},{float(lon):.5f}"
    route_ids = route_ids_for(route_filter)
    radius = NEAR_ROUTE_RADIUS_M if route_ids else NEAR_ANY_RADIUS_M
    answer["radius_m"] = radius
    try:
        stops = stops_near(lat, lon, route_ids, radius)
        if not stops:
            answer["status"] = "no_stops"
            return answer
        ids = list(stops)[:25]
        params = {"filter[stop]": ",".join(ids), "include": "route,trip,stop", "page[limit]": 200}
        if route_ids:
            params["filter[route]"] = ",".join(route_ids)
        preds = prediction_rows(mbta_request(f"{BASE_URL}/predictions", params=params).json(), now)
    except Exception as e:
        print(f"Nearby departures unavailable: {e}")
        answer["status"] = "unavailable"
        return answer
    answer["updated_at"] = now.isoformat()
    answer["checked_text"] = now.strftime("%-I:%M %p")
    # Platforms of a station on the route: map them back to the station we found.
    names = {s["name"].lower(): sid for sid, s in stops.items()}
    for p in preds:
        if p["stop_id"] not in stops and p.get("parent_id") not in stops and p.get("stop_name", "").lower() in names:
            p["parent_id"] = names[p["stop_name"].lower()]
    preds = [p for p in preds if arrivals.route_matches(route_filter, p["route_id"])]

    metas = {}
    def meta(rid):
        if rid not in metas:
            metas[rid] = route_meta(rid)
        return metas[rid]

    toward = parsed.get("toward")
    filtered = direction_filter(preds, toward, parsed.get("direction"), meta)
    if filtered is None:
        filtered = preds
    elif toward:
        found = arrivals.resolve_station(toward, STATION_DATA, STATION_ALIASES)
        answer["toward"] = found["name"] if found["status"] == "ok" else toward.title()
    elif parsed.get("direction") is not None:
        answer["toward"] = parsed.get("direction_word")

    groups = transit.nearest_stop_departures(filtered, stops, per_group=3, max_groups=8 if route_ids else 10)
    for grp in groups:
        m = meta(grp["route_id"])
        walk = transit.walk_minutes(grp["distance"])
        answer["groups"].append({
            "route_id": grp["route_id"], "direction_id": grp["direction_id"],
            "label": arrivals.direction_label(grp, m), "route_name": route_display_name(grp["route_id"], m),
            "stop_id": grp["stop_id"], "stop_name": grp["stop_name"], "distance_m": grp["distance"], "walk_min": walk,
            "items": [{
                "headsign": p["headsign"], "time": p["time"], "event": p["event"], "status": p["status"],
                "platform_name": p["platform_name"], "route_id": p["route_id"], "direction_id": p["direction_id"],
                "countdown": arrivals.countdown_text(p, now), "clock": format_time(p["time"]) or "",
                "catchable": (arrivals.minutes_until(p["time"], now) or 0) >= walk,
            } for p in grp["items"]],
        })
    if answer["groups"]:
        first = answer["groups"][0]
        answer["station"] = {"id": first["stop_id"], "name": first["stop_name"]}
    else:
        answer["status"] = "none"
        if filtered is not preds and preds:
            answer["other_directions"] = [
                {"label": arrivals.direction_label(x, meta(x["route_id"])), "next": arrivals.countdown_text(x["items"][0], now)}
                for x in transit.nearest_stop_departures(preds, stops, per_group=1)[:3]]

    try:
        if route_ids:
            raw = mbta_request(f"{BASE_URL}/alerts", params={"filter[route]": ",".join(route_ids)}).json().get('data', [])
            for a in sorted((transit.alert_payload(x, now) for x in raw), key=transit.alert_priority):
                if a["active_now"] and a["category"] == "service" and a["severity"] >= transit.STATUS_MIN_SEVERITY:
                    answer["alerts"].append(a["header"])
    except Exception:
        pass
    return answer


def arrivals_response(user_msg, parsed):
    """Chat response for an arrivals question: grounded text plus structured arrivals."""
    if not any(m.get("role") == "user" and m.get("content") == user_msg for m in g.history[-1:]):
        g.history.append({"role": "user", "content": user_msg})
    answer = answer_arrivals(parsed)
    reply = arrivals.build_reply(answer, get_now())
    g.history.append({"role": "assistant", "content": reply})
    result = {"reply": reply, "arrivals": answer}
    if answer.get("kind") == "near":
        if answer["status"] == "need_location":
            result["request_location"] = True
        point = answer.get("point")
        if point:
            result["context"] = {k: v for k, v in {"near_point": point, "near_place": answer.get("place"),
                                                    "route": answer.get("route")}.items() if v}
    elif answer.get("station"):
        result["context"] = {"station_name": answer["station"]["name"], "station_id": answer["station"]["id"],
                             "route": answer.get("route")}
    return result


def trip_response(user_msg, origin, destination, when=None):
    """Chat answer for a trip question, from the same planner as the Trip Planner page."""
    if not any(m.get("role") == "user" and m.get("content") == user_msg for m in g.history[-1:]):
        g.history.append({"role": "user", "content": user_msg})
    origin_coords = None
    here = {"here", "my location", "current location", "my current location", "where i am", "me", "my place"}
    there = g.chat_context.get("station_name") or g.chat_context.get("trip_destination")
    if origin and (origin.lower() in here or origin.lower() == "location" or arrivals.is_here(origin)):
        origin = "USE_CURRENT_LOCATION"
    elif origin and origin.lower() in ("there", "that station", "that stop") and there:
        origin = there
    if destination and destination.lower() in ("there", "that station", "that stop") and g.chat_context.get("station_name"):
        destination = g.chat_context["station_name"]
    if not origin or origin == "USE_CURRENT_LOCATION":
        if g.location.get("station"):
            origin = g.location["station"]
        elif g.location.get("lat") and g.location.get("lon"):
            origin, origin_coords = "your location", f"{g.location['lat']},{g.location['lon']}"
        else:
            origin = None
    if not origin:
        result = {"code": "need_origin", "destination": destination, "destination_name": destination, "routes": []}
        reply = trips.build_trip_reply(result)
        g.history.append({"role": "assistant", "content": reply})
        return {"reply": reply, "trip": result, "request_location": True,
                "context": {"trip_destination": destination}}
    result = plan_trip_core(origin, destination, origin_coords=origin_coords,
                            when=when["time"] if when else None, arrive_by=bool(when and when["arrive_by"]))
    if when:
        result["when_label"] = when["label"]
        result["when_time"] = when["time"].isoformat()
        result["arrive_by"] = bool(when["arrive_by"] and not result.get("missed_deadline"))
    result["routes"] = trips.sort_routes(result.get("routes") or [], arrive_by=bool(result.get("arrive_by")))
    reply = trips.build_trip_reply(result, get_now().strftime("%-I:%M %p"))
    g.history.append({"role": "assistant", "content": reply})
    context = {"trip_origin": result.get("origin_name") if origin_coords is None else None,
               "trip_destination": result.get("destination_name") or destination}
    if result["code"] == "unresolved" and "origin" in (result.get("unresolved") or []):
        context["trip_origin"] = None          # ask again for the start, keep the destination
    return {"reply": reply, "trip": result, "context": {k: v for k, v in context.items() if v}}


def process_chat_message(user_msg):
    """Main chat processing function"""

    # Trip questions use the Trip Planner's own routing; arrival questions use live
    # predictions. Neither lets the model invent times or steps.
    is_place = lambda s: find_station_data(s) is not None or s.lower() in LANDMARKS
    trip_q = trips.parse_trip_question(user_msg, is_place=is_place, now=get_now()) \
        or trips.parse_trip_followup(user_msg, g.chat_context, is_place=is_place)
    if trip_q:
        return trip_response(user_msg, trip_q["origin"], trip_q["destination"], trip_q.get("when"))

    parsed = arrivals.parse_arrival_question(user_msg, STATION_DATA, STATION_ALIASES, g.chat_context)
    if parsed:
        return arrivals_response(user_msg, parsed)

    intent = get_smart_intent(user_msg)
    
    if not intent:
        intent = {"intent": "general"}
    
    intent_type = intent.get('intent', 'general')
    
    data_context = {
        "intent_detected": intent_type,
        "user_query": user_msg,
        "user_location": g.location.get("station"),
        "data": {}
    }

    if intent_type in ['greeting', 'thanks', 'help', 'general']:
        if g.location.get("station"):
            data_context["data"]["user_near_station"] = g.location["station"]
            data_context["data"]["available_lines"] = g.location.get("lines", [])
    
    elif intent_type == 'alerts':
        route = intent.get('specific_route')
        route_id = resolve_route(route) if route else None
        alerts = get_alerts_for_route(route_id, limit=5)
        
        data_context["data"] = {
            "route_checked": route if route else "All System",
            "active_alerts": alerts if alerts else "None - Good Service"
        }
    
    elif intent_type == 'bus':
        bus_route = intent.get('bus_route')
        origin = intent.get('origin')
        dest = intent.get('destination')
        
        if origin == "USE_CURRENT_LOCATION" and g.location.get("station"):
            origin = g.location["station"]
            data_context["data"]["location_auto_detected"] = True
        
        if bus_route:
            predictions, stop_name = get_bus_predictions(bus_route, origin, dest)
            data_context["data"] = {
                "bus_route": bus_route,
                "stop_found": stop_name,
                "destination_filter": dest,
                "predictions": predictions
            }
        else:
            data_context["data"] = {"error": "Missing bus route number"}
    
    elif intent_type == 'commuter_rail':
        line = intent.get('commuter_line', '')
        origin = intent.get('origin')
        dest = intent.get('destination')
        
        if origin == "USE_CURRENT_LOCATION" and g.location.get("station"):
            origin = g.location["station"]
            data_context["data"]["location_auto_detected"] = True
            
        predictions, stop_name = get_commuter_rail_predictions(line, origin, dest)
        
        data_context["data"] = {
            "line": line,
            "stop_found": stop_name,
            "destination_filter": dest,
            "predictions": predictions
        }
    
    elif intent_type == 'trip_planning':
        dest = intent.get('destination')
        if dest:
            return trip_response(user_msg, intent.get('origin'), dest)
        data_context["data"] = {"error": "Missing Destination",
                                "message": "Where would you like to go? Tell me a station or address."}

    elif intent_type == 'prediction':
        origin = intent.get('origin')
        if origin == "USE_CURRENT_LOCATION" or (origin and arrivals.is_here(origin)):
            # Nearest stops to the rider (asks for their location when it isn't shared).
            return arrivals_response(user_msg, {
                "station_query": None, "near_me": True,
                "route": arrivals.parse_route(intent.get('specific_route') or user_msg) or arrivals.parse_route(user_msg),
                "toward": intent.get('destination'), "direction": None, "direction_word": None, "followup": False,
            })
        origin = origin or g.chat_context.get("station_name")
        if origin:
            # Same grounded path as the deterministic parser: live predictions, no invented times.
            return arrivals_response(user_msg, {
                "station_query": origin,
                "route": arrivals.parse_route(intent.get('specific_route') or user_msg) or arrivals.parse_route(user_msg),
                "toward": intent.get('destination'),
                "direction": None, "direction_word": None, "followup": False,
            })
        data_context["data"] = {"error": "Unknown Origin",
                                "message": "Which station are you asking about? For example: next Red Line train at Harvard."}

    final_reply = generate_natural_response(user_msg, data_context)
    g.history.append({"role": "assistant", "content": final_reply})
    
    return {"reply": final_reply}


# ==================== FLASK ROUTES ====================

# Cache-busting token for static assets: the deployed commit on Vercel, else start time.
ASSET_VERSION = (os.environ.get('VERCEL_GIT_COMMIT_SHA') or str(int(datetime.now().timestamp())))[:10]


# Every ES module gets the version in its URL via an import map, so a deploy can never
# be served stale. (Vercel normalizes file mtimes, so Flask's mtime+size ETags could
# match across deploys and return 304 for changed files of the same size.)
_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
IMPORT_MAP = {"imports": {
    f"/static/{os.path.relpath(os.path.join(root, f), _STATIC_DIR)}": f"/static/{os.path.relpath(os.path.join(root, f), _STATIC_DIR)}?v={ASSET_VERSION}"
    for root, _, files in os.walk(os.path.join(_STATIC_DIR, 'js')) for f in files if f.endswith('.js')
}}


@app.after_request
def static_cache_headers(response):
    if request.path.startswith('/static/'):
        if request.args.get('v') == ASSET_VERSION:
            response.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
        else:
            response.headers['Cache-Control'] = 'no-store'
            response.headers.pop('ETag', None)
            response.headers.pop('Last-Modified', None)
    return response


@app.route('/')
def home():
    return render_template('index.html', asset_version=ASSET_VERSION, import_map=IMPORT_MAP)


@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "service": "charlie-mbta-chatbot",
        "version": "2.1.0",
        "features": ["chatbot", "location", "map", "bold-timing"]
    })


# ==================== LOCATION API ENDPOINTS ====================

@app.route('/api/set-location', methods=['POST'])
def set_location():
    
    data = request.get_json()
    lat = data.get('lat')
    lon = data.get('lon')
    
    if not lat or not lon:
        return jsonify({"error": "Missing lat/lon"}), 400
    
    nearest = find_nearest_station(lat, lon)
    
    if nearest:
        g.location = {
            "lat": lat,
            "lon": lon,
            "station": nearest["name"],
            "stop_id": nearest["id"],
            "distance": nearest["distance"],
            "lines": nearest["lines"]
        }
        return jsonify({
            "success": True,
            "nearest_station": nearest,
            "location": g.location,
            "message": f"📍 You're near {nearest['name']} ({nearest['distance']}m away)"
        })
    else:
        g.location = {"lat": lat, "lon": lon, "station": None, "stop_id": None, "lines": []}
        return jsonify({
            "success": False,
            "location": g.location,
            "message": "No MBTA stations found within 2km of your location."
        })


@app.route('/api/get-location')
def get_location():
    return jsonify(g.location)


@app.route('/api/stations')
def get_all_stations():
    stations_list = []
    seen_ids = set()
    
    for name, data in STATION_DATA.items():
        if data["id"] in seen_ids:
            continue
        seen_ids.add(data["id"])
        
        stations_list.append({
            "name": name,
            "id": data["id"],
            "lat": data["lat"],
            "lon": data["lon"],
            "lines": data["lines"]
        })
    
    return cached(jsonify(stations_list), 86400)


@app.route('/api/nearest-station')
def nearest_station():
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    
    if not lat or not lon:
        return jsonify({"error": "Missing lat/lon parameters"}), 400
    
    nearest = find_nearest_station(lat, lon)
    
    if nearest:
        return jsonify(nearest)
    else:
        return jsonify({"error": "No stations found nearby"}), 404


# ==================== EXISTING API ENDPOINTS ====================

@app.route('/api/chat', methods=['POST'])
def chat_post():
    data = request.get_json()
    user_msg = data.get('message', '').strip()
    if not user_msg:
        return jsonify({"reply": "👀 Type something!", "history": g.history})
    result = process_chat_message(user_msg)
    result["history"] = g.history[-MAX_HISTORY:]
    return jsonify(result)


@app.route('/chat')
def chat_get():
    user_msg = request.args.get('msg', '').strip()
    if not user_msg:
        return jsonify({"reply": "👀 Type something!"})
    return jsonify(process_chat_message(user_msg))


@app.route('/api/routes')
def get_routes():
    try:
        response = mbta_request(f"{BASE_URL}/routes", headers=HEADERS)
        data = response.json()
        routes_by_type = {"subway": [], "bus": [], "commuter_rail": [], "ferry": []}
        
        for route in data.get('data', []):
            route_id = route['id']
            attrs = route['attributes']
            route_type = attrs.get('type', 3)
            
            route_info = {
                "id": route_id,
                "name": attrs.get('long_name') or attrs.get('short_name', route_id),
                "short_name": attrs.get('short_name', ''),
                "color": f"#{attrs.get('color', '000000')}",
                "text_color": f"#{attrs.get('text_color', 'FFFFFF')}",
                "direction_names": attrs.get('direction_names', ['Outbound', 'Inbound']),
                "direction_destinations": attrs.get('direction_destinations', ['', ''])
            }
            
            if route_type in [0, 1]: routes_by_type["subway"].append(route_info)
            elif route_type == 2: routes_by_type["commuter_rail"].append(route_info)
            elif route_type == 3: routes_by_type["bus"].append(route_info)
            elif route_type == 4: routes_by_type["ferry"].append(route_info)
        
        return cached(jsonify(routes_by_type), 3600)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/alerts')
def get_alerts():
    try:
        route_filter = request.args.get('route')
        params = {"sort": "-severity,-updated_at", "page[limit]": 250}
        if route_filter:
            params["filter[route]"] = route_filter

        response = mbta_request(f"{BASE_URL}/alerts", headers=HEADERS, params=params, timeout=15)
        data = response.json()
        now = get_now()

        alerts = [transit.alert_payload(a, now) for a in data.get('data', [])]
        alerts.sort(key=transit.alert_priority)

        return cached(jsonify({"alerts": alerts, "count": len(alerts), "updated_at": now.isoformat()}), 20)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/predictions')
def get_predictions():
    try:
        stop_id = request.args.get('stop')
        route_id = request.args.get('route')
        direction_id = request.args.get('direction')
        
        if not stop_id and not route_id:
            return jsonify({"error": "Provide stop or route"}), 400
        
        params = {"include": "stop,route,trip,vehicle", "page[limit]": 40}
        if stop_id: params["filter[stop]"] = stop_id
        if route_id: params["filter[route]"] = route_id
        if direction_id: params["filter[direction_id]"] = direction_id
        
        response = mbta_request(f"{BASE_URL}/predictions", headers=HEADERS, params=params)
        data = response.json()
        
        included = {f"{item['type']}_{item['id']}": item for item in data.get('included', [])}
        
        predictions = []
        for pred in data.get('data', []):
            attrs = pred['attributes']
            rels = pred.get('relationships', {})
            
            route_data = rels.get('route', {}).get('data', {})
            stop_data = rels.get('stop', {}).get('data', {})
            trip_data = rels.get('trip', {}).get('data', {})
            
            route_info = included.get(f"route_{route_data.get('id', '')}", {})
            stop_info = included.get(f"stop_{stop_data.get('id', '')}", {})
            trip_info = included.get(f"trip_{trip_data.get('id', '')}", {})
            
            arrival_time = attrs.get('arrival_time') or attrs.get('departure_time')
            if not arrival_time: continue
            
            arrival_dt = parser.isoparse(arrival_time)
            route_attrs = route_info.get('attributes', {})
            
            predictions.append({
                "id": pred['id'],
                "arrival_time": arrival_time,
                "arrival_formatted": format_time(arrival_time),
                "time_away": format_time_diff(arrival_dt),
                "status": attrs.get('status', ''),
                "direction_id": attrs.get('direction_id'),
                "route": {
                    "id": route_data.get('id', ''),
                    "name": route_attrs.get('long_name') or route_attrs.get('short_name', ''),
                    "color": f"#{route_attrs.get('color', '000000')}"
                },
                "stop": {
                    "id": stop_data.get('id', ''),
                    "name": stop_info.get('attributes', {}).get('name', '')
                },
                "trip": {
                    "id": trip_data.get('id', ''),
                    "headsign": trip_info.get('attributes', {}).get('headsign', '')
                }
            })
        
        # Sort by the displayed time (arrival, or departure when there is no arrival)
        predictions.sort(key=lambda p: parser.isoparse(p["arrival_time"]))
        predictions = predictions[:20]
        return jsonify({"predictions": predictions, "count": len(predictions)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/vehicles')
def get_vehicles():
    try:
        route_id = request.args.get('route')
        limit = min(max(request.args.get('limit', 100, type=int), 1), 500)
        params = {"include": "stop,route,trip", "page[limit]": limit}
        if route_id: params["filter[route]"] = route_id
        
        response = mbta_request(f"{BASE_URL}/vehicles", headers=HEADERS, params=params)
        data = response.json()
        
        included = {f"{item['type']}_{item['id']}": item for item in data.get('included', [])}
        
        vehicles = []
        for vehicle in data.get('data', []):
            attrs = vehicle['attributes']
            rels = vehicle.get('relationships', {})
            
            route_data = rels.get('route', {}).get('data', {})
            route_info = included.get(f"route_{route_data.get('id', '')}", {})
            route_attrs = route_info.get('attributes', {})
            
            stop_rel = (rels.get('stop', {}) or {}).get('data') or {}
            trip_rel = (rels.get('trip', {}) or {}).get('data') or {}
            stop_attrs = included.get(f"stop_{stop_rel.get('id', '')}", {}).get('attributes', {})
            trip_attrs = included.get(f"trip_{trip_rel.get('id', '')}", {}).get('attributes', {})

            vehicles.append({
                "id": vehicle['id'],
                "label": attrs.get('label', ''),
                "latitude": attrs.get('latitude'),
                "longitude": attrs.get('longitude'),
                "bearing": attrs.get('bearing'),
                "current_status": attrs.get('current_status', ''),
                "direction_id": attrs.get('direction_id'),
                "updated_at": attrs.get('updated_at'),
                "occupancy_status": attrs.get('occupancy_status'),
                "headsign": trip_attrs.get('headsign', ''),
                "stop_name": stop_attrs.get('name', ''),
                "route": {
                    "id": route_data.get('id', ''),
                    "name": route_attrs.get('long_name') or route_attrs.get('short_name', ''),
                    "color": f"#{route_attrs.get('color', '000000')}"
                }
            })
        
        return cached(jsonify({"vehicles": vehicles, "count": len(vehicles), "updated_at": get_now().isoformat()}), 5)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/stops')
def get_stops():
    try:
        route_id = request.args.get('route')
        params = {"page[limit]": 100}
        if route_id: params["filter[route]"] = route_id
        
        response = mbta_request(f"{BASE_URL}/stops", headers=HEADERS, params=params)
        data = response.json()
        
        stops = []
        for stop in data.get('data', []):
            attrs = stop['attributes']
            stops.append({
                "id": stop['id'],
                "name": attrs.get('name', ''),
                "latitude": attrs.get('latitude'),
                "longitude": attrs.get('longitude'),
                "municipality": attrs.get('municipality', ''),
                "wheelchair_boarding": attrs.get('wheelchair_boarding')
            })
        
        return jsonify({"stops": stops, "count": len(stops)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/line/<line_id>')
def get_line_status(line_id):
    try:
        now = get_now()
        pred_response = mbta_request(f"{BASE_URL}/predictions", headers=HEADERS, timeout=15, params={
            "filter[route]": line_id, "include": "stop,trip", "page[limit]": 400})
        pred_data = pred_response.json()

        vehicle_response = mbta_request(f"{BASE_URL}/vehicles", headers=HEADERS, timeout=15,
            params={"filter[route]": line_id, "include": "stop,trip"})
        vehicle_data = vehicle_response.json()

        alert_response = mbta_request(f"{BASE_URL}/alerts", headers=HEADERS, timeout=15,
            params={"filter[route]": line_id, "sort": "-severity"})
        alert_data = alert_response.json()

        stops_response = mbta_request(f"{BASE_URL}/stops", headers=HEADERS, timeout=15,
            params={"filter[route]": line_id, "fields[stop]": "name,wheelchair_boarding"})
        stops_data = stops_response.json()

        route_response = mbta_request(f"{BASE_URL}/routes/{line_id}", headers=HEADERS, timeout=15)
        route_attrs = route_response.json().get('data', {}).get('attributes', {})

        included = {f"{item['type']}_{item['id']}": item for item in pred_data.get('included', [])}

        def parent_of(stop_id):
            stop = included.get(f"stop_{stop_id}", {})
            parent = stop.get('relationships', {}).get('parent_station', {}).get('data')
            return parent['id'] if parent else stop_id

        predictions_by_stop = {}
        by_station = {}
        for pred in pred_data.get('data', []):
            attrs = pred['attributes']
            if transit.is_cancelled(attrs):
                continue
            rels = pred.get('relationships', {})
            stop_id = rels.get('stop', {}).get('data', {}).get('id', '')
            trip_id = rels.get('trip', {}).get('data', {}).get('id', '')
            stop_info = included.get(f"stop_{stop_id}", {})
            trip_info = included.get(f"trip_{trip_id}", {})

            event_time = transit.effective_time(attrs)
            event_dt = transit.parse_time(event_time)
            if not event_dt or transit.is_departed(event_dt, now):
                continue

            entry = {
                "time": event_time,
                "time_formatted": format_time(event_time),
                "time_away": format_time_diff(event_dt),
                "headsign": trip_info.get('attributes', {}).get('headsign', ''),
                "direction_id": attrs.get('direction_id'),
                "status": attrs.get('status') or "",
                "event": transit.event_kind(attrs),
            }
            stop_name = stop_info.get('attributes', {}).get('name', stop_id)
            predictions_by_stop.setdefault(stop_id, {"stop_id": stop_id, "stop_name": stop_name, "predictions": []})
            predictions_by_stop[stop_id]["predictions"].append(entry)
            by_station.setdefault(parent_of(stop_id), []).append(entry)

        for stop in predictions_by_stop.values():
            stop["predictions"] = transit.sort_by_time(stop["predictions"])

        # Station sequence (MBTA returns route stops in route order) with the next
        # departures in each direction.
        stations = []
        for stop in stops_data.get('data', []):
            preds = transit.sort_by_time(by_station.get(stop['id'], []))
            next_by_direction = {}
            for p in preds:
                next_by_direction.setdefault(str(p["direction_id"]), [])
                if len(next_by_direction[str(p["direction_id"])]) < 2:
                    next_by_direction[str(p["direction_id"])].append(p)
            stations.append({
                "id": stop['id'],
                "name": stop['attributes'].get('name', ''),
                "wheelchair_boarding": stop['attributes'].get('wheelchair_boarding'),
                "next": next_by_direction,
            })

        v_included = {f"{item['type']}_{item['id']}": item for item in vehicle_data.get('included', [])}
        vehicles = []
        for v in vehicle_data.get('data', []):
            a = v['attributes']
            rels = v.get('relationships', {})
            stop_rel = (rels.get('stop', {}) or {}).get('data') or {}
            trip_rel = (rels.get('trip', {}) or {}).get('data') or {}
            vehicles.append({
                "id": v['id'], "label": a.get('label', ''),
                "latitude": a.get('latitude'), "longitude": a.get('longitude'),
                "bearing": a.get('bearing'), "direction_id": a.get('direction_id'),
                "current_status": a.get('current_status', ''), "updated_at": a.get('updated_at'),
                "stop_name": v_included.get(f"stop_{stop_rel.get('id', '')}", {}).get('attributes', {}).get('name', ''),
                "headsign": v_included.get(f"trip_{trip_rel.get('id', '')}", {}).get('attributes', {}).get('headsign', ''),
            })

        alerts = [transit.alert_payload(a, now) for a in alert_data.get('data', [])]
        alerts.sort(key=transit.alert_priority)

        return cached(jsonify({
            "line": {
                "id": line_id,
                "name": route_attrs.get('long_name') or route_attrs.get('short_name', line_id),
                "short_name": route_attrs.get('short_name', ''),
                "color": f"#{route_attrs.get('color', '000000')}",
                "type": route_attrs.get('type'),
                "direction_names": route_attrs.get('direction_names', []),
                "direction_destinations": route_attrs.get('direction_destinations', []),
            },
            "predictions_by_stop": list(predictions_by_stop.values()),
            "stations": stations,
            "vehicles": vehicles,
            "alerts": alerts,
            "updated_at": now.isoformat(),
        }), 10)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/dashboard')
def get_dashboard():
    try:
        now = get_now()
        alert_response = mbta_request(f"{BASE_URL}/alerts", headers=HEADERS, timeout=15,
            params={"sort": "-severity", "page[limit]": 250})
        alert_data = alert_response.json()
        alerts = [transit.alert_payload(a, now) for a in alert_data.get('data', [])]

        # Distinct alerts per route (an alert lists a route once per affected stop).
        alerts_by_route = {}
        for alert in alerts:
            for route in alert["affected_routes"]:
                alerts_by_route.setdefault(route, {})[alert["id"]] = alert

        # One request for every subway line's vehicles, counted per route.
        vehicle_response = mbta_request(f"{BASE_URL}/vehicles", headers=HEADERS, timeout=15,
            params={"filter[route]": ",".join(SUBWAY_ROUTES)})
        vehicle_counts = {}
        for v in vehicle_response.json().get('data', []):
            route = (v.get('relationships', {}).get('route', {}).get('data') or {}).get('id')
            if route:
                vehicle_counts[route] = vehicle_counts.get(route, 0) + 1

        subway_status = {}
        for line in SUBWAY_ROUTES:
            line_alerts = list(alerts_by_route.get(line, {}).values())
            status, active_service = transit.line_status(line_alerts)
            subway_status[line] = {
                "name": line,
                "color": LINE_COLORS.get(line, "#000000"),
                "vehicle_count": vehicle_counts.get(line, 0),
                "alert_count": len(line_alerts),
                "service_alert_count": len(active_service),
                "accessibility_alert_count": sum(1 for a in line_alerts if a["category"] == "accessibility"),
                "status": status,
                "headline": (sorted(active_service, key=transit.alert_priority)[0]["service_effect"]
                             if active_service else ""),
            }

        severe_alerts = [
            {k: a[k] for k in ("id", "effect", "severity", "header", "short_header")}
            for a in sorted(alerts, key=transit.alert_priority)
            if a["severity"] >= 7 and a["active_now"] and a["category"] == "service"
        ]

        return cached(jsonify({
            "subway_status": subway_status,
            "severe_alerts": severe_alerts[:10],
            "total_alerts": len(alerts),
            "updated_at": now.isoformat(),
            "user_location": g.location
        }), 15)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def station_arrivals(stop_id, now):
    """Upcoming, time-sorted predictions at a station. Shared by the station board and
    Charlie AI, so both answer from the same MBTA snapshot (cached for a few seconds).
    Raises MBTAError when the MBTA can't be reached."""
    pred_response = mbta_request(f"{BASE_URL}/predictions", headers=HEADERS, timeout=15,
        params={"filter[stop]": stop_id, "include": "route,trip,stop", "page[limit]": 60})
    return prediction_rows(pred_response.json(), now)


def prediction_rows(pred_data, now):
    """Upcoming, time-sorted rows from a /predictions response (route, trip and stop included)."""
    included = {f"{item['type']}_{item['id']}": item for item in pred_data.get('included', [])}

    predictions = []
    for pred in pred_data.get('data', []):
        attrs = pred['attributes']
        if transit.is_cancelled(attrs):
            continue
        rels = pred.get('relationships', {})
        route_data = rels.get('route', {}).get('data', {}) or {}
        trip_data = rels.get('trip', {}).get('data', {}) or {}
        child_stop = rels.get('stop', {}).get('data', {}) or {}
        route_info = included.get(f"route_{route_data.get('id', '')}", {})
        trip_info = included.get(f"trip_{trip_data.get('id', '')}", {})
        stop_info = included.get(f"stop_{child_stop.get('id', '')}", {}).get('attributes', {})

        # Display and sort by the same time: arrival, else departure (terminals).
        arrival_time = transit.effective_time(attrs)
        arrival_dt = transit.parse_time(arrival_time)
        if not arrival_dt or transit.is_departed(arrival_dt, now):
            continue
        route_attrs = route_info.get('attributes', {})

        stop_rels = included.get(f"stop_{child_stop.get('id', '')}", {}).get('relationships', {})
        predictions.append({
            "stop_id": child_stop.get('id', ''),
            "parent_id": ((stop_rels.get('parent_station') or {}).get('data') or {}).get('id', ''),
            "stop_name": stop_info.get('name') or "",
            "route_id": route_data.get('id', ''),
            "route_name": route_attrs.get('long_name') or route_attrs.get('short_name', ''),
            "route_short_name": route_attrs.get('short_name', ''),
            "route_color": f"#{route_attrs.get('color', '000000')}",
            "route_type": route_attrs.get('type'),
            "headsign": trip_info.get('attributes', {}).get('headsign', ''),
            "direction_id": attrs.get('direction_id'),
            "platform_name": stop_info.get('platform_name') or "",
            "status": attrs.get('status') or "",
            "event": transit.event_kind(attrs),
            "time": arrival_time,
            "arrival_formatted": format_time(arrival_time),
            "time_away": format_time_diff(arrival_dt),
        })

    # Sort by the time we display. MBTA's arrival_time sort misplaces trains that
    # only have a departure_time (e.g. trains leaving a terminal like Oak Grove).
    return transit.sort_by_time(predictions)


@app.route('/api/station/<stop_id>')
def get_station_info(stop_id):
    try:
        now = get_now()
        stop_response = mbta_request(f"{BASE_URL}/stops/{stop_id}", headers=HEADERS, timeout=15)
        stop_attrs = stop_response.json().get('data', {}).get('attributes', {})
        predictions = station_arrivals(stop_id, now)[:30]
        alert_response = mbta_request(f"{BASE_URL}/alerts", headers=HEADERS, timeout=15,
            params={"filter[stop]": stop_id})
        alert_data = alert_response.json()

        alerts = [transit.alert_payload(a, now) for a in alert_data.get('data', [])]
        alerts = [a for a in alerts if a["active_now"]]
        alerts.sort(key=transit.alert_priority)

        station_meta = next((d for d in STATION_DATA.values() if d.get("id") == stop_id), {})

        return cached(jsonify({
            "station": {
                "id": stop_id,
                "name": stop_attrs.get('name', ''),
                "latitude": stop_attrs.get('latitude'),
                "longitude": stop_attrs.get('longitude'),
                "wheelchair_boarding": stop_attrs.get('wheelchair_boarding'),
                "municipality": stop_attrs.get('municipality', ''),
                "lines": station_meta.get("lines", []),
            },
            "predictions": predictions,
            "alerts": alerts,
            "updated_at": now.isoformat(),
        }), 10)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/shapes')
def get_shapes():
    """Encoded polylines for route paths (canonical shapes when the MBTA marks them)."""
    try:
        route_ids = request.args.get('route', ",".join(SUBWAY_ROUTES))
        # Shapes don't name their route, so fetch per-route lists of shape ids.
        shapes = []
        for route_id in route_ids.split(','):
            r = mbta_request(f"{BASE_URL}/shapes", headers=HEADERS, timeout=20,
                params={"filter[route]": route_id, "fields[shape]": "polyline"})
            items = r.json().get('data', [])
            canonical = [s for s in items if s['id'].startswith('canonical')]
            for s in (canonical or items):
                shapes.append({"route_id": route_id, "id": s['id'], "polyline": s['attributes']['polyline']})

        return cached(jsonify({"shapes": shapes, "count": len(shapes)}), 86400)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/mode-shapes')
def get_mode_shapes():
    """Typical route paths for a whole mode (bus or commuter_rail) in one response,
    labelled with route number/name and color, for the map's bus and Commuter Rail layers."""
    try:
        mode = request.args.get('mode', 'bus')
        route_type = {"bus": 3, "commuter_rail": 2}.get(mode)
        if route_type is None:
            return jsonify({"error": "mode must be bus or commuter_rail"}), 400

        routes = mbta_request(f"{BASE_URL}/routes", params={"filter[type]": route_type}).json().get('data', [])
        meta = {r['id']: r['attributes'] for r in routes}
        if not meta:
            return jsonify({"shapes": [], "count": 0})

        patterns = mbta_request(f"{BASE_URL}/route_patterns", timeout=30, params={
            "filter[route]": ",".join(meta), "include": "representative_trip.shape",
            "fields[route_pattern]": "direction_id,typicality,canonical,name",
            "fields[shape]": "polyline",
        }).json()
        included = {f"{i['type']}_{i['id']}": i for i in patterns.get('included', [])}

        shapes, seen = [], set()
        for pat in patterns.get('data', []):
            attrs = pat['attributes']
            if attrs.get('typicality') != 1 and not attrs.get('canonical'):
                continue
            route_id = (pat['relationships'].get('route', {}).get('data') or {}).get('id')
            trip_id = (pat['relationships'].get('representative_trip', {}).get('data') or {}).get('id')
            trip = included.get(f"trip_{trip_id}", {})
            shape_id = ((trip.get('relationships', {}).get('shape', {}) or {}).get('data') or {}).get('id')
            shape = included.get(f"shape_{shape_id}")
            if not route_id or route_id not in meta or not shape or shape_id in seen:
                continue
            seen.add(shape_id)
            r = meta[route_id]
            shapes.append({
                "route_id": route_id,
                "short_name": r.get('short_name') or "",
                "long_name": r.get('long_name') or "",
                "color": f"#{r.get('color') or '000000'}",
                "direction_id": attrs.get('direction_id'),
                "headsign": (trip.get('attributes') or {}).get('headsign', ''),
                "polyline": shape['attributes']['polyline'],
            })

        return cached(jsonify({"mode": mode, "shapes": shapes, "count": len(shapes)}), 86400)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== GOOGLE MAPS TRIP PLANNER API ====================
# Uses Google Maps Directions API for accurate multi-modal routing

GOOGLE_MAPS_API_KEY = os.environ.get('GOOGLE_MAPS_API_KEY')


def plan_trip_core(origin, destination, origin_coords=None, dest_coords=None, origin_id=None, dest_id=None,
                   when=None, arrive_by=False):
    """Plan a trip. Shared by the Trip Planner endpoint and Charlie AI so both give
    identical itineraries. Google Directions when a key is configured, else the free
    Transitous (MOTIS) router. Each place is resolved first, so the result says which
    field couldn't be found, versus no itineraries, versus a routing failure.

    Returns {"code": "ok"|"no_itineraries"|"unresolved"|"routing_unavailable", ...}."""
    origin_station = station_by_id(origin_id) or find_station_data(origin)
    dest_station = station_by_id(dest_id) or find_station_data(destination)
    origin_mark, dest_mark = landmark(origin), landmark(destination)
    origin_coords = origin_coords or (f"{origin_station['lat']},{origin_station['lon']}" if origin_station
                                      else origin_mark and origin_mark["coords"])
    dest_coords = dest_coords or (f"{dest_station['lat']},{dest_station['lon']}" if dest_station
                                  else dest_mark and dest_mark["coords"])
    base = {"origin": origin, "destination": destination,
            "origin_name": origin_station["name"] if origin_station else (origin_mark["name"] if origin_mark else origin),
            "destination_name": dest_station["name"] if dest_station else (dest_mark["name"] if dest_mark else destination),
            "routes": [], "notices": []}

    # Google Directions only plans "now"; a requested time goes to the free router.
    routes = [] if when else get_google_directions(origin_coords or f"{origin}, Boston, MA",
                                                   dest_coords or f"{destination}, Boston, MA", origin, destination)
    if not routes:
        try:
            from_place = origin_coords or _resolve_place(origin)
            to_place = dest_coords or _resolve_place(destination)
        except RoutingError as e:
            return {**base, "code": "routing_unavailable", "error": str(e)}
        unresolved = [f for f, p in (("origin", from_place), ("destination", to_place)) if not p]
        if unresolved:
            return {**base, "code": "unresolved", "unresolved": unresolved,
                    "message": "We couldn’t find that place in the Boston area."}
        try:
            routes = get_transitous_directions(from_place, to_place, base["destination_name"], when=when, arrive_by=arrive_by)
            # Never offer a trip that has already left.
            cutoff = get_now().timestamp() - 60
            routes = [r for r in routes if not r.get("depart_ts") or r["depart_ts"] >= cutoff]
            if arrive_by and not routes:
                # Nothing leaving now makes the deadline: say so and show the soonest arrivals.
                base["missed_deadline"] = True
                routes = get_transitous_directions(from_place, to_place, base["destination_name"])
        except RoutingError as e:
            return {**base, "code": "routing_unavailable", "error": str(e)}

    notices = []
    try:
        notices = annotate_trip_alerts(routes, origin_station, dest_station)
    except Exception as e:
        print(f"Trip alert matching skipped: {e}")
    for r in routes:
        r.pop("_legs", None)
    return {**base, "routes": routes, "notices": notices, "code": "ok" if routes else "no_itineraries"}


@app.route('/api/trip-planner', methods=['POST'])
def plan_trip():
    data = request.get_json(silent=True) or {}
    origin = (data.get('origin') or '').strip()
    destination = (data.get('destination') or '').strip()
    if not origin or not destination:
        return jsonify({"error": "Origin and destination are required"}), 400
    # Optional "Depart at" / "Arrive by" time: local Boston time ("2026-11-16T18:10") or ISO with offset.
    when, arrive_by = None, bool(data.get('arrive_by'))
    if data.get('when'):
        try:
            when = parser.isoparse(str(data['when']))
        except (ValueError, TypeError):
            return jsonify({"error": "That date or time isn’t valid.", "code": "bad_time"}), 400
        when = BOSTON_TZ.localize(when) if when.tzinfo is None else when.astimezone(BOSTON_TZ)
        when = when.replace(second=0, microsecond=0)
        now = get_now()
        if when < now - timedelta(minutes=2):
            return jsonify({"error": "That time has already passed. Pick a time from now on.", "code": "past_time"}), 400
        if when > now + timedelta(days=60):
            return jsonify({"error": "Pick a time within the next 60 days.", "code": "bad_time"}), 400
    result = plan_trip_core(origin, destination, data.get('origin_coords'), data.get('dest_coords'),
                            data.get('origin_id'), data.get('dest_id'), when=when, arrive_by=arrive_by and when is not None)
    if result["code"] == "routing_unavailable":
        return jsonify({"error": result["error"], "code": result["code"]}), 502
    if when:
        result["when"] = {"time": when.isoformat(), "arrive_by": arrive_by,
                          "label": trips.when_label(when, arrive_by, get_now())}
    return jsonify(result)


def station_by_id(stop_id):
    if not stop_id:
        return None
    for name, d in STATION_DATA.items():
        if d["id"] == stop_id:
            return {"name": name, **d}
    return None


def annotate_trip_alerts(routes, origin_station, dest_station):
    """Attach matched MBTA alerts to each itinerary and return trip-level notices for
    closures at the chosen origin or destination station."""
    legs = [leg for r in routes for leg in r.get("_legs", [])]
    route_ids = sorted({leg["route_id"] for leg in legs if leg.get("route_id")})
    station_ids = [s["id"] for s in (origin_station, dest_station) if s]
    raw = []
    if route_ids:
        raw += mbta_request(f"{BASE_URL}/alerts", params={"filter[route]": ",".join(route_ids)}).json().get('data', [])
    if station_ids:
        raw += mbta_request(f"{BASE_URL}/alerts", params={"filter[stop]": ",".join(station_ids)}).json().get('data', [])
    alerts = list({a["id"]: a for a in raw}.values())

    for r in routes:
        found = {}
        for leg in r.get("_legs", []):
            for m in transit.match_leg_alerts(alerts, leg):
                if m["id"] not in found:
                    found[m["id"]] = {**m, "step": leg["name"], "route_id": leg["route_id"],
                                      "stop_names": [leg["stop_names"].get(s, s) for s in m["stops"]]}
        r["alerts"] = list(found.values())

    now = get_now()
    starts = [transit.parse_time(r.get("depart_iso")) for r in routes if r.get("depart_iso")]
    ends = [transit.parse_time(r.get("arrive_iso")) for r in routes if r.get("arrive_iso")]
    window = (min(starts) if starts else now, max(ends) if ends else now + timedelta(hours=2))

    notices = []
    for role, station in (("origin", origin_station), ("destination", dest_station)):
        if not station:
            continue
        closures = transit.station_closures(alerts, station["id"], *window)
        if not closures:
            continue
        name = station["name"]
        used = [s for r in routes for s in ([r.get("first_board")] if role == "origin" else [r.get("last_alight")]) if s]
        via = sorted({s for s in used if s.lower() != name.lower()})
        serves = any(s.lower() == name.lower() for s in used)
        c = closures[0]
        if routes and via and not serves:
            how = (f"These options {'start from' if role == 'origin' else 'reach the area via'} "
                   f"{' or '.join(via[:3])} {'after' if role == 'origin' else 'and'} a short walk.")
        else:
            how = ""
        notices.append({
            "role": role, "station": name, "station_id": station["id"], "alert_id": c["id"],
            "effect": c["effect"], "severity": c["severity"], "header": c["header"], "routes": c["routes"],
            "direction_specific": bool(c["direction_ids"]), "via": via, "explanation": how,
        })
    return notices


def get_google_directions(origin, destination, origin_name, dest_name):
    """
    Get transit directions from Google Maps Directions API.
    Returns multiple route alternatives with all transit modes.
    """
    if not GOOGLE_MAPS_API_KEY:
        return []
    try:
        url = "https://maps.googleapis.com/maps/api/directions/json"
        params = {
            "origin": origin,
            "destination": destination,
            "mode": "transit",
            "alternatives": "true",
            "transit_mode": "bus|subway|train|tram|rail",
            "key": GOOGLE_MAPS_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        
        if data.get('status') != 'OK':
            print(f"Google Maps API error: {data.get('status')} - {data.get('error_message', '')}")
            return []
        
        routes = []
        
        for route in data.get('routes', []):
            leg = route.get('legs', [{}])[0]
            
            duration_text = leg.get('duration', {}).get('text', '')
            duration_value = leg.get('duration', {}).get('value', 0) // 60  # Convert to minutes
            
            depart_time = leg.get('departure_time', {}).get('text', '')
            arrive_time = leg.get('arrival_time', {}).get('text', '')
            
            # Process steps
            steps = []
            detailed_steps = []
            transfers = -1  # Will count transit steps - 1
            walk_time = 0
            fare = None
            
            for step in leg.get('steps', []):
                travel_mode = step.get('travel_mode', '')
                duration = step.get('duration', {}).get('text', '')
                instruction = step.get('html_instructions', '').replace('<b>', '').replace('</b>', '').replace('<div style="font-size:0.9em">', ' - ').replace('</div>', '')
                
                if travel_mode == 'WALKING':
                    walk_duration = step.get('duration', {}).get('value', 0) // 60
                    walk_time += walk_duration
                    
                    detailed_steps.append({
                        "type": "walk",
                        "instruction": f"Walk {step.get('distance', {}).get('text', '')}",
                        "details": instruction,
                        "duration": duration
                    })
                    
                elif travel_mode == 'TRANSIT':
                    transfers += 1
                    transit_details = step.get('transit_details', {})
                    line = transit_details.get('line', {})
                    
                    # Get transit info
                    vehicle_type = line.get('vehicle', {}).get('type', 'BUS')
                    line_name = line.get('short_name') or line.get('name', '')
                    line_color = line.get('color', '#666666')
                    headsign = transit_details.get('headsign', '')
                    
                    departure_stop = transit_details.get('departure_stop', {}).get('name', '')
                    arrival_stop = transit_details.get('arrival_stop', {}).get('name', '')
                    departure_time_step = transit_details.get('departure_time', {}).get('text', '')
                    arrival_time_step = transit_details.get('arrival_time', {}).get('text', '')
                    num_stops = transit_details.get('num_stops', 0)
                    
                    # Determine step type
                    step_type = get_transit_type(vehicle_type, line_name)
                    
                    # Add to visual steps (route summary)
                    steps.append({
                        "type": step_type,
                        "name": get_display_name(line_name, vehicle_type),
                        "short_name": line_name or vehicle_type[:3],
                        "color": line_color,
                        "vehicle_type": vehicle_type
                    })
                    
                    # Add detailed steps
                    detailed_steps.append({
                        "type": "transit",
                        "instruction": f"Take {get_display_name(line_name, vehicle_type)}",
                        "details": f"Board at {departure_stop}" + (f" toward {headsign}" if headsign else ""),
                        "time": departure_time_step,
                        "duration": duration,
                        "color": line_color,
                        "num_stops": num_stops
                    })
                    
                    detailed_steps.append({
                        "type": "transit",
                        "instruction": f"Get off at {arrival_stop}",
                        "details": f"{num_stops} stop{'s' if num_stops != 1 else ''}" if num_stops else "Arrive",
                        "time": arrival_time_step,
                        "color": line_color
                    })
            
            # Get fare if available
            if 'fare' in route:
                fare = route['fare'].get('text', '')
            elif 'fare' in leg:
                fare = leg['fare'].get('text', '')
            
            # Build route object
            route_obj = {
                "duration": duration_value,
                "duration_text": duration_text,
                "transfers": max(0, transfers),
                "depart_time": depart_time,
                "arrive_time": arrive_time,
                "depart_ts": leg.get('departure_time', {}).get('value'),
                "arrive_ts": leg.get('arrival_time', {}).get('value'),
                "time_range": f"{depart_time}—{arrive_time}" if depart_time and arrive_time else duration_text,
                "walk_time": walk_time,
                "fare": fare or estimate_fare(steps),
                "steps": steps,
                "detailed_steps": detailed_steps,
                "warnings": route.get('warnings', []),
                "summary": route.get('summary', '')
            }
            
            routes.append(route_obj)
        
        # Sort by duration
        routes.sort(key=lambda x: x['duration'])
        
        return routes[:6]  # Return top 6 routes
        
    except Exception as e:
        print(f"Google Directions API error: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_transit_type(vehicle_type, line_name):
    """Convert Google's vehicle type to our display type"""
    vehicle_type = vehicle_type.upper()
    line_name = (line_name or '').upper()
    
    # Check for specific MBTA lines
    if 'RED' in line_name or 'ORANGE' in line_name or 'BLUE' in line_name or 'GREEN' in line_name:
        return 'subway'
    
    if 'SL' in line_name or 'SILVER' in line_name:
        return 'silver-line'
    
    if vehicle_type in ['SUBWAY', 'METRO_RAIL', 'HEAVY_RAIL']:
        return 'subway'
    elif vehicle_type in ['TRAM', 'LIGHT_RAIL']:
        return 'tram'
    elif vehicle_type in ['COMMUTER_TRAIN', 'RAIL', 'LONG_DISTANCE_TRAIN']:
        return 'commuter-rail'
    elif vehicle_type == 'BUS':
        return 'bus'
    elif vehicle_type == 'FERRY':
        return 'ferry'
    else:
        return 'transit'


def get_display_name(line_name, vehicle_type):
    """Get a nice display name for the transit line"""
    if not line_name:
        return vehicle_type.replace('_', ' ').title()
    
    # Handle MBTA specific names
    line_upper = line_name.upper()
    
    if line_upper in ['RED', 'ORANGE', 'BLUE']:
        return f"{line_name} Line"
    elif 'GREEN' in line_upper:
        return f"Green Line {line_name.replace('Green-', '').replace('Green ', '')}" if '-' in line_name or ' ' in line_name else "Green Line"
    elif line_upper.startswith('SL') or 'SILVER' in line_upper:
        return f"Silver Line {line_name.replace('SL', '').replace('Silver', '').strip()}"
    elif line_upper.startswith('CR-'):
        return f"{line_name.replace('CR-', '')} Line (CR)"
    elif vehicle_type in ['BUS']:
        return f"Bus {line_name}"
    elif vehicle_type in ['COMMUTER_TRAIN', 'RAIL']:
        return f"{line_name} Line"
    else:
        return line_name


def estimate_fare(steps):
    """Estimate fare based on transit types used"""
    if not steps:
        return "$2.40"
    
    has_subway = any(s.get('type') == 'subway' for s in steps)
    has_bus = any(s.get('type') == 'bus' for s in steps)
    has_cr = any(s.get('type') == 'commuter-rail' for s in steps)
    has_ferry = any(s.get('type') == 'ferry' for s in steps)
    
    if has_cr:
        return "$6.50+"  # CR fares vary by zone
    if has_ferry:
        return "$3.70"
    if has_subway and has_bus:
        return "$2.40"  # Free transfer from subway to bus
    if has_subway:
        return "$2.40"
    if has_bus:
        return "$1.70"
    
    return "$2.40"


TRANSITOUS_URL = "https://api.transitous.org/api"
TRANSITOUS_HEADERS = {"User-Agent": "charlie-mbta-trip-planner"}


class RoutingError(RuntimeError):
    pass


SERVICE_RADIUS_KM = 80  # MBTA service area, measured from downtown Boston


def _resolve_place(place):
    """'lat,lon' for a coordinate string, else geocode free text in the Boston area.

    Returns None when nothing in the service area matches (e.g. "zzzzz"); raises
    RoutingError when the geocoder itself can't be reached."""
    parts = [p.strip() for p in str(place).split(',')]
    if len(parts) == 2:
        try:
            float(parts[0]); float(parts[1])
            return f"{parts[0]},{parts[1]}"
        except ValueError:
            pass
    text = str(place).replace(', Boston, MA', '').strip()
    try:
        # No "Boston" suffix: it made nonsense like "zzzzz" match "New Boston, NH".
        resp = requests.get(f"{TRANSITOUS_URL}/v1/geocode",
                            params={"text": text, "place": "42.3601,-71.0589"},
                            headers=TRANSITOUS_HEADERS, timeout=15)
    except requests.RequestException as e:
        raise RoutingError("The trip planning service didn’t respond. Try again in a moment.") from e
    if resp.status_code != 200:
        raise RoutingError(f"The trip planning service is unavailable right now (HTTP {resp.status_code}).")
    try:
        results = resp.json() or []
    except ValueError:
        raise RoutingError("The trip planning service sent an unreadable response.")
    for r in results:
        try:
            if haversine_distance(42.3601, -71.0589, r['lat'], r['lon']) <= SERVICE_RADIUS_KM * 1000:
                return f"{r['lat']},{r['lon']}"
        except (KeyError, TypeError):
            continue
    return None


def _fmt_time(iso_str):
    try:
        return parser.isoparse(iso_str).astimezone(BOSTON_TZ).strftime('%-I:%M %p')
    except Exception:
        return ''


def _fmt_minutes(minutes):
    if minutes >= 60:
        return f"{minutes // 60} hr {minutes % 60} min"
    return f"{minutes} min"


TRANSITOUS_MODE_MAP = {
    "SUBWAY": "SUBWAY", "METRO": "SUBWAY", "TRAM": "TRAM", "BUS": "BUS", "COACH": "BUS",
    "FERRY": "FERRY", "RAIL": "COMMUTER_TRAIN", "REGIONAL_RAIL": "COMMUTER_TRAIN",
    "REGIONAL_FAST_RAIL": "COMMUTER_TRAIN", "SUBURBAN": "COMMUTER_TRAIN",
    "LONG_DISTANCE": "LONG_DISTANCE_TRAIN", "HIGHSPEED_RAIL": "LONG_DISTANCE_TRAIN",
}


def _epoch(iso_str):
    dt = transit.parse_time(iso_str)
    return int(dt.timestamp()) if dt else None


def get_transitous_directions(from_place, to_place, dest_name=None, when=None, arrive_by=False):
    """Itineraries from the free Transitous (MOTIS) API, in the same shape as Google results.

    from_place/to_place are 'lat,lon'. Raises RoutingError when the router fails; an
    empty list means it answered but found no itineraries."""
    try:
        resp = requests.get(f"{TRANSITOUS_URL}/v5/plan",
                            params={"fromPlace": from_place, "toPlace": to_place, "numItineraries": 6,
                                    **({"time": when.replace(second=0, microsecond=0).isoformat(), "arriveBy": "true" if arrive_by else "false"} if when else {})},
                            headers=TRANSITOUS_HEADERS, timeout=20)
    except requests.RequestException as e:
        raise RoutingError("The trip planning service didn’t respond. Try again in a moment.") from e
    if getattr(resp, "status_code", 200) != 200:
        raise RoutingError(f"The trip planning service is unavailable right now (HTTP {resp.status_code}).")
    try:
        data = resp.json()
    except ValueError:
        raise RoutingError("The trip planning service sent an unreadable response.")

    routes = []
    seen = set()
    for it in data.get('itineraries', []):
        steps, detailed_steps, legs_meta = [], [], []
        walk_time = 0
        first_board = last_alight = None
        for leg in it.get('legs', []):
            mode = leg.get('mode', '')
            leg_minutes = round(leg.get('duration', 0) / 60)
            from_name = leg.get('from', {}).get('name', '')
            to_name = leg.get('to', {}).get('name', '')

            if mode == 'WALK':
                if (leg.get('distance') or 0) < 30 and leg_minutes <= 1:
                    continue                                   # "Walk 0 ft" between a stop and itself
                walk_time += leg_minutes
                dist = leg.get('distance') or 0
                dist_text = f"{dist / 1609.34:.1f} mi" if dist >= 160 else f"{int(dist * 3.281)} ft"
                final = to_name in ('END', '')
                target = (dest_name or 'your destination') if final else to_name
                origin_label = from_name if from_name not in ('START', '') else ''
                detailed_steps.append({
                    "type": "walk",
                    "instruction": f"Walk {dist_text} to {target}" if dist >= 30 else f"Walk to {target}",
                    "details": f"From {origin_label}" if origin_label else "From your starting point",
                    "duration": _fmt_minutes(max(1, leg_minutes))
                })
                continue

            vehicle_type = TRANSITOUS_MODE_MAP.get(mode, 'BUS')
            short = leg.get('routeShortName') or ''
            long_name = leg.get('routeLongName') or ''
            line_name = short or long_name
            color = f"#{leg['routeColor']}" if leg.get('routeColor') else '#666666'
            headsign = leg.get('headsign', '')
            num_stops = len(leg.get('intermediateStops') or []) + 1

            if vehicle_type == 'BUS' and short and ' ' in short:
                display = short  # e.g. "Green Line C Shuttle"
            elif vehicle_type == 'BUS' and short and not short.upper().startswith('SL'):
                display = f"Bus {short}"
            elif short.upper().startswith('SL'):
                display = f"Silver Line {short[2:]}".strip()
            elif vehicle_type == 'COMMUTER_TRAIN':
                display = long_name if long_name.endswith('Line') else f"{line_name} Line"
            elif long_name:
                display = long_name
            else:
                display = get_display_name(line_name, vehicle_type)

            first_board = first_board or from_name
            last_alight = to_name
            stops = [leg.get('from', {})] + list(leg.get('intermediateStops') or []) + [leg.get('to', {})]
            legs_meta.append({
                "name": display,
                "route_id": transit.mbta_id(leg.get('routeId')),
                "direction_id": int(leg['directionId']) if str(leg.get('directionId', '')).isdigit() else None,
                # Stops the rider uses (board/alight) vs. stops the vehicle only passes.
                "stop_ids": {transit.mbta_id(s.get('stopId')) for s in (leg.get('from', {}), leg.get('to', {})) if s.get('stopId')},
                "passed_stop_ids": {transit.mbta_id(s.get('stopId')) for s in (leg.get('intermediateStops') or []) if s.get('stopId')},
                "stop_names": {transit.mbta_id(s.get('stopId')): s.get('name', '') for s in stops if s.get('stopId')},
                "start": transit.parse_time(leg.get('startTime')),
                "end": transit.parse_time(leg.get('endTime')),
            })
            steps.append({
                "type": get_transit_type(vehicle_type, line_name),
                "name": display,
                "short_name": short or (long_name.split()[0] if long_name else vehicle_type[:3]),
                "color": color,
                "vehicle_type": vehicle_type
            })
            detailed_steps.append({
                "type": "transit",
                "instruction": f"Take {display}",
                "details": f"Board at {from_name}" + (f" toward {headsign}" if headsign else ""),
                "time": _fmt_time(leg.get('startTime', '')),
                "duration": _fmt_minutes(leg_minutes),
                "color": color,
                "num_stops": num_stops
            })
            detailed_steps.append({
                "type": "transit",
                "instruction": f"Get off at {to_name}",
                "details": f"{num_stops} stop{'s' if num_stops != 1 else ''}",
                "time": _fmt_time(leg.get('endTime', '')),
                "color": color
            })

        if not steps:
            continue

        duration = round(it.get('duration', 0) / 60)
        depart = _fmt_time(it.get('startTime', ''))
        arrive = _fmt_time(it.get('endTime', ''))
        key = (depart, arrive, tuple(s['name'] for s in steps))
        if key in seen:
            continue
        seen.add(key)

        routes.append({
            "duration": duration,
            "duration_text": _fmt_minutes(duration),
            "transfers": it.get('transfers', max(0, len(steps) - 1)),
            "depart_time": depart,
            "arrive_time": arrive,
            "depart_iso": it.get('startTime'),
            "arrive_iso": it.get('endTime'),
            "depart_ts": _epoch(it.get('startTime')),
            "arrive_ts": _epoch(it.get('endTime')),
            "time_range": f"{depart}—{arrive}" if depart and arrive else _fmt_minutes(duration),
            "walk_time": walk_time,
            "fare": estimate_fare(steps),
            "steps": steps,
            "detailed_steps": detailed_steps,
            "warnings": [],
            "first_board": first_board,
            "last_alight": last_alight,
            "summary": " → ".join(s['name'] for s in steps),
            "_legs": legs_meta,
        })

    return routes[:6]


# Places riders name that the geocoder misses or places on a nearby garage/stop.
LANDMARKS = {
    "bu": ("Boston University", "42.3505,-71.1054"), "boston university": ("Boston University", "42.3505,-71.1054"),
    "fenway park": ("Fenway Park", "42.3467,-71.0972"), "faneuil hall": ("Faneuil Hall", "42.3600,-71.0568"),
    "boston common": ("Boston Common", "42.3550,-71.0656"), "the common": ("Boston Common", "42.3550,-71.0656"),
    "gillette": ("Gillette Stadium", "42.0909,-71.2643"), "gillette stadium": ("Gillette Stadium", "42.0909,-71.2643"),
    "mit": ("MIT", "42.3601,-71.0942"), "harvard university": ("Harvard University", "42.3770,-71.1167"),
    "td garden": ("TD Garden", "42.3662,-71.0621"), "the garden": ("TD Garden", "42.3662,-71.0621"),
    "mgh": ("Mass General Hospital", "42.3626,-71.0685"), "mass general": ("Mass General Hospital", "42.3626,-71.0685"),
    "massachusetts general hospital": ("Mass General Hospital", "42.3626,-71.0685"),
    "bpl": ("Boston Public Library", "42.3493,-71.0782"), "boston public library": ("Boston Public Library", "42.3493,-71.0782"),
    "logan": ("Logan Airport", "42.3656,-71.0096"), "logan airport": ("Logan Airport", "42.3656,-71.0096"),
    "the aquarium": ("New England Aquarium", "42.3591,-71.0498"), "new england aquarium": ("New England Aquarium", "42.3591,-71.0498"),
    "northeastern": ("Northeastern University", "42.3398,-71.0892"), "northeastern university": ("Northeastern University", "42.3398,-71.0892"),
    "umass boston": ("UMass Boston", "42.3134,-71.0381"), "seaport": ("Seaport", "42.3510,-71.0440"),
}


def landmark(text):
    hit = LANDMARKS.get(re.sub(r"\s+", " ", str(text or "").lower().strip(" ,.?!")))
    return {"name": hit[0], "coords": hit[1]} if hit else None


_ABBREV = [(r"\brd\b", "road"), (r"\bst\b", "street"), (r"\bave?\b", "avenue"), (r"\bsq\b", "square"),
           (r"\bctr\b", "center"), (r"\bhts\b", "heights"), (r"\bjct\b", "junction"), (r"\bmt\b", "mount")]


def _norm_place(text):
    import re as _re
    t = _re.sub(r"[^a-z0-9/ ]", " ", str(text or "").lower())
    t = _re.sub(r"\s+(station|stop|sta|t stop)$", "", _re.sub(r"\s+", " ", t).strip())
    for pattern, full in _ABBREV:
        t = _re.sub(pattern, full, t)
    return t.strip()


def find_station_data(query):
    """Station data for a name, alias or abbreviation ("Chiswick Rd", "Park St", "hynes").
    Addresses that merely contain a station-like word are left to the geocoder."""
    if not query:
        return None
    q = _norm_place(query)
    if not q:
        return None
    alias = STATION_ALIASES.get(str(query).lower().strip()) or STATION_ALIASES.get(q)
    if alias and alias in STATION_DATA:
        return {"name": alias, **STATION_DATA[alias]}
    for name, data in STATION_DATA.items():
        if _norm_place(name) == q:
            return {"name": name, **data}
    # Unambiguous prefix of one station's name ("hynes convention" -> Hynes Convention Center).
    # Street addresses (leading house number) are never treated as stations.
    if q[:1].isdigit():
        return None
    hits = {}
    for name, data in STATION_DATA.items():
        if _norm_place(name).startswith(q) and len(q) >= 4:
            hits.setdefault(data["id"], (name, data))
    if len(hits) == 1:
        name, data = next(iter(hits.values()))
        return {"name": name, **data}
    return None


if __name__ == '__main__':
    import sys
    port = int(os.environ.get('PORT', 5001))
    if '--port' in sys.argv:
        idx = sys.argv.index('--port') + 1
        if idx < len(sys.argv): port = int(sys.argv[idx])
    
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║  🚇 MBTA Live Transit Dashboard with AI Chatbot + Location    ║
║  ─────────────────────────────────────────────────────────────║
║  Server: http://localhost:{port}                               ║
║  MBTA API: {'✅ Connected' if MBTA_API_KEY else '❌ Missing'}                                    ║
║  OpenAI:   {'✅ Connected' if OPENAI_API_KEY else '❌ Missing (basic chatbot)'}                  ║
║  Google:   {'✅ Connected' if GOOGLE_MAPS_API_KEY else '❌ Missing'}                             ║
║  Features: Chat, Map, Location, Trip Planner                  ║
╚═══════════════════════════════════════════════════════════════╝
""")
    
    app.run(debug=True, host='0.0.0.0', port=port)
