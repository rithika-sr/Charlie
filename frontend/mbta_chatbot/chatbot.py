import streamlit as st
import requests
import json
from datetime import datetime, timezone
import folium
from streamlit_folium import st_folium
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import re

from google.transit import gtfs_realtime_pb2  # NEW: GTFS-RT bindings

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(page_title="MBTA AI Chatbot", page_icon="🚇", layout="wide")

# Light background for clarity
st.markdown(
    """
    <style>
    body {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ================================================================
# CONSTANTS
# ================================================================
MBTA_BASE = "https://api-v3.mbta.com"
PREDICT_API = "https://charlie-mbta-api-588293495748.us-central1.run.app/predict"

TRIP_UPDATES_URL = "https://cdn.mbta.com/realtime/TripUpdates.pb"

ROUTE_COLORS = {
    "Red": "red",
    "Mattapan": "red",
    "Orange": "orange",
    "Blue": "blue",
    "Green": "green",
    "Green-B": "green",
    "Green-C": "green",
    "Green-D": "green",
    "Green-E": "green",
}

LINE_KEYWORDS = {
    "red line": ["Red"],
    "orange line": ["Orange"],
    "blue line": ["Blue"],
    "green line": ["Green-B", "Green-C", "Green-D", "Green-E"],
    "green b": ["Green-B"],
    "green c": ["Green-C"],
    "green d": ["Green-D"],
    "green e": ["Green-E"],
    "silver line": ["741", "742", "743", "746", "749", "751"],  # SL1..SL5 etc.
}

# ================================================================
# MBTA API HELPERS
# ================================================================
def safe_parse_epoch(ts: int | None):
    if not ts:
        return None
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        return None


@st.cache_data(ttl=600)
def get_routes():
    """Return dict route_id -> (long_name, type)."""
    url = f"{MBTA_BASE}/routes"
    try:
        r = requests.get(url).json()
    except Exception:
        return {}
    routes = {}
    for row in r.get("data", []):
        rid = row["id"]
        long_name = row["attributes"]["long_name"] or row["attributes"]["short_name"] or rid
        rtype = row["attributes"]["type"]
        routes[rid] = (long_name, rtype)
    return routes


@st.cache_data(ttl=1800)
def get_stops():
    """Return dict stop_id -> stop_name and name->ids mapping."""
    url = f"{MBTA_BASE}/stops"
    try:
        r = requests.get(url).json()
    except Exception:
        return {}, {}

    id_to_name = {}
    # name_lower -> list[stop_id]
    name_to_ids = {}
    for row in r.get("data", []):
        sid = row["id"]
        name = row["attributes"]["name"]
        if not name:
            continue
        id_to_name[sid] = name
        key = name.lower()
        name_to_ids.setdefault(key, []).append(sid)
    return id_to_name, name_to_ids


# ---------- GTFS-RT: TripUpdates ----------
@st.cache_data(ttl=15)
def load_trip_updates():
    """
    Load MBTA GTFS-RT TripUpdates and flatten into a list of dicts:
    each dict is one (route, trip, stop, stop_sequence, timestamp).
    """
    try:
        resp = requests.get(TRIP_UPDATES_URL, timeout=10)
        resp.raise_for_status()
    except Exception:
        return []

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(resp.content)

    updates = []
    for entity in feed.entity:
        if not entity.HasField("trip_update"):
            continue
        tu = entity.trip_update
        route_id = tu.trip.route_id
        trip_id = tu.trip.trip_id

        # GTFS-RT spec doesn't guarantee headsign here; we'll fill later from JSON if needed
        headsign = None

        for stu in tu.stop_time_update:
            stop_id = stu.stop_id
            stop_seq = stu.stop_sequence

            arr_ts = stu.arrival.time if stu.HasField("arrival") and stu.arrival.time else None
            dep_ts = stu.departure.time if stu.HasField("departure") and stu.departure.time else None
            ts = dep_ts or arr_ts
            if not ts:
                continue

            updates.append(
                {
                    "route_id": route_id,
                    "trip_id": trip_id,
                    "stop_id": stop_id,
                    "stop_sequence": stop_seq,
                    "timestamp": ts,
                    "headsign": headsign,  # placeholder; may stay None
                }
            )

    return updates


# ---------- OPTIONAL: map dest text -> trip_ids using JSON (for headsigns) ----------
def get_trips_matching_destination(stop_id: str, route_hint_ids, dest_headsign_hint: str | None):
    """
    Use MBTA JSON only to find which trip_ids have a headsign containing dest_headsign_hint.
    Times still come from GTFS-RT.
    """
    if not dest_headsign_hint:
        return set(), {}

    url = f"{MBTA_BASE}/predictions?filter[stop]={stop_id}&include=trip,route"
    try:
        r = requests.get(url, timeout=10).json()
    except Exception:
        return set(), {}

    # trip_id -> headsign
    trip_heads: dict[str, str | None] = {}
    for inc in r.get("included", []):
        if inc["type"] == "trip":
            trip_heads[inc["id"]] = inc["attributes"].get("headsign")

    valid_trips = set()
    for row in r.get("data", []):
        rel = row.get("relationships", {})
        trip_id = rel.get("trip", {}).get("data", {}).get("id")
        route_id = rel.get("route", {}).get("data", {}).get("id")
        if not trip_id or not route_id:
            continue
        if route_hint_ids and route_id not in route_hint_ids:
            continue
        hs = (trip_heads.get(trip_id) or "").lower()
        if dest_headsign_hint.lower() in hs:
            valid_trips.add(trip_id)

    return valid_trips, trip_heads


# ---------- STOP-BASED PREDICTIONS (NOW GTFS-RT) ----------
def fetch_predictions_for_stop(
    stop_id: str,
    route_hint_ids=None,
    dest_headsign_hint: str | None = None,
):
    """
    Use GTFS-RT TripUpdates for exact live times.
    Optionally filter by:
      - route_hint_ids (e.g., Green-B)
      - destination headsign substring ("boston college") using JSON-only for mapping.
    """
    from math import inf

    if route_hint_ids is None:
        route_hint_ids = []

    updates = load_trip_updates()
    routes = get_routes()

    # If they gave a destination, map that text -> set of trip_ids using JSON (for headsigns)
    trip_filter = set()
    trip_heads = {}
    if dest_headsign_hint:
        trip_filter, trip_heads = get_trips_matching_destination(stop_id, route_hint_ids, dest_headsign_hint)

    rows = []
    for u in updates:
        if u["stop_id"] != stop_id:
            continue
        if route_hint_ids and u["route_id"] not in route_hint_ids:
            continue
        if trip_filter and u["trip_id"] not in trip_filter:
            continue

        dt = safe_parse_epoch(u["timestamp"])
        countdown = None
        if dt:
            now = datetime.now(timezone.utc)
            countdown = round((dt - now).total_seconds() / 60.0, 1)

        route_id = u["route_id"]
        route_name = routes.get(route_id, (route_id, None))[0]

        # headsign from JSON if available; otherwise None
        hs = trip_heads.get(u["trip_id"]) if trip_heads else None

        rows.append(
            {
                "Route ID": route_id,
                "Route Name": route_name,
                "Headsign": hs,
                "Time": dt.isoformat() if dt else None,
                "Countdown (min)": countdown,
            }
        )

    rows = sorted(
        rows,
        key=lambda x: x["Countdown (min)"] if x["Countdown (min)"] is not None else inf,
    )
    return rows


# ---------- ROUTE-LEVEL PREDICTIONS (GTFS-RT) ----------
def get_live_predictions_for_route(route_id: str):
    """
    Route-wide view using GTFS-RT TripUpdates.
    Returns list of dicts with Stop Sequence + Countdown.
    """
    from math import inf

    updates = load_trip_updates()
    rows = []
    for u in updates:
        if u["route_id"] != route_id:
            continue
        dt = safe_parse_epoch(u["timestamp"])
        countdown = None
        if dt:
            now = datetime.now(timezone.utc)
            countdown = round((dt - now).total_seconds() / 60.0, 1)

        rows.append(
            {
                "Stop Sequence": u["stop_sequence"],
                "Stop ID": u["stop_id"],
                "Countdown (min)": countdown,
                "Status": "live",
            }
        )

    rows = sorted(
        rows,
        key=lambda x: x["Countdown (min)"] if x["Countdown (min)"] is not None else inf,
    )
    return rows


def get_live_vehicles(route_id: str):
    """
    For the map, REST vehicles endpoint is fine (same real-time source).
    """
    url = f"{MBTA_BASE}/vehicles?filter[route]={route_id}"
    try:
        r = requests.get(url).json()
    except Exception:
        return []
    locs = []
    for row in r.get("data", []):
        a = row["attributes"]
        lat, lon = a.get("latitude"), a.get("longitude")
        if lat is None or lon is None:
            continue
        locs.append({"lat": lat, "lon": lon, "label": a.get("label")})
    return locs


def predict_delay(route_id: str, stop_seq: int):
    body = {"route_id": route_id, "stop_sequence": stop_seq}
    try:
        r = requests.post(PREDICT_API, json=body, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ================================================================
# TEXT INTERPRETATION HELPERS (unchanged)
# ================================================================
def find_stop_mentions(msg_lower: str, id_to_name: dict):
    """
    Very simple matching: look for any stop name that appears as a substring
    in the message. Return a list of (stop_id, stop_name) sorted by length desc.
    """
    matches = []
    for sid, name in id_to_name.items():
        nlow = name.lower()
        if nlow in msg_lower:
            matches.append((sid, name))
    # longer names first to avoid 'Arlington' vs 'Arlington Center'
    matches.sort(key=lambda x: len(x[1]), reverse=True)
    return matches


def extract_line_hints(msg_lower: str):
    route_ids = []
    for key, rids in LINE_KEYWORDS.items():
        if key in msg_lower:
            route_ids.extend(rids)
    # If they directly mention 'route 1' or 'bus 1'
    m = re.search(r"route\s+(\w+)", msg_lower)
    if m:
        route_ids.append(m.group(1))
    return list(set(route_ids))


def extract_destination_hint(msg_lower: str):
    """
    Extremely simple heuristic: if there's a 'to X', return X as text,
    and we'll match it against headsigns.
    """
    if " to " in msg_lower:
        dest_part = msg_lower.split(" to ", 1)[1]
        # strip common suffixes
        dest_part = dest_part.replace("station", "").replace("stop", "")
        dest_part = dest_part.strip()
        # only first few words
        dest_tokens = dest_part.split()
        if dest_tokens:
            return " ".join(dest_tokens[:3])
    return None


def interpret_freeform(message: str, routes: dict, id_to_name: dict):
    """
    Main NLP-ish interpreter.

    Returns:
      - ("STOP_QUERY", stop_id, stop_name, route_hint_ids, dest_hint)
      - "ROUTE_LIST"
      - ("LIVE_ROUTE", route_id)
      - ("PREDICT", route_id, stop_seq)
      - "BAD_PREDICT"
      - "UNKNOWN"
    """
    msg = message.strip()
    msg_lower = msg.lower()

    # 1) List routes
    if "list" in msg_lower and "route" in msg_lower:
        return "ROUTE_LIST"

    # 2) Predict delay for route X stop Y
    if "predict" in msg_lower or "delay" in msg_lower:
        try:
            parts = msg_lower.split()
            r_index = parts.index("route") + 1
            s_index = parts.index("stop") + 1
            rid = parts[r_index]
            stop = int(parts[s_index])
            return ("PREDICT", rid, stop)
        except Exception:
            return "BAD_PREDICT"

    # 3) Live info for a specific route ("live for red line")
    if "live" in msg_lower and "route" in msg_lower:
        try:
            parts = msg_lower.split()
            r_index = parts.index("route") + 1
            rid = parts[r_index]
            return ("LIVE_ROUTE", rid)
        except Exception:
            pass

    # 4) If they mention a route by id or name directly
    for rid, (name, _) in routes.items():
        if rid.lower() in msg_lower or name.lower() in msg_lower:
            if "live" in msg_lower or "next" in msg_lower:
                return ("LIVE_ROUTE", rid)

    # 5) Try stop-based query: look for any stop names mentioned
    stop_matches = find_stop_mentions(msg_lower, id_to_name)
    if stop_matches:
        origin_sid, origin_name = stop_matches[0]
        route_hint_ids = extract_line_hints(msg_lower)
        dest_hint = extract_destination_hint(msg_lower)
        return ("STOP_QUERY", origin_sid, origin_name, route_hint_ids, dest_hint)

    # 6) Fallback
    return "UNKNOWN"


# ================================================================
# MAP DRAWING
# ================================================================
def draw_map(route_id: str, routes: dict):
    vehicles = get_live_vehicles(route_id)
    if not vehicles:
        st.info("No active vehicles on this route right now.")
        return

    long_name, _ = routes[route_id]
    avg_lat = sum(v["lat"] for v in vehicles) / len(vehicles)
    avg_lon = sum(v["lon"] for v in vehicles) / len(vehicles)

    color = "blue"
    for key, c in ROUTE_COLORS.items():
        if key.lower() in long_name.lower():
            color = c
            break

    fmap = folium.Map(
        location=[avg_lat, avg_lon],
        zoom_start=13,
        tiles="CartoDB Positron",
        control_scale=True,
    )
    for v in vehicles:
        folium.CircleMarker(
            location=[v["lat"], v["lon"]],
            radius=7,
            color=color,
            fill=True,
            fill_color=color,
            popup=v["label"],
        ).add_to(fmap)

    st.markdown(f"#### 🗺 Live vehicles for {long_name} ({route_id})")
    st_folium(fmap, width=750, height=450)


# ================================================================
# HEAT STRIP
# ================================================================
def draw_heat(preds, route_id: str, routes: dict):
    stops = [p["Stop Sequence"] for p in preds if p["Countdown (min)"] is not None]
    mins = [p["Countdown (min)"] for p in preds if p["Countdown (min)"] is not None]

    if not mins:
        st.info("Not enough arrival data to draw heat strip.")
        return

    data = np.array([mins])

    fig, ax = plt.subplots(figsize=(max(6, len(stops) * 0.4), 1.8))
    im = ax.imshow(data, aspect="auto")
    ax.set_yticks([])
    ax.set_xticks(range(len(stops)))
    ax.set_xticklabels(stops, rotation=45, ha="right")
    ax.set_xlabel("Stop Sequence")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Minutes until arrival")

    long_name, _ = routes[route_id]
    ax.set_title(f"Delay Heat Strip — {long_name} ({route_id})")

    st.pyplot(fig)


# ================================================================
# VOICE INPUT (placeholder)
# ================================================================
def voice_component():
    """
    Shows a mic button and transcript using Web Speech API inside an iframe.
    """
    html = """
    <html>
      <body>
        <button id="start-btn">🎤 Start / Stop Voice</button>
        <p id="status">Click the mic and speak…</p>
        <p><strong>Transcript:</strong> <span id="transcript"></span></p>

        <script>
        const btn = document.getElementById("start-btn");
        const status = document.getElementById("status");
        const transcriptSpan = document.getElementById("transcript");
        let recognizing = false;
        let recognition = null;

        function setup() {
          const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
          if (!SpeechRecognition) {
            status.textContent = "Speech recognition not supported in this browser.";
            btn.disabled = true;
            return;
          }
          recognition = new SpeechRecognition();
          recognition.lang = "en-US";
          recognition.interimResults = false;
          recognition.continuous = false;

          recognition.onstart = () => {
            recognizing = true;
            status.textContent = "Listening… speak now.";
          };
          recognition.onend = () => {
            recognizing = false;
            status.textContent = "Stopped listening.";
          };
          recognition.onerror = (event) => {
            recognizing = false;
            status.textContent = "Error: " + event.error;
          };
          recognition.onresult = (event) => {
            let text = "";
            for (let i = event.resultIndex; i < event.results.length; ++i) {
              text += event.results[i][0].transcript;
            }
            transcriptSpan.textContent = text;
          };
        }

        btn.onclick = () => {
          if (!recognition) {
            setup();
            if (!recognition) return;
          }
          if (recognizing) {
            recognition.stop();
          } else {
            recognition.start();
          }
        };

        setup();
        </script>
      </body>
    </html>
    """
    components.html(html, height=220)


# ================================================================
# INITIAL STATE
# ================================================================
if "history" not in st.session_state:
    st.session_state.history = [
        {
            "role": "assistant",
            "text": "Hi! I am your MBTA AI assistant 🚇\n\n"
            "You can type things like:\n"
            "- `I'm at Park Street, next train to Boston College`\n"
            "- `at Hynes Convention Center next train to Riverside`\n"
            "- `live for Red Line`\n"
            "- `predict delay for route 1 stop 5`\n",
        }
    ]

if "draft" not in st.session_state:
    st.session_state.draft = ""

routes = get_routes()
id_to_name, name_to_ids = get_stops()
route_labels = sorted([f"{name} ({rid})" for rid, (name, _) in routes.items()])

# ================================================================
# SIDEBAR CONTROLS
# ================================================================
with st.sidebar:
    st.header("🚏 MBTA Tools")

    if routes:
        selected_label = st.selectbox("Select Route", route_labels)
        current_rid = selected_label.split("(")[-1].replace(")", "").strip()
        long_name, _ = routes[current_rid]
        st.markdown(f"**Selected:** {long_name} (`{current_rid}`)")

        if st.button("📡 Live Arrivals (GTFS-RT)"):
            st.json(get_live_predictions_for_route(current_rid))

        if st.button("🗺 Show Map"):
            draw_map(current_rid, routes)

        if st.button("🔥 Delay Heat Strip"):
            predictions = get_live_predictions_for_route(current_rid)
            draw_heat(predictions, current_rid, routes)

        st.markdown("---")
        st.markdown("### 🔮 Model Delay Prediction")
        stop_seq_sidebar = st.number_input("Stop sequence", min_value=1, step=1, value=1)
        if st.button("Predict Delay"):
            st.json(predict_delay(current_rid, stop_seq_sidebar))
    else:
        st.error("Could not load MBTA routes. Check your internet connection / MBTA API.")

    st.markdown("---")
    st.markdown("### 🎤 Voice Input (demo)")
    st.caption("Use Chrome for best results. Transcript stays in this box for now.")
    voice_component()

# ================================================================
# MAIN CHAT INTERFACE
# ================================================================
st.title("🚇 MBTA Real-Time Chatbot (GTFS-RT)")

# Show message history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])

# Text input (can be filled by typing)
user_text = st.text_input(
    "Type your question (e.g., `I'm at Park Street, next train to Boston College`):",
    value=st.session_state.draft,
    key="chat_input",
)

col_send, col_clear = st.columns([1, 1])
send_clicked = col_send.button("Send")
if col_clear.button("Clear input"):
    st.session_state.draft = ""
    st.rerun()

if send_clicked and user_text.strip():
    # Persist draft
    st.session_state.draft = user_text

    # Append user message
    st.session_state.history.append({"role": "user", "text": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # Process via interpreter
    intent = interpret_freeform(user_text, routes, id_to_name)

    # ---------- HANDLE INTENTS ----------
    if intent == "ROUTE_LIST":
        reply = "Here are the MBTA routes I know:\n" + "\n".join(
            [f"- {name} (`{rid}`)" for rid, (name, _) in routes.items()]
        )

    elif intent == "BAD_PREDICT":
        reply = "To predict delay, try: `predict delay for route 1 stop 5`."

    elif intent == "UNKNOWN":
        reply = (
            "I couldn't fully understand that.\n\n"
            "Try things like:\n"
            "- `I'm at Park Street, next train to Boston College`\n"
            "- `I'm at Airport, next blue line train`\n"
            "- `list all routes`\n"
            "- `predict delay for route 1 stop 5`"
        )

    elif isinstance(intent, tuple) and intent[0] == "PREDICT":
        rid, stop_seq = intent[1], intent[2]
        result = predict_delay(rid, stop_seq)
        if "error" in result:
            reply = f"Error calling prediction API: `{result['error']}`"
        else:
            reply = (
                f"Delay prediction for route `{rid}`, stop `{stop_seq}`:\n\n"
                f"```json\n{json.dumps(result, indent=2)}\n```"
            )

    elif isinstance(intent, tuple) and intent[0] == "LIVE_ROUTE":
        rid = intent[1]
        preds = get_live_predictions_for_route(rid)
        if preds:
            long_name, _ = routes.get(rid, (rid, 3))
            lines = [f"Live arrivals for **{long_name} ({rid})**:"]
            for p in preds[:10]:
                lines.append(
                    f"- Stop seq {p['Stop Sequence']}: "
                    f"{p['Countdown (min)']} min"
                )
            reply = "\n".join(lines)
        else:
            reply = f"No live prediction data available for route `{rid}` right now."

    elif isinstance(intent, tuple) and intent[0] == "STOP_QUERY":
        origin_sid, origin_name, route_hint_ids, dest_hint = intent[1], intent[2], intent[3], intent[4]
        preds = fetch_predictions_for_stop(origin_sid, route_hint_ids, dest_hint)
        if not preds:
            extra = ""
            if dest_hint:
                extra += f" with destination like '{dest_hint}'"
            if route_hint_ids:
                extra += f" on routes {route_hint_ids}"
            reply = (
                f"I looked for upcoming trips from **{origin_name}**{extra}, "
                "but couldn't find any live vehicles right now."
            )
        else:
            lines = [f"Upcoming trips from **{origin_name}**:"]
            for p in preds[:8]:
                hs = p["Headsign"] or "(destination unknown)"
                lines.append(
                    f"- {p['Route Name']} (`{p['Route ID']}`) → {hs}: "
                    f"in {p['Countdown (min)']} min"
                )
            reply = "\n".join(lines)

    else:
        reply = "Something unexpected happened while parsing your request."

    # Show assistant reply & store
    with st.chat_message("assistant"):
        st.markdown(reply)
    st.session_state.history.append({"role": "assistant", "text": reply})

    # Clear draft and rerun
    st.session_state.draft = ""
    st.rerun()