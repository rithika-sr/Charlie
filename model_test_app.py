# model_test_app.py  (ROOT of Charlie-main)

import streamlit as st
from typing import List, Dict, Any

from chatbot.model_utils import predict_delay
from chatbot.mbta_client import (
    MBTAAPIError,
    resolve_stop_id,
    get_next_arrival,
)

st.set_page_config(page_title="Charlie – MBTA Real-Time Test Interface", page_icon="🚇")

st.title("🚇 Charlie – MBTA Real-Time Test Interface")
st.write(
    """
This interface tests:

• **MBTA V3 API** real-time predictions  
• Your **ML delay prediction model**  
• Simple **station name extraction** from natural language  

Use this to validate that your end-to-end Charlie system is working.
"""
)

st.markdown("---")

# -----------------------------
# 1. USER INPUT
# -----------------------------
st.subheader("Ask Charlie")

user_question = st.text_input(
    'Ask Charlie something (e.g., "I\'m at Hynes next train to Riverside")',
    value="I'm at Hynes Convention Center, next train to Riverside",
)

st.caption("Charlie will try to detect the station from your sentence. You can adjust it below if needed.")

# naive station extraction based on 'at'
default_station = ""
lower_q = user_question.lower()
if " at " in lower_q:
    part = lower_q.split(" at ", 1)[1]
    # trim on keywords like 'next', 'to', ','
    for cut in [" next", " to ", ",", ".", "?"]:
        if cut in part:
            part = part.split(cut, 1)[0]
    default_station = part.strip().title()

station_input = st.text_input(
    "Station / Stop name (you can edit this if detection is wrong)",
    value=default_station or "Hynes Convention Center",
)

max_deps = st.slider(
    "How many upcoming departures do you want to see?",
    min_value=1,
    max_value=10,
    value=5,
)

run_btn = st.button("Run Charlie Test 🚀")


# -----------------------------
# 2. HELPER TO RENDER TABLE
# -----------------------------
def render_departures_table(preds: List[Dict[str, Any]]) -> None:
    if not preds:
        st.info("No upcoming predictions from MBTA for this stop right now.")
        return

    rows = []
    for p in preds:
        route_label = f"{p.get('route_name', '?')} ({p.get('route_id')})"
        when = p.get("arrival_time") or p.get("departure_time") or "N/A"
        status = p.get("status") or "no status"
        vehicle_id = p.get("vehicle_id") or "-"
        lat = p.get("vehicle_lat")
        lon = p.get("vehicle_lon")

        if lat is not None and lon is not None:
            vehicle_pos = f"{lat:.5f}, {lon:.5f}"
        else:
            vehicle_pos = "-"

        rows.append(
            {
                "Route": route_label,
                "Direction ID": p.get("direction_id"),
                "Arrival / Departure Time": when,
                "Status": status,
                "Vehicle ID": vehicle_id,
                "Vehicle Position (lat, lon)": vehicle_pos,
                "Stop Seq": p.get("stop_sequence"),
            }
        )

    st.table(rows)


# -----------------------------
# 3. MAIN RUN LOGIC
# -----------------------------
if run_btn:
    if not station_input.strip():
        st.error("Please provide a station / stop name.")
    else:
        st.markdown("### 📡 Real-Time Departures (MBTA API)")

        try:
            # 1) resolve free-text station name to stop_id
            stop_id, resolved_name = resolve_stop_id(station_input.strip())

            st.success(f"Using MBTA stop **{resolved_name}** (stop_id: `{stop_id}`)")

            # 2) Get next arrival + list of predictions
            next_pred, all_preds = get_next_arrival(stop_id)

            if not all_preds:
                st.warning(
                    f"No upcoming predictions found right now for `{resolved_name}`. "
                    "Try another station or time of day."
                )
            else:
                # show table of upcoming predictions
                render_departures_table(all_preds)

                # show a small summary for the "next" one
                next_time = next_pred.get("arrival_time") or next_pred.get("departure_time") or "N/A"
                next_route = next_pred.get("route_name", "?")
                next_status = next_pred.get("status") or "no status"

                st.markdown(
                    f"""
**Next incoming vehicle** at **{resolved_name}**  
• Route: `{next_route}`  
• Time: `{next_time}`  
• Status: `{next_status}`
"""
                )

            st.markdown("### 🤖 ML Delay Prediction")

            # 3) Use the first prediction to feed your ML model
            if all_preds:
                first = all_preds[0]
                dir_id = first.get("direction_id") or 0
                stop_seq = first.get("stop_sequence") or 0

                try:
                    delay_prob = predict_delay(direction_id=dir_id, stop_sequence=stop_seq)
                    st.success(f"Predicted delay probability: **{delay_prob:.4f}**")
                except Exception as e:
                    st.error(f"ML model error: {e}")
            else:
                st.info("No predictions available to run through the ML delay model.")

            st.markdown("### 💬 Charlie Chatbot Summary")
            st.write(
                f"""
Based on real-time MBTA data and the ML delay model:

• Station: **{station_input.strip()}**  
• Resolved MBTA stop: **{resolved_name}** (`{stop_id}`)  
• Number of upcoming predictions received: **{len(all_preds)}**

(LLM natural-language responses will be added next.)
"""
            )

        except MBTAAPIError as e:
            st.error(f"MBTA API error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")