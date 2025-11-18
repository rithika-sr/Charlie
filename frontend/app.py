import streamlit as st
import requests

# Your Cloud Run URL
API_URL = "https://charlie-mbta-api-588293495748.us-central1.run.app/predict"

st.set_page_config(page_title="MBTA Delay Predictor", page_icon="🚇")

st.title("🚇 MBTA Delay Prediction")
st.write("Enter the route details below to forecast delay probability using your deployed ML model.")

direction = st.selectbox("Direction ID", [0, 1])
stop_sequence = st.number_input("Stop Sequence", min_value=1, max_value=30, step=1)

if st.button("Predict Delay"):
    payload = {
        "direction_id": int(direction),
        "stop_sequence": int(stop_sequence)
    }

    with st.spinner("Calling Cloud Run API..."):
        response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        result = response.json()

        st.success("Prediction Successful ✔")

        st.metric("Probability of Delay", f"{result['prob_delayed']:.4f}")
        
        label = "Delayed" if result["label"] == 1 else "On Time"
        st.metric("Prediction", label)
        
        st.caption(f"Model Version: {result['model_version']}")
    else:
        st.error(f"API Error: {response.text}")