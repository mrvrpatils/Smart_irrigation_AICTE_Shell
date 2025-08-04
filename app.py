import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Set up the page configuration first
st.set_page_config(
    page_title="Smart Irrigation Dashboard",
    page_icon="💧",
    layout="wide"
)

# Function to load our trained models.
# Using st.cache_resource so it only loads once and the app runs faster.
@st.cache_resource
def load_artifacts():
    try:
        pipeline = joblib.load("irrigation_pipeline.pkl")
        mean_values = joblib.load("mean_sensor_values.pkl")
        return pipeline, mean_values
    except FileNotFoundError as e:
        st.error("ERROR: A required model file was not found.")
        st.error(f"Details: {e}")
        st.error("Please make sure 'irrigation_pipeline.pkl' and 'mean_sensor_values.pkl' are in the same folder as this script.")
        st.stop()

# Load the models into global variables
pipeline, mean_values = load_artifacts()
sensor_names = [f'sensor_{i}' for i in range(20)]

# This helper function simulates the 20 sensor values based on a single temperature.
# It's a simplified model of how I expect the environment to behave.
def map_temp_to_sensors(temp):
    # Normalize temperature to a 0-1 scale, assuming a typical range of 0-40°C
    temp_norm = max(0, min(temp / 40.0, 1.0))
    
    # Simple environmental assumptions: higher temp means lower humidity and more wind
    humidity_norm = 1.0 - temp_norm * 0.9 
    wind_norm = temp_norm * 0.5
    
    # The soil moisture logic was a bit tricky, this formula felt the most realistic
    soil_moisture = (1 - humidity_norm) + (temp_norm * 0.5)
    
    # Generate sensor groups with a bit of random noise for realism
    sensors_0_9 = np.clip(np.random.normal(loc=soil_moisture, scale=0.1, size=10), 0, 1)
    sensors_10_14 = np.clip(np.random.normal(loc=temp_norm, scale=0.05, size=5), 0, 1)
    sensors_15_19 = np.clip(np.random.normal(loc=(humidity_norm + wind_norm) / 2, scale=0.1, size=5), 0, 1)
    
    # Combine and ensure all values are within the valid [0, 1] range
    all_sensors = np.concatenate([sensors_0_9, sensors_10_14, sensors_15_19])
    return list(np.clip(all_sensors, 0, 1))


# --- Sidebar UI Elements ---
st.sidebar.title("🌿 Sensor Inputs & Settings")
st.sidebar.info("Adjust sensor values or select a preset, then see the analysis on the main page.")

# Pre-defined scenarios for easy testing
PRESETS = {
    "Select a Scenario...": [0.5] * 20,
    "Hot and Dry Afternoon": [0.1, 0.9, 0.15, 0.85, 0.2, 0.05, 0.9, 0.1, 0.8, 0.1, 0.1, 0.95, 0.2, 0.8, 0.1, 0.9, 0.1, 0.9, 0.05, 0.8],
    "Cool Morning After Rain": [0.9, 0.2, 0.85, 0.3, 0.95, 0.8, 0.2, 0.9, 0.2, 0.8, 0.9, 0.25, 0.9, 0.3, 0.8, 0.2, 0.9, 0.2, 0.85, 0.3],
    "Humid but Windy Day": [0.4, 0.6, 0.8, 0.5, 0.3, 0.7, 0.6, 0.8, 0.5, 0.6, 0.4, 0.6, 0.8, 0.5, 0.7, 0.6, 0.4, 0.6, 0.7, 0.5]
}

# Initialize session state for the sensor values if it's the first run
if 'sensor_values' not in st.session_state:
    st.session_state.sensor_values = PRESETS["Select a Scenario..."]

# This function is a callback that gets triggered when the user selects a preset
def update_sensors_from_preset():
    st.session_state.sensor_values = PRESETS[st.session_state.scenario_choice]

st.sidebar.selectbox(
    "Weather Scenario Presets:",
    options=list(PRESETS.keys()),
    key="scenario_choice",
    on_change=update_sensors_from_preset
)

# I'm putting the manual sliders inside an expander to keep the sidebar from getting too cluttered
sensor_values_list = []
with st.sidebar.expander("Manually Adjust Sensor Values (0-1)"):
    for i in range(20):
        value = st.slider(
            f"Sensor {i}", 0.0, 1.0, st.session_state.sensor_values[i], 0.01, key=f"slider_{i}"
        )
        sensor_values_list.append(value)

# Inputs for the cost analysis part of the dashboard
st.sidebar.subheader("Analysis Settings")
water_cost_per_unit = st.sidebar.number_input("Water Cost (₹ per 1000L)", min_value=10.0, value=120.0, step=5.0)
water_flow_rate = st.sidebar.number_input("Sprinkler Flow Rate (Liters per hour)", min_value=100, value=500, step=50)


# --- Main Page Content ---
st.title("💧 Smart Irrigation Decision Support")
st.markdown("Compare the AI's recommendation with your manual override to see the impact on cost and crop health.")

# --- Prediction Logic ---
# The model needs a DataFrame with the correct column names
input_df = pd.DataFrame([sensor_values_list], columns=sensor_names)
ai_prediction = pipeline.predict(input_df)[0]
ai_probabilities = pipeline.predict_proba(input_df)

# Display the main AI recommendation at the top
st.header("🤖 AI Recommendation")
cols = st.columns(3)
for i, status in enumerate(ai_prediction):
    prob_on = ai_probabilities[i][0, 1]
    with cols[i]:
        st.metric(f"Parcel {i}", "WATER" if status == 1 else "DO NOT WATER")
        st.progress(prob_on)
        st.caption(f"Confidence to Water: {prob_on:.1%}")

st.markdown("---")

# --- Manual Override and Analysis Section ---
st.header("👤 Your Manual Override")
st.write("Turn sprinklers ON or OFF to see the consequences of your decision compared to the AI.")
manual_override = [0, 0, 0]
analysis_cols = st.columns(3)

# Initialize totals for the summary at the bottom
total_water_saved = 0
total_cost_saved = 0

for i in range(3):
    with analysis_cols[i]:
        st.subheader(f"Parcel {i}")
        # The toggle switch lets the user override the AI's suggestion
        manual_on = st.toggle("Manual Sprinkler ON/OFF", value=(ai_prediction[i] == 1), key=f"toggle_{i}")
        
        # Show the correct image based on the toggle's current state
        if manual_on:
            st.image("sprinkler_on.jpg", width=100)
        else:
            st.image("sprinkler_off.jpg", width=100)
        
        ai_on = (ai_prediction[i] == 1)
        water_saved, cost_saved = 0, 0

        # Case 1: AI says water, user disagrees
        if ai_on and not manual_on:
            water_saved = water_flow_rate
            cost_saved = (water_saved / 1000) * water_cost_per_unit
            st.success(f"💧 Water Saved: {water_saved} L/hr")
            st.success(f"💰 Cost Saved: ₹{cost_saved:.2f}/hr")
        # Case 2: AI says don't water, user disagrees
        elif not ai_on and manual_on:
            water_saved = -water_flow_rate
            cost_saved = -((water_flow_rate / 1000) * water_cost_per_unit)
            st.warning(f"💧 Water Used: {water_flow_rate} L/hr")
            st.warning(f"💸 Extra Cost: ₹{abs(cost_saved):.2f}/hr")
        # Case 3: Both agree
        else:
            st.info("✅ AI and manual decisions align.")
        
        # Add to the running totals
        total_water_saved += water_saved
        total_cost_saved += cost_saved
        
        # Now, calculate and show the crop stress risk
        st.markdown("**Crop Stress Risk**")
        if ai_on and not manual_on:
            # This logic calculates how similar the current conditions are to the
            # average "thirsty" conditions the model learned from.
            on_means_np = np.array(list(mean_values[f'parcel_{i}']['on_means'].values()))
            current_vals_np = np.array(sensor_values_list)
            distance = np.linalg.norm(current_vals_np - on_means_np)
            norm_of_means = np.linalg.norm(on_means_np)
            
            stress_score = max(0, min(1 - (distance / norm_of_means), 1)) if norm_of_means > 0 else 0
            
            st.progress(stress_score)
            st.error(f"High risk of under-watering detected!")
        else:
            st.progress(0.0)
            st.write("Low risk of under-watering.")

# A final summary at the bottom of the page
st.markdown("---")
st.header("📈 Overall Impact Summary")
summary_cols = st.columns(2)
with summary_cols[0]:
    st.metric("Total Water Savings (L/hr)", f"{total_water_saved:,.0f}")
with summary_cols[1]:
    st.metric("Total Cost Savings (₹/hr)", f"₹{total_cost_saved:,.2f}")

# Fun little feature: show balloons if the user saved money
if total_cost_saved > 0:
    st.balloons()
    st.success("Great decision! You've saved both water and money by following the AI recommendation.")
elif total_cost_saved < 0:
    st.warning("The manual override resulted in higher costs compared to the AI's suggestion.")

st.markdown("---")

# --- Bonus Feature: Quick Temperature Check ---
st.header("🌡️ Quick Check by Temperature")
st.write("Use this slider to see how the irrigation need changes based on a single temperature value. This is a simulation and does not affect the main controls in the sidebar.")

temp_input = st.slider("Select Temperature (°C)", 0.0, 45.0, 25.0, 0.5)

# Simulate sensor values and get a prediction just for this section
temp_based_sensors = map_temp_to_sensors(temp_input)
temp_input_df = pd.DataFrame([temp_based_sensors], columns=sensor_names)
temp_prediction = pipeline.predict(temp_input_df)[0]

# Display the results for the temperature check
st.write(f"**At {temp_input}°C, the model's recommendation is:**")
temp_cols = st.columns(3)
for i in range(3):
    with temp_cols[i]:
        status = temp_prediction[i]
        if status == 1:
            st.success(f"**Parcel {i}:** WATER")
            st.image("sprinkler_on.jpg", width=80)
        else:
            st.error(f"**Parcel {i}:** DO NOT WATER")
            st.image("sprinkler_off.jpg", width=80)