import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="EV Purchase Predictor",
    page_icon="🔋",
    layout="wide",
)

# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/lgbmc.joblib")


model = load_model()

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("Will You Buy an EV?")
st.caption("Enter the customer information below to estimate the purchase outcome.")

# ---------------------------------------------------------
# Customer information
# ---------------------------------------------------------
st.subheader("Customer")

col1, col2, col3, col4 = st.columns(4)

with col1:
    age = st.number_input(
        "Age",
        min_value=0,
        max_value=120,
        value=30,
        step=1,
    )

with col2:
    income = st.number_input(
        "Annual Income ($)",
        min_value=0.0,
        value=50000.0,
        step=1000.0,
        format="%.0f",
    )

with col3:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Other"],
    )

with col4:
    city_type = st.selectbox(
        "City Type",
        ["Urban", "Suburban", "Rural"],
    )

# ---------------------------------------------------------
# Driving & vehicle
# ---------------------------------------------------------
st.subheader("Driving & Vehicle")

col1, col2, col3, col4 = st.columns(4)

with col1:
    commute = st.number_input(
        "Daily Commute (km)",
        min_value=0.0,
        value=20.0,
        step=1.0,
    )

with col2:
    cars_owned = st.number_input(
        "Cars Owned",
        min_value=0,
        value=1,
        step=1,
    )

with col3:
    current_car = st.selectbox(
        "Current Car",
        ["Sedan", "SUV", "Hatchback", "Truck"],
    )

with col4:
    range_anxiety = st.selectbox(
        "Range Anxiety",
        ["Low", "Medium", "High"],
    )

# ---------------------------------------------------------
# Charging & EV considerations
# ---------------------------------------------------------
st.subheader("Charging & EV Considerations")

col1, col2, col3, col4 = st.columns(4)

with col1:
    charging_stations_home = st.number_input(
        "Stations Near Home",
        min_value=0,
        value=1,
        step=1,
    )

with col2:
    charging_stations_work = st.number_input(
        "Stations Near Work",
        min_value=0,
        value=1,
        step=1,
    )

with col3:
    home_charging = st.selectbox(
        "Home Charging",
        ["Yes", "No"],
    )

with col4:
    subsidy = st.selectbox(
        "Subsidy Available",
        ["Yes", "No"],
    )

# ---------------------------------------------------------
# Environmental concern
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    env_concern = st.number_input(
        "Environmental Concern",
        min_value=0.0,
        value=5.0,
        step=0.1,
    )

# ---------------------------------------------------------
# Custom features
# ---------------------------------------------------------
is_30k_spike = int(income == 30000.0)
is_millionaire_cliff = int(income >= 170537.0)
is_dead_zone = int((income >= 38000.0) & (income <= 42000.0))

# ---------------------------------------------------------
# Model input
# ---------------------------------------------------------
input_data = pd.DataFrame([{
    "Age": age,
    "Annual_Income_USD": income,
    "Daily_Commute_km": commute,
    "Number_of_Cars_Owned": cars_owned,
    "Charging_Stations_Near_Home": charging_stations_home,
    "Charging_Stations_Near_Work": charging_stations_work,
    "Environmental_Concern_Level": env_concern,
    "Gender": gender,
    "City_Type": city_type,
    "Current_Car_Type": current_car,
    "Home_Charging_Possible": home_charging,
    "Subsidy_Available": subsidy,
    "Range_Anxiety_Level": range_anxiety,
    "is_30k_spike": is_30k_spike,
    "is_millionaire_cliff": is_millionaire_cliff,
    "is_dead_zone": is_dead_zone,
}])

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
st.divider()

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict = st.button(
        "🔋 Predict",
        type="primary",
        use_container_width=True,
    )

if predict:
    probs = model.predict_proba(input_data)
    prob_no = probs[0, 0]
    prob_yes = probs[0, 1]

    if prob_yes >= prob_no:
        result = "Will Buy an EV"
        probability = prob_yes
        st.success("### ⚡ Will Buy an EV")
    else:
        result = "Will Not Buy an EV"
        probability = prob_no
        st.info("### 🚗 Will Not Buy an EV")

    # Large, easy-to-see result
    st.metric(
        label="Model Confidence",
        value=f"{probability:.1%}",
    )