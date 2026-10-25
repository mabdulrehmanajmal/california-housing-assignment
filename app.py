
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="California Housing Predictor", page_icon="🏠")

st.title("California Housing Price Prediction")
st.write("Enter the area's housing characteristics to estimate its median house value.")

@st.cache_resource
def load_model():
    return joblib.load("california_housing_model.joblib")

model = load_model()

medinc = st.number_input("Median income (dataset units)", min_value=0.0, value=5.0)
houseage = st.number_input("Median house age (years)", min_value=1.0, value=30.0)
averooms = st.number_input("Average rooms", min_value=1.0, value=5.0)
avebedrms = st.number_input("Average bedrooms", min_value=0.1, value=1.0)
population = st.number_input("Area population", min_value=1.0, value=1000.0)
aveoccup = st.number_input("Average household occupancy", min_value=0.1, value=3.0)
latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=34.0)
longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-118.0)

if st.button("Predict House Value"):
    data = pd.DataFrame([{
        "MedInc": medinc,
        "HouseAge": houseage,
        "AveRooms": averooms,
        "AveBedrms": avebedrms,
        "Population": population,
        "AveOccup": aveoccup,
        "Latitude": latitude,
        "Longitude": longitude
    }])

    prediction = float(model.predict(data)[0])
    st.success(f"Estimated median house value: ${prediction * 100000:,.0f}")
    st.caption("This is a machine-learning estimate, not an official property valuation.")
