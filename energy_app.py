
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="PJM Energy Consumption Forecast",
    page_icon="⚡"
)

# Load model
import gzip
import shutil
import joblib
import os
if not os.path.exists("random_forest_model.pkl"):
    with gzip.open("random_forest_model.pkl.gz","rb") as f_in:
        with open("random_forest_model.pkl","wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    model = joblib.load("random_forest_model.pkl")

# Load forecast
forecast = pd.read_csv("30_day_energy_forecast.csv")
forecast["Datetime"] = pd.to_datetime(forecast["Datetime"])

# Title
st.title("⚡ PJM Hourly Energy Consumption Forecast")

st.write(
    "This application uses a Random Forest model "
    "to forecast hourly energy consumption for the next 30 days."
)

# 30-Day Forecast
st.subheader("📈 30-Day Energy Consumption Forecast")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(
    forecast["Datetime"],
    forecast["Predicted_Energy"]
)
ax.set_xlabel("Date")
ax.set_ylabel("Predicted Energy")
ax.set_title("Next 30 Days Energy Forecast")
plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

# Forecast Data
st.subheader("📊 Forecast Data")

st.markdown(
    forecast.head(20).to_html(index=False),
    unsafe_allow_html=True
)

# Individual Prediction
st.subheader("🔮 Predict Energy Consumption")

hour = st.number_input(
    "Hour",
    min_value=0,
    max_value=23,
    value=12
)

day_of_week = st.number_input(
    "Day of Week",
    min_value=0,
    max_value=6,
    value=2
)

month = st.number_input(
    "Month",
    min_value=1,
    max_value=12,
    value=6
)

if st.button("Predict Energy Consumption"):

    input_data = pd.DataFrame({
        "Hour": [hour],
        "DayOfWeek": [day_of_week],
        "Month": [month]
    })

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Energy Consumption: {prediction[0]:.2f} MW"
    )
