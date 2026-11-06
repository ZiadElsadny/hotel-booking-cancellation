
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Hotel Cancellation Predictor",
    page_icon="🏨",
    layout="wide"
)

st.title("🏨 Hotel Booking Cancellation Predictor")
st.write("Predict the probability of a hotel booking cancellation.")

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE_DIR, "hotel_model.pkl"))

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE_DIR, "hotel_data.csv"))

model = load_model()
df = load_data()

# Columns used during model training
excluded_columns = [
    "is_canceled",
    "reservation_status",
    "reservation_status_date",
    "booking_changes",
    "lead_time_group"
]

input_columns = [
    col for col in df.columns
    if col not in excluded_columns
]

# Feature engineering used during training
st.subheader("Enter Booking Details")

input_data = {}

with st.form("booking_form"):
    cols = st.columns(3)

    for i, col in enumerate(input_columns):
        with cols[i % 3]:

            if pd.api.types.is_numeric_dtype(df[col]):
                median_value = df[col].median()

                if pd.isna(median_value):
                    median_value = 0

                input_data[col] = st.number_input(
                    col.replace("_", " ").title(),
                    value=float(median_value)
                )

            else:
                options = (
                    df[col]
                    .dropna()
                    .astype(str)
                    .value_counts()
                    .head(30)
                    .index.tolist()
                )

                if options:
                    input_data[col] = st.selectbox(
                        col.replace("_", " ").title(),
                        options
                    )
                else:
                    input_data[col] = st.text_input(
                        col.replace("_", " ").title()
                    )

    submitted = st.form_submit_button("Predict Cancellation")

if submitted:

    input_df = pd.DataFrame([input_data])

    # Feature engineering
    input_df["total_guests"] = (
        input_df["adults"]
        + input_df["children"].fillna(0)
        + input_df["babies"]
    )

    input_df["total_nights"] = (
        input_df["stays_in_weekend_nights"]
        + input_df["stays_in_week_nights"]
    )

    # Match training columns and order
    input_df = input_df[input_columns]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    st.metric(
        "Cancellation Probability",
        f"{probability * 100:.2f}%"
    )

    if prediction == 1:
        st.warning("The model predicts that this booking may be canceled.")
    else:
        st.success("The model predicts that this booking may not be canceled.")

    st.progress(float(probability))
