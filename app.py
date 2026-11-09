
import streamlit as st
import pandas as pd
import joblib

# Page settings
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# Application title
st.title("🎓 Student Performance Prediction System")

st.write(
    "A machine learning application that predicts "
    "whether students may be academically at risk."
)

st.info(
    "This application is for educational demonstration "
    "only. Predictions should not be the sole basis "
    "for real academic decisions."
)

# Load the trained model
artifact = joblib.load("student_model.pkl")

model = artifact["model"]
expected_columns = artifact["feature_columns"]

# Upload section
st.header("Upload Student Dataset")

uploaded_file = st.file_uploader(
    "Choose a student CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Student Data")
    st.dataframe(data.head(10))

    # Remove G3 if included
    if "G3" in data.columns:
        data = data.drop(columns=["G3"])

    # Validate required columns
    missing = [
        col for col in expected_columns
        if col not in data.columns
    ]

    if missing:

        st.error(
            "Missing required columns: "
            + ", ".join(missing)
        )

    else:

        # Prepare uploaded information
        X_new = data[expected_columns].copy()

        # Predict student risk
        predictions = model.predict(X_new)

        probabilities = model.predict_proba(X_new)[:, 1]

        # Create results table
        results = data.copy()

        results["Prediction"] = [
            "AT RISK" if p == 1
            else "NOT AT RISK"
            for p in predictions
        ]

        results["At Risk Probability (%)"] = (
            probabilities * 100
        ).round(2)

        # Display results
        st.subheader("Prediction Results")

        st.dataframe(
            results,
            use_container_width=True
        )

        # Summary
        total = len(results)

        at_risk = int((predictions == 1).sum())

        not_at_risk = int((predictions == 0).sum())

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Students", total)
        col2.metric("AT RISK", at_risk)
        col3.metric("NOT AT RISK", not_at_risk)

        # Download results
        csv = results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Prediction Results",
            data=csv,
            file_name="student_predictions.csv",
            mime="text/csv"
        )
