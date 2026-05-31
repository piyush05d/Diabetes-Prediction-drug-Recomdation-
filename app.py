import streamlit as st
import pandas as pd
import joblib
import json

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.components.recommender import DrugRecommender


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    layout="centered"
)

st.title("🩺 Diabetes Prediction & Smart Drug Recommendation")
st.write("Enter patient clinical details below")


# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=22.0
    )

    glucose = st.number_input(
        "Glucose Level",
        min_value=50.0,
        max_value=400.0,
        value=100.0
    )

    hba1c = st.number_input(
        "HbA1c",
        min_value=3.0,
        max_value=15.0,
        value=5.5
    )

with col2:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    systolic_bp = st.number_input(
        "Systolic BP",
        min_value=80,
        max_value=250,
        value=120
    )

    diastolic_bp = st.number_input(
        "Diastolic BP",
        min_value=40,
        max_value=150,
        value=80
    )


# ---------------- PREDICTION BUTTON ----------------
if st.button("🔍 Predict"):

    try:

        # ---------------- LOAD PREPROCESSOR ----------------
        preprocessor = joblib.load(
            "models/preprocessor.pkl"
        )

        expected_cols = list(
            preprocessor.feature_names_in_
        )

        # ---------------- CREATE INPUT ----------------
        input_dict = {
            col: 0 for col in expected_cols
        }

        # Fill actual user inputs
        input_dict.update({
            "age": age,
            "bmi": bmi,
            "glucose": glucose,
            "hba1c": hba1c,
            "gender": gender,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp
        })

        input_df = pd.DataFrame([input_dict])

        # Ensure correct column order
        input_df = input_df[expected_cols]

        # ---------------- PREDICTION ----------------
        predictor = PredictionPipeline()

        result = predictor.predict(input_df)

        st.success(
            f"✅ Predicted Diabetes Type: {result['type']}"
        )

        st.info(
            f"Prediction Confidence: {result['confidence']}%"
        )

        # ---------------- RECOMMENDATION ----------------
        recommender = DrugRecommender()

        best = recommender.recommend_best(
            result["type"],
            hba1c,
            glucose,
            bmi
        )

        st.subheader("💊 Best Recommended Treatment")

        st.write(
            "**Medicine:**",
            best["medicine"]
        )

        st.write(
            "**Dosage:**",
            best["dosage"]
        )

        st.write(
            "**Diet:**",
            best["diet"]
        )

        st.write(
            "**Precautions:**",
            best["precautions"]
        )

        # ---------------- MODEL METRICS ----------------
        st.subheader("📊 Model Performance")

        try:

            with open(
                "models/metrics.json",
                "r"
            ) as f:

                metrics = json.load(f)

            st.write(
                f"**Best Model:** {metrics['best_model']}"
            )

            st.write(
                f"**Accuracy:** {metrics['accuracy']:.4f}"
            )

            st.write(
                f"**Cross Validation Score:** {metrics['cv_score']:.4f}"
            )

        except:
            st.warning(
                "⚠ Metrics file not found"
            )

    except Exception as e:

        st.error(f"❌ Error: {e}")