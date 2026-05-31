import joblib
import pandas as pd


class PredictionPipeline:

    def __init__(self):

        self.model = joblib.load(
            "models/diabetes_model.pkl"
        )

        self.preprocessor = joblib.load(
            "models/preprocessor.pkl"
        )

        self.label_encoder = joblib.load(
            "models/label_encoder.pkl"
        )

    def predict(self, input_df):

        expected_cols = (
            self.preprocessor.feature_names_in_
        )

        for col in expected_cols:

            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[
            expected_cols
        ]

        # Transform
        data = self.preprocessor.transform(
            input_df
        )

        # Predict
        pred = self.model.predict(data)[0]

        diabetes_type = (
            self.label_encoder
            .inverse_transform([pred])[0]
        )

        # Confidence
        prob = (
            self.model
            .predict_proba(data)[0]
        )

        confidence = round(
            max(prob) * 100,
            2
        )

        return {
            "label": int(pred),
            "type": diabetes_type,
            "confidence": confidence
        }