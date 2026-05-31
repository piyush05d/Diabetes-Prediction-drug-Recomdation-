import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelEncoder
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


class DataTransformation:

    def initiate_transformation(self, df):

        print("🔄 Transforming dataset...")

        # Clean columns
        df.columns = [
            c.strip().lower() for c in df.columns
        ]

        # ---------------- TARGET ----------------
        target_col = "diabetes_type_name"

        if target_col not in df.columns:
            raise Exception(
                f"❌ {target_col} not found"
            )

        # Encode target
        le = LabelEncoder()

        df["diabetes_label"] = le.fit_transform(
            df[target_col]
        )

        # Save encoder
        os.makedirs("models", exist_ok=True)

        joblib.dump(
            le,
            "models/label_encoder.pkl"
        )

        # ---------------- IMPORTANT FEATURES ----------------
        selected_cols = [
            "age",
            "bmi",
            "glucose",
            "hba1c",
            "gender",
            "systolic_bp",
            "diastolic_bp"
        ]

        selected_cols = [
            c for c in selected_cols
            if c in df.columns
        ]

        X = df[selected_cols]

        y = df["diabetes_label"]

        print("📊 Class Distribution:")
        print(y.value_counts())

        # ---------------- NUMERIC / CATEGORICAL ----------------
        num_cols = X.select_dtypes(
            include=["int64", "float64"]
        ).columns

        cat_cols = X.select_dtypes(
            include=["object"]
        ).columns

        # ---------------- PIPELINES ----------------
        num_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ])

        cat_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ])

        preprocessor = ColumnTransformer([
            (
                "num",
                num_pipeline,
                num_cols
            ),
            (
                "cat",
                cat_pipeline,
                cat_cols
            )
        ])

        # ---------------- SPLIT ----------------
        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )
        )

        # ---------------- TRANSFORM ----------------
        X_train = preprocessor.fit_transform(
            X_train
        )

        X_test = preprocessor.transform(
            X_test
        )

        # Save preprocessor
        joblib.dump(
            preprocessor,
            "models/preprocessor.pkl"
        )

        print("✅ Transformation completed")

        return (
            X_train,
            X_test,
            y_train,
            y_test,
            preprocessor
        )