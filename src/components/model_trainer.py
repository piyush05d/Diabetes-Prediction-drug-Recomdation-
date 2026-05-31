import os
import json
import joblib
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score
)

warnings.filterwarnings("ignore")


class ModelTrainer:

    def train(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        print("🚀 Training Models...\n")

        models = {

            "Logistic Regression":
            LogisticRegression(
                max_iter=2000,
                C=0.3,
                class_weight="balanced"
            ),

            "AdaBoost":
            AdaBoostClassifier(),

            "XGBoost":
            XGBClassifier(
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1,
                reg_lambda=1,
                eval_metric="mlogloss"
            )
        }

        params = {

            "Logistic Regression": {
                "C": [0.1, 0.3, 0.5]
            },

            "AdaBoost": {
                "n_estimators": [50, 100]
            },

            "XGBoost": {
                "n_estimators": [100, 200]
            }
        }

        best_model = None
        best_score = 0
        best_name = ""

        results = {}

        # ---------------- TRAIN LOOP ----------------
        for name, model in models.items():

            print(f"🔍 Tuning {name}...")

            grid = GridSearchCV(
                estimator=model,
                param_grid=params[name],
                cv=5,
                scoring="accuracy",
                n_jobs=-1
            )

            grid.fit(
                X_train,
                y_train
            )

            best_estimator = (
                grid.best_estimator_
            )

            # Cross validation
            cv_score = cross_val_score(
                best_estimator,
                X_train,
                y_train,
                cv=5
            ).mean()

            # Prediction
            y_pred = best_estimator.predict(
                X_test
            )

            acc = accuracy_score(
                y_test,
                y_pred
            )

            results[name] = acc

            print(
                f"\n{name} Accuracy: {acc}"
            )

            print(
                f"{name} CV Score: {cv_score}"
            )

            print(
                classification_report(
                    y_test,
                    y_pred
                )
            )

            # Track best
            if acc > best_score:

                best_score = acc

                best_model = best_estimator

                best_name = name

        # ---------------- SAVE ----------------
        os.makedirs(
            "models",
            exist_ok=True
        )

        joblib.dump(
            best_model,
            "models/diabetes_model.pkl"
        )

        metrics = {
            "best_model": best_name,
            "accuracy": best_score,
            "cv_score": float(cv_score)
        }

        with open(
            "models/metrics.json",
            "w"
        ) as f:

            json.dump(metrics, f)

        print("\n🏆 Best Model:", best_name)

        return best_model