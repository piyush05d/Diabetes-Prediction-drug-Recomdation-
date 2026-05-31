import pandas as pd


class DrugRecommender:

    def __init__(self):

        self.df = pd.read_csv(
            "data/medicine.csv"
        )

        self.df.columns = [
            c.strip().lower()
            for c in self.df.columns
        ]

        self.df["diabetes_type"] = (
            self.df["diabetes_type"]
            .astype(str)
            .str.lower()
        )

    def recommend_best(
        self,
        diabetes_type,
        hba1c,
        glucose,
        bmi
    ):

        dtype = diabetes_type.lower()

        df = self.df[
            self.df["diabetes_type"]
            .str.contains(dtype)
        ]

        # Fallback
        if df.empty:
            df = self.df.copy()

        # ---------------- SCORING ----------------
        scores = []

        for _, row in df.iterrows():

            score = 0

            # HbA1c
            if hba1c >= 8:
                score += 3
            elif hba1c >= 6.5:
                score += 2
            else:
                score += 1

            # Glucose
            if glucose > 180:
                score += 2
            elif glucose > 120:
                score += 1

            # BMI
            if bmi > 30:
                score += 2
            elif bmi > 25:
                score += 1

            scores.append(score)

        df["score"] = scores

        best = (
            df.sort_values(
                by="score",
                ascending=False
            )
            .iloc[0]
        )

        return {
            "medicine": best.get(
                "medicine",
                "N/A"
            ),

            "dosage": best.get(
                "dosage",
                "N/A"
            ),

            "diet": best.get(
                "diet",
                "N/A"
            ),

            "precautions": best.get(
                "precautions",
                "N/A"
            )
        }