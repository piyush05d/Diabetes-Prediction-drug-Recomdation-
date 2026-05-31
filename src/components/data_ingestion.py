import pandas as pd
import os


class DataIngestion:

    def initiate_data_ingestion(self):

        print("📥 Loading datasets...")

        diabetes_path = os.path.join(
            "data",
            "true_diabetes_dataset.csv"
        )

        medicine_path = os.path.join(
            "data",
            "medicine.csv"
        )

        diabetes_df = pd.read_csv(diabetes_path)

        medicine_df = pd.read_csv(medicine_path)

        print("✅ Datasets loaded successfully")

        return diabetes_df, medicine_df