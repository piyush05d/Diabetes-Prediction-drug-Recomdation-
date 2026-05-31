from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainingPipeline:

    def start(self):

        ingestion = DataIngestion()
        diabetes_df, _ = ingestion.initiate_data_ingestion()

        transformation = DataTransformation()
        X_train, X_test, y_train, y_test, _ = \
            transformation.initiate_transformation(diabetes_df)

        trainer = ModelTrainer()
        trainer.train(X_train, X_test, y_train, y_test)

        print("🎉 Training Done")