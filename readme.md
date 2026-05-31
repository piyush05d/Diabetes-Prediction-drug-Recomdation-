first step
run this code on terminal
python main.py

then run this
 python -m streamlit run app.py
 

 # 🩺 Diabetes Prediction & Smart Drug Recommendation System

A Machine Learning-based healthcare application that predicts the type of diabetes and recommends the most suitable medicine based on patient clinical data.

---

## 🚀 Features

- 🔍 Predicts Diabetes Type (0 / 1 / 2)
- 💊 Recommends **best medicine** based on patient condition
- 📊 Displays model performance in UI
- 🧠 Uses multiple ML models:
  - Logistic Regression
  - AdaBoost
  - XGBoost
- ⚙️ Hyperparameter tuning using GridSearchCV
- 📈 Cross-validation for robust evaluation
- 📄 Classification report & confusion matrix
- 🌐 Interactive dashboard using Streamlit

---

## 📁 Project Structure
project/
│
├── data/
│   ├── unified_diabetes_75k.csv
│   └── medicine.csv
│
├── models/
│   ├── diabetes_model.pkl
│   ├── preprocessor.pkl
│   └── metrics.json
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── recommender.py
│   │
│   └── pipeline/
│       ├── training_pipeline.py
│       └── prediction_pipeline.py
│
├── app.py
├── main.py
└── README.md
---## ⚙️ Installation### 1. Clone Repository```bashgit clone <your-repo-url>cd project
2. Install Dependencies
pip install pandas scikit-learn xgboost streamlit joblib

▶️ Usage
🔹 Step 1: Train the Model
python main.py
🔹 Step 2: Run the Streamlit App
python -m streamlit run app.py

🧠 Machine Learning Pipeline
1. Data Ingestion


Loads diabetes dataset and medicine dataset


2. Data Transformation


Handles missing values


Encodes categorical variables


Scales numerical features


Removes irrelevant/leakage columns


3. Model Training


Trains multiple models:


Logistic Regression


AdaBoost


XGBoost




Uses GridSearchCV for hyperparameter tuning


Evaluates using:


Accuracy


Cross-validation score


Classification report


Confusion matrix




Selects and saves the best model


4. Prediction


Uses trained model + preprocessor


Predicts diabetes type


5. Recommendation System


Filters medicines based on prediction


Uses:


HbA1c


Glucose


BMI




Applies scoring system to recommend best medicine



📊 Models Used
ModelDescriptionLogistic RegressionBaseline linear modelAdaBoostBoosting ensembleXGBoostAdvanced gradient boosting

💊 Recommendation Logic


Predict diabetes type


Evaluate patient condition:


HbA1c level


Glucose level


BMI




Score medicines and select best match



🖥️ Streamlit Dashboard
Input Features:


Age


BMI


Glucose Level


HbA1c


Gender


Hypertension


Heart Disease


Smoking History


Output:


Predicted Diabetes Type


Best Medicine Recommendation


Dosage, Diet, Precautions


Model Metrics:


Accuracy


Cross-validation score





📊 Example Output
Predicted: Type 2 Diabetes💊 Best Recommended Treatment:Medicine: MetforminDosage: 500mg dailyDiet: Low sugar dietPrecautions: Avoid alcohol📊 Model Performance:Best Model: XGBoostAccuracy: 0.87Cross Validation: 0.85

🧠 Key Concepts Used


Machine Learning Pipelines


Feature Engineering


Model Selection & Tuning


Cross Validation


Classification Models


Streamlit Deployment



⚠️ Important Notes


Ensure medicine.csv contains:
diabetes_type, medicine, dosage, diet, precautions


Run training before launching UI



🎯 Future Improvements


📊 Visualization dashboard (graphs)


🧠 Explainable AI (SHAP)


📄 PDF report generation


🤖 AutoML integration



👨‍💻 Author
Piyush 

📌 License
This project is for educational and academic purposes.
---# 🔥 Why this README is strong✔ Covers full pipeline  ✔ Includes ML + UI + evaluation  ✔ GitHub-ready  ✔ Viva-ready  ✔ Clean structure  ---# 🚀 If you want nextI can also give:- `requirements.txt`- GitHub project description (short)- PPT for your viva- Demo script explanationJust say 👍