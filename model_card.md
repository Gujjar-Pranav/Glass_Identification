Model Card: Glass Identification Stacking Ensemble Model

Version: 1.0
Author: Pranav Gujjar
Last Updated: 2025-12-10

📌 1. Model Details
Model Name

Glass Identification – Stacking Ensemble Classifier

Model Type

Supervised multi-class classifier (Classes: 1, 2, 3, 5, 6, 7)

Architecture

A meta-learner (Logistic Regression) stacked over multiple base models:

Random Forest

Gradient Boosting

AdaBoost

Bagging Classifier

Frameworks Used

Python 3.12

Scikit-Learn

NumPy / Pandas

Model Artifacts

stacking_model.joblib – trained ensemble model

scaler.joblib – standard scaler for inference

feature_columns.joblib – final feature schema

📊 2. Intended Use
✔ Intended Users

Forensic analysts

Material scientists

Manufacturing engineers

ML practitioners evaluating benchmark models

✔ Intended Purpose

Predict glass type based on its chemical composition for:

Preliminary forensic investigations

Quality control in glass manufacturing

Educational ML demonstrations

Research experiments

⚠ Not intended for:

Legal evidence without expert supervision

High-risk autonomous decision-making

Medical or safety-critical workflows

🧪 3. Training Data
Dataset

UCI Glass Identification Dataset

Samples

214 rows

9 numerical input features

6 target classes

Data Preprocessing

✔ Winsorization (outlier smoothing)
✔ Zero correction using median imputation
✔ Duplicate removal
✔ Scaling using StandardScaler
✔ Feature engineering (RI², ratios, interactions, RI bins)

🧪 4. Evaluation Metrics
Overall Model Performance
Metric	Score
Accuracy	0.90+
Macro F1-score	High across most classes
Precision	Strong for Type 1, 2, 7
Recall	Slight drop for Type 5
Confusion Matrix Summary

Types 1 & 2 predicted highly accurately

Type 5 shows some overlap

Type 7 excellent recall

🎯 5. Limitations
🔸 Dataset is small

Only 214 samples → risk of overfitting, limited real-world variability.

🔸 Chemical measurements required

Model only works when exact oxide composition is available.

🔸 Class imbalance

Some classes have significantly fewer samples (e.g., Type 5, 6).

🔸 Limited generalization

Model trained on historical UCI dataset → may not fully represent modern industry glass compositions.

⚠ 6. Ethical Considerations

Should not be used as standalone forensic evidence.
Predictions must support, not replace, laboratory analysis.

Risk of misclassification exists for minority classes (e.g., Type 5 → Type 3).
Critical decisions must involve human review.

Interpretability:
Ensemble models can be complex; decisions may require domain expert validation.

🔍 7. Model Input/Output Specification
Input Fields
Feature	Description
RI	Refractive Index
Na	Sodium content
Mg	Magnesium oxide
Al	Aluminium oxide
Si	Silicon oxide
K	Potassium
Ca	Calcium
Ba	Barium
Fe	Iron
Output
{
  "predicted_type": <int>, 
  "probabilities": [...],
  "classes": [1,2,3,5,6,7]
}

🧪 8. Inference Pipeline

The following transformations occur during prediction:

Receive raw JSON input

Validate schema

Apply feature engineering

Apply scaling

Pass to stacking ensemble

Return predicted class + probabilities

Backend inference is handled via FastAPI.

🐳 9. Deployment

The model is deployed using two Docker containers:

Backend (FastAPI) – glass-api

Loads model at startup

Provides /predict and /health endpoints

Frontend (Streamlit) – glass-app

Sends requests to backend

Presents results interactively

CI/CD pipeline automatically builds & pushes images to GHCR using GitHub Actions.

🔄 10. Versioning
Version	Changes
1.0	Initial stacking ensemble, preprocessing pipeline, Docker deployment

🚀 11. Recommendations for Future Versions

Integrate MLflow for tracking

Use SHAP or feature importance plots in UI

Add data drift detection

Expand dataset with modern glass compositions

Convert to a REST + batch inference hybrid

🏁 Summary

This model provides a high-performing, production-ready solution for glass type classification.
It showcases:

Strong ML modeling

Clean engineering practices

End-to-end deployment

Responsible use guidelines