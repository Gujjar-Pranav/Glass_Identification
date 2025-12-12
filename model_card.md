# Model Card — Glass Identification Stacking Ensemble

**Version:** 1.0  
**Author:** Pranav Gujjar  
**Last Updated:** 2025-12-10  

---

## 1. Model Overview

### Model Name
Glass Identification — Stacking Ensemble Classifier

### Model Type
Supervised multi-class classification  
Target classes: **1, 2, 3, 5, 6, 7**

### Model Architecture
Stacking ensemble consisting of:
- Base learners:
  - Random Forest (tuned)
  - Gradient Boosting
  - AdaBoost
- Meta-learner:
  - Logistic Regression

### Frameworks and Tools
- Python 3.12
- scikit-learn
- pandas, numpy

### Model Artifacts
- `stacking_model.joblib` — trained ensemble model
- `scaler.joblib` — fitted `StandardScaler`
- `feature_columns.joblib` — ordered feature schema

---

## 2. Intended Use

### Intended Users
- Forensic analysts
- Material scientists
- Manufacturing and quality-control engineers
- ML practitioners and students

### Intended Purpose
- Predict glass type from chemical composition for:
  - preliminary forensic analysis
  - quality control workflows
  - educational and research use cases

### Not Intended For
- Legal evidence without expert verification
- Safety-critical or autonomous decision-making
- Medical or regulatory applications

---

## 3. Training Data

### Dataset
UCI Glass Identification Dataset

### Dataset Characteristics
- 214 samples
- 9 numerical input features
- 6 target classes

### Preprocessing
- Zero-value correction using median imputation
- Duplicate and constant column removal
- IQR-based winsorization for outlier control
- Feature scaling using `StandardScaler`
- Feature engineering:
  - squared refractive index (`RI²`)
  - quantile-based RI bins
  - interaction term (`Al × Si`)
  - chemical ratios (`Na/Ca`, `Mg/Ca`)

All preprocessing steps are reused during inference.

---

## 4. Evaluation Metrics

### Overall Performance
- Accuracy: **> 0.90**
- Macro F1-score: strong across most classes
- Precision: highest for Types 1, 2, and 7
- Recall: slightly lower for Type 5 due to class overlap

### Error Characteristics
- Types 1 and 2 show high separability
- Type 5 exhibits overlap with neighboring classes
- Ensemble model improves robustness compared to single classifiers

---

## 5. Limitations

- Small dataset size (214 samples) limits generalization.
- Class imbalance affects minority classes (Types 5 and 6).
- Requires accurate chemical composition measurements.
- Model trained on historical data may not fully represent modern glass formulations.

---

## 6. Ethical Considerations

- Predictions should support, not replace, expert judgment.
- Misclassification risk exists, particularly for minority classes.
- Results must be interpreted by domain experts in forensic contexts.
- Ensemble complexity reduces interpretability compared to simpler models.

---

## 7. Input and Output Specification

### Input Features
| Feature | Description |
|-------|-------------|
| RI | Refractive index |
| Na | Sodium content |
| Mg | Magnesium oxide |
| Al | Aluminium oxide |
| Si | Silicon oxide |
| K | Potassium |
| Ca | Calcium |
| Ba | Barium |
| Fe | Iron |

### Output
```json
{
  "predicted_type": <int>
}
Class probabilities are available internally via predict_proba and used in the Streamlit dashboard.

8. Inference Pipeline
Prediction flow:

Receive raw JSON input

Validate input schema

Apply preprocessing and feature engineering

Align features to training schema

Scale inputs using persisted scaler

Predict using stacking ensemble

Return predicted class

Inference is served via a FastAPI backend.

9. Deployment
Backend:

FastAPI service (glass-api)

Loads model artifacts at startup

Exposes /predict and /health endpoints

Frontend:

Streamlit application (glass-app)

Sends inference requests to backend

Displays predictions and confidence scores

Deployment:

Dockerized services orchestrated locally via Docker Compose

CI/CD pipeline builds and publishes images to GitHub Container Registry

10. Versioning
Version	Description
1.0	Initial stacking ensemble with preprocessing, API, UI, and Docker-based deployment

11. Future Improvements
Add experiment tracking and model versioning

Integrate feature attribution methods (e.g., SHAP)

Implement data drift monitoring

Expand dataset with modern glass compositions

Support batch inference alongside REST API

Summary
This model provides a high-performing and reproducible solution for glass type classification.
It demonstrates:

disciplined ensemble modeling

consistent training and inference pipelines

production-oriented deployment

responsible usage considerations