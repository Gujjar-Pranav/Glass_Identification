# Glass Identification – Ensemble Learning Case Study

## 1. Problem Statement
The goal of this project is to classify types of glass based on their chemical composition and refractive index. This mimics a real-world forensic scenario where identifying glass type from fragments can support investigations.

## 2. Data & Challenges
- **Dataset:** UCI Glass Identification dataset.
- **Challenges:**
  - Small dataset (214 rows).
  - Class imbalance across glass types.
  - Potential outliers and noisy measurements.
  - Need for robust, generalizable models rather than a single overfitted classifier.

## 3. Approach

### 3.1 Preprocessing
- Replaced zero values in numeric features with median values.
- Removed duplicate rows and constant columns.
- Performed IQR-based winsorization to limit the influence of extreme outliers.
- Applied StandardScaler to normalize numeric features.
- Used SMOTE to oversample minority classes and mitigate class imbalance.

### 3.2 Feature Engineering
- Designed domain-inspired features:
  - Oxide ratios: Na/Ca, Mg/Ca.
  - Structural interaction term: Al × Si.
  - Non-linear transformation: RI².
  - RI binning converted to ordinal and then one-hot encoded.
- These features helped the models capture non-linear relationships and compositional structure more effectively.

### 3.3 Models & Hyperparameter Tuning
- Baseline models:
  - Random Forest, Bagging, AdaBoost, Gradient Boosting.
- Hyperparameter tuning:
  - Random Forest optimized using both GridSearchCV and RandomizedSearchCV with f1-macro scoring.
- Final ensemble:
  - Stacking Ensemble combining:
    - Tuned Random Forest
    - Gradient Boosting
    - AdaBoost
    - Logistic Regression as meta-learner.

## 4. Results

### 4.1 Model Performance (Test Set Accuracy)
- Random Forest (baseline): ~0.84
- Random Forest (tuned): ~0.86
- Bagging: ~0.77
- AdaBoost: ~0.65
- Gradient Boosting: ~0.86
- **Stacking Ensemble:** ~0.91 (best)

The stacking model achieved the highest accuracy and strong macro F1-score, with particularly good recall for minority glass types (3, 5, and 6).

### 4.2 Key Insights
- Feature engineering significantly improved model performance compared to raw features.
- Hyperparameter tuning of Random Forest yielded a measurable accuracy gain.
- Boosting and stacking techniques outperformed simple bagging and baseline models.
- The Stacking Ensemble was able to leverage complementary strengths of RF, GB, and AdaBoost.

## 5. Deployment & Visualization

### 5.1 Streamlit App
- Built an interactive web app using Streamlit:
  - Model comparison dashboard (accuracy bar chart).
  - Confusion matrix and classification report.
  - EDA: correlation heatmap & sample data view.
  - Live prediction form for entering glass composition and getting the predicted type.

### 5.2 FastAPI Service
- Implemented a FastAPI endpoint (`/predict`) that:
  - Accepts feature values as JSON.
  - Applies the full preprocessing & feature engineering pipeline.
  - Returns the predicted glass type from the stacking model.

### 5.3 Explainability with SHAP
- Used SHAP TreeExplainer on a tuned Random Forest to:
  - Generate global feature importance (summary plot).
  - Understand which features drive the predictions (e.g., RI, Ca, Na, Na/Ca ratio).

## 6. Tech Stack
- Python, pandas, numpy, scikit-learn, imbalanced-learn
- Streamlit for UI
- FastAPI + Uvicorn for API
- SHAP for model explainability
- GitHub Actions for CI
- Docker for containerization

## 7. Conclusion
This project demonstrates an end-to-end applied machine learning workflow:
- From data cleaning and feature engineering
- To model training and tuning
- To interpretability, deployment, and a user-facing interface.

The Stacking Ensemble achieved ~91% accuracy and provides a strong baseline for further experimentation with advanced gradient boosting frameworks or interpretability techniques.
