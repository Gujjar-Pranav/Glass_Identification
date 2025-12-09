# Model Card – Glass Identification Stacking Ensemble

## 1. Model Overview
- **Task:** Multiclass classification – predict glass type.
- **Dataset:** UCI Glass Identification dataset.
- **Final model:** Stacking Ensemble with:
  - Tuned Random Forest
  - Gradient Boosting
  - AdaBoost
  - Logistic Regression as meta-learner.

## 2. Data
- **Inputs:** Refractive index and oxide composition (Na, Mg, Al, Si, K, Ca, Ba, Fe).
- **Preprocessing:**
  - Replace zeros with median (numeric features).
  - Remove duplicates and constant columns.
  - Outlier handling via IQR-based winsorization.
  - Feature scaling via StandardScaler.
  - Class imbalance addressed with SMOTE.
- **Feature Engineering:**
  - Ratios: Na/Ca, Mg/Ca.
  - Interaction: Al × Si.
  - Non-linear: RI².
  - RI binning + one-hot encoding.

## 3. Performance
- **Metric:** Accuracy and macro F1-score on held-out test set.
- **Results:**
  - Stacking Ensemble: ~0.91 accuracy.
  - Tuned Random Forest & Gradient Boosting: ~0.86 accuracy.
  - Baseline RF: ~0.84 accuracy.
  - Bagging: ~0.77, AdaBoost: ~0.65.

## 4. Intended Use
- Educational / demonstration purposes.
- Not for critical safety or real forensic decisions.

## 5. Limitations
- Small dataset → risk of overfitting and unstable metrics.
- Synthetic data via SMOTE → distribution differs from real-world.
- No domain expert validation included.

## 6. Ethical Considerations
- Dataset is synthetic / lab-based; no direct personal data.
- If used in real forensic contexts, must be validated and audited.

## 7. Future Work
- Add XGBoost/LightGBM for comparison.
- Add SHAP-based explanations for stacking model directly.
- Deploy scalable API and monitoring for drift.
