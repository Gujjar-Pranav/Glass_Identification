# Glass Identification – End-to-End Machine Learning System  
**Portfolio Case Study**

**Author:** Pranav Gujjar  
**Tech Stack:** Python, scikit-learn, FastAPI, Streamlit, Docker, GitHub Actions

---

## 1. Problem Overview

Glass classification plays a critical role in forensic science, manufacturing quality control, and material analysis.  
The objective of this project is to predict the **type of glass** based on its chemical composition using a **production-oriented, end-to-end machine learning system**.

The dataset is derived from the UCI Glass Identification Dataset and includes:
- Refractive Index (RI)
- Chemical oxides: Na, Mg, Al, Si, K, Ca, Ba, Fe
- Target glass type: 1, 2, 3, 5, 6, 7

This project intentionally goes beyond model training to demonstrate **full ML system engineering**, including:
- automated preprocessing and feature engineering
- ensemble-based modeling
- real-time inference
- UI integration
- containerization
- CI/CD automation

---

## 2. Project Objectives

### Primary Goals
- Build a robust and accurate glass classification model.
- Design a reproducible preprocessing and inference pipeline.
- Expose predictions through a FastAPI-based service.
- Provide an interactive dashboard for non-technical users.
- Package the system using Docker for consistent execution.
- Automate validation and builds using GitHub Actions.

### Why This Project Stands Out
Most ML projects stop at experimentation.  
This project demonstrates how models are **engineered, served, deployed, and maintained** in real-world environments.

---

## 3. Data Pipeline & Feature Engineering

### Data Cleaning
- Removed duplicate records and constant columns.
- Replaced invalid zero values using median imputation.
- Applied IQR-based winsorization to control extreme outliers.

### Feature Engineering
- Non-linear refractive index features (`RI²`, quantile-based RI bins).
- Interaction term between aluminium and silicon (`Al × Si`).
- Chemical ratio features (`Na/Ca`, `Mg/Ca`).
- Feature scaling using `StandardScaler`.

### Preprocessing Design
All transformations are implemented as reusable modules:
- `src/data_prep.py`
- `src/features.py`
- `src/infer.py`

The preprocessing pipeline is **identical for training and inference**, enforced through persisted artifacts:
- `models/scaler.joblib`
- `models/feature_columns.joblib`

---

## 4. Model Development

### Models Trained
- Random Forest (baseline)
- Tuned Random Forest
- Bagging Classifier
- AdaBoost
- Gradient Boosting
- **Stacking Ensemble (final model)**

### Model Selection
The stacking ensemble was selected due to its superior generalization performance and robustness across classes.  
It combines complementary learners to reduce bias and variance.

### Final Model Performance
- Test accuracy consistently above **0.90**
- Strong macro F1-score across glass types
- Improved handling of overlapping classes compared to single models

---

## 5. FastAPI Backend – Real-Time Inference

The backend serves as a **stateless inference layer**.

### Endpoints
- `GET /health` – service health check
- `POST /predict` – returns predicted glass type

### Key Characteristics
- Loads trained artifacts once at startup.
- Automatically applies preprocessing and feature engineering.
- Scales features and performs ensemble inference.
- Responds in milliseconds.
- Designed to be lightweight and container-ready.

---

## 6. Streamlit Frontend – Interactive Dashboard

The Streamlit application provides an accessible interface for model interaction.

### Capabilities
- Adjustable sliders for all nine input features.
- Explicit predict and reset actions.
- Prediction summary with class probabilities.
- Rolling history of recent predictions with timestamps.
- Built-in backend health monitoring.

### Dashboard Sections
- Interactive feature exploration.
- System architecture explanation.
- Model performance insights and conclusions.

This design enables non-technical stakeholders to interact with the ML system confidently.

---

## 7. Docker Deployment Architecture

The system is deployed locally using Docker Compose with two services:

### Services
1. **glass-api**
   - FastAPI inference backend
   - Runs on `http://localhost:8000`

2. **glass-app**
   - Streamlit frontend
   - Communicates with backend via Docker network
   - Runs on `http://localhost:8501`

This separation mirrors production deployment patterns and ensures reproducibility.

---

## 8. CI/CD Pipeline (GitHub Actions)

The CI/CD pipeline automates validation and container builds.

### Automated Steps
- Python syntax validation.
- Build FastAPI Docker image and push to GHCR.
- Build Streamlit Docker image and push to GHCR.
- Tag images with `latest` and commit SHA.

### Benefits
- Reproducible builds.
- Versioned container images.
- No dependency on local development environments.
- Production-style workflow for ML systems.

---

## 9. Results & Insights

### Key Findings
- Refractive Index, Sodium, and Calcium are the strongest predictors.
- Glass types 1, 2, and 7 show high separability.
- Types 5 and 6 exhibit overlap, effectively handled by the ensemble model.

### Practical Value
- Enables rapid forensic material classification.
- Standardizes prediction workflows.
- Easily deployable ML system using Docker.
- User-friendly interface for domain experts.

---

## 10. Lessons Learned

### Technical
- Designing reusable preprocessing pipelines.
- Enforcing training–inference parity.
- Serving ML models with FastAPI.
- Packaging multi-service ML systems with Docker.
- Implementing CI/CD for ML workflows.

### Non-Technical
- Importance of explainability and trust for stakeholders.
- Balancing model performance and interpretability.
- Building interfaces that support confident decision-making.

---

## 11. Future Enhancements

- Model versioning and experiment tracking.
- Automated retraining workflows.
- Monitoring and observability.
- Cloud deployment (AWS, GCP, DigitalOcean, Fly.io).
- Data drift detection.

---

## 12. Conclusion

This project demonstrates **real-world machine learning system engineering**, not just model development.  
It integrates data processing, ensemble modeling, inference services, UI design, containerization, and CI/CD into a cohesive and production-ready architecture.

It serves as a strong portfolio example of applied ML engineering and MLOps practices.
