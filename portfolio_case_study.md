Glass Identification – End-to-End Machine Learning System
Portfolio Case Study

Author: Pranav Gujjar
Tech Stack: Python, Scikit-Learn, FastAPI, Streamlit, Docker, GitHub Actions, CI/CD

🔍 1. Problem Overview

Glass classification is essential in forensic analysis, manufacturing quality control, and material science.
The goal of this project is to predict the type of glass based on its chemical composition using a complete end-to-end ML system.

The raw UCI dataset includes:

Refractive index (RI)

Chemical oxides (Na, Mg, Al, Si, K, Ca, Ba, Fe)

Target glass type (1, 2, 3, 5, 6, 7)

This project goes beyond model training — it includes:

✔ Fully automated preprocessing
✔ Feature engineering
✔ Multiple ML models + stacking ensemble
✔ FastAPI backend for inference
✔ Interactive Streamlit UI
✔ Docker-based deployment
✔ CI/CD pipeline via GitHub Actions

🧱 2. Project Objectives
🎯 Primary Goals

Build a robust ML model to classify glass types.

Create a real-time prediction service using FastAPI.

Build an interactive web dashboard using Streamlit.

Package everything into reproducible Docker containers.

Automate build & deployment using CI/CD.

🔥 Why This Project Stands Out

Most ML projects stop after training a model.
This project demonstrates full ML system engineering, including:

Serving

Monitoring

Versioning

Deployment

UI integration

Developer workflows with CI/CD

📊 3. Data Pipeline & Feature Engineering
🧽 Data Cleaning

Removed duplicate and constant columns

Replaced invalid zero values

Median imputation

Winsorization to handle extreme outliers

⚙️ Feature Engineering

Non-linear RI features (RI², RI buckets)

Interaction terms (Al × Si)

Chemical ratios (Na/Ca, Mg/Ca)

Standard Scaling

Manual feature importance analysis

📂 Preprocessing Pipeline

All transformations are reproducible through:

src/data_prep.py
src/features.py
src/infer.py


A consistent set of features is stored in:

models/feature_columns.joblib
models/scaler.joblib

🤖 4. Model Development
Models Trained

Random Forest (baseline)

Tuned Random Forest (GridSearch)

Bagging Classifier

AdaBoost

Gradient Boosting

Stacking Ensemble → Final model

🏆 Best Performing Model

The Stacking Ensemble achieved:

Metric	Score
Accuracy	0.90+
Macro F1 Score	High across all classes
Robustness	Best generalization
Why Stacking?

Combines strengths of multiple algorithms → reduced variance, reduced bias, better handling of mixed feature interactions.

🧪 5. FastAPI Backend – Real-Time Inference

The backend exposes:

GET /health – quick uptime check

POST /predict – returns predicted glass type

⚙️ Key Features

Loads models at startup

Standardizes input

Applies feature engineering automatically

Returns prediction in milliseconds

Light, fast, production-ready API

Example Request
{
  "RI": 1.52,
  "Na": 13.4,
  "Mg": 3.6,
  "Al": 1.2,
  "Si": 72.0,
  "K": 0.4,
  "Ca": 8.8,
  "Ba": 0.0,
  "Fe": 0.1
}

🖥 6. Streamlit Frontend – Interactive Dashboard

A polished and intuitive dashboard with:

Tab 1 — Interactive Data Exploration

Adjustable sliders for all 9 input features

Predict button

Reset button

Latest prediction summary

Probability chart

Last 5 predictions (with timestamps)

Tab 2 — System Architecture

High-level architecture diagram

Explanation of pipelines

Tech stack breakdown

Tab 3 — Model Insights

Accuracy comparison among models

Confusion matrix

F1-score per class

Final conclusions

This UI allows non-technical stakeholders to interact with the ML model easily.

🐳 7. Docker Deployment Architecture

The system uses two lightweight containers:

1️⃣ glass-api

FastAPI backend for inference
Runs on: http://localhost:8000

2️⃣ glass-app

Streamlit frontend
Automatically communicates with backend via internal Docker network
Runs on: http://localhost:8501

docker-compose.yml
version: "3.9"

services:
  glass-api:
    image: ghcr.io/<owner>/glass-identification-api:latest
    ports:
      - "8000:8000"

  glass-app:
    image: ghcr.io/<owner>/glass-identification-app:latest
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://glass-api:8000
    depends_on:
      - glass-api

🤖 8. CI/CD Pipeline (GitHub Actions)
Automated Steps:

✔ Python syntax checks
✔ Build API image → push to GHCR
✔ Build APP image → push to GHCR
✔ Tag images with latest and commit SHA

This ensures:

Reproducibility

Versioned deployments

No dependency on local environment

Production-level workflow

📈 9. Final Results & Insights
Key Findings

RI, Sodium, and Calcium are the strongest predictors.

Types 1, 2, and 7 have high separability.

Types 5 and 6 show minor overlap, but ensemble model handles them well.

Business Value

Enables rapid forensic material classification

Standardizes prediction pipeline

Easy-to-deploy Dockerized ML system

Human-friendly UI for domain experts

🧭 10. Lessons Learned
Technical

Building repeatable ML preprocessing workflows

Designing a clean inference pipeline

Model serving patterns with FastAPI

Packaging multi-service apps with Docker

Implementing CI/CD for machine learning projects

Non-Technical

Importance of explainability for stakeholders

Balancing accuracy and interpretability

Creating UI that helps non-technical users trust ML results

🚀 11. Future Enhancements

Add Prometheus/Grafana monitoring

Introduce model versioning (MLflow)

Auto retraining pipeline

Deploy on cloud (AWS / GCP / DigitalOcean / Fly.io)

Add validation dataset drift detection

🏁 12. Conclusion

This project demonstrates real-world ML system engineering, not just model training.
It includes:

✔ Complete data pipeline
✔ Strong ML model (Stacking Ensemble)
✔ Backend + frontend integration
✔ Docker deployment
✔ CI/CD automation
✔ Production-ready architecture

It is a strong showcase of end-to-end machine learning & MLOps skills.