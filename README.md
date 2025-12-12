# Glass Identification – End-to-End ML System
🔬 Machine Learning • 🧱 FastAPI • 🎨 Streamlit • 📦 Docker • 🤖 GitHub Actions CI/CD

A complete end-to-end machine learning system for predicting **glass type** from chemical composition and refractive index using the UCI Glass Dataset.  
The project demonstrates practical ML engineering, including preprocessing, ensemble modeling, reproducible inference, API serving, UI integration, containerization, and CI/CD automation.

---

## Overview

This system is designed as a **production-style ML application**, not a notebook-only experiment.  
It covers the full ML lifecycle:
- data preprocessing and feature engineering
- model training and evaluation
- ensemble-based model selection
- real-time inference via REST API
- interactive user interface
- containerized local deployment
- automated CI/CD using GitHub Actions

---

## Features

### Machine Learning Pipeline
- Data cleaning with zero-value handling, duplicate removal, and constant column removal
- Outlier handling using IQR-based winsorization
- Feature engineering capturing chemical ratios, interactions, and non-linear effects
- Stratified train/test split with feature scaling
- Class imbalance handled using **SMOTE**
- Multiple models trained and evaluated:
  - Random Forest (baseline and tuned)
  - Bagging
  - AdaBoost
  - Gradient Boosting
- **Final model:** Stacking Ensemble persisted to disk
- Evaluation metrics saved including accuracy, classification report, and confusion matrix

### FastAPI Backend
- Stateless inference service
- Loads trained artifacts once at startup
- Performs preprocessing, scaling, and prediction in a single request flow
- Endpoints:
  - `POST /predict`
  - `GET /health`

### Streamlit Frontend
- Interactive dashboard for model exploration
- Adjustable sliders for all chemical features
- Explicit prediction trigger
- Probability distribution visualization
- Displays recent prediction history
- Integrated API health monitoring with local fallback inference

### Dockerized Deployment
- Backend and frontend deployed as independent containers
- Orchestrated locally using Docker Compose
- Ensures consistent execution across environments

### CI/CD with GitHub Actions
- Automated Python syntax validation
- Docker image builds for API and UI services
- Images pushed to GitHub Container Registry
- Tagged with `latest` and commit SHA

---

## System Architecture

- Modular end-to-end ML system with clear separation of concerns

**Streamlit Frontend**
- Collects feature inputs and triggers predictions
- Visualizes predictions, probabilities, and history
- Monitors FastAPI health and falls back to local inference

**FastAPI Backend**
- Serves real-time inference requests
- Validates input and executes preprocessing, scaling, and prediction
- Designed to be stateless and container-ready

**Inference Pipeline**
- Reuses the exact preprocessing and feature engineering logic from training
- Aligns inference features with the training schema
- Applies the persisted scaler and stacking ensemble
- Ensures deterministic and reproducible predictions

**Model Artifacts**
- Stored under the `models/` directory
- Include trained model, scaler, feature columns, and metrics
- Guarantee training–inference parity

**Orchestration, Containerization, and CI/CD**
- `main.py` trains models only if artifacts are missing and launches the UI
- Docker isolates backend and frontend services
- GitHub Actions validates code and builds container images

**Project -Screenshots**

<img width="200" height="200" alt="Screenshot 2025-12-12 at 19 56 10" src="https://github.com/user-attachments/assets/0f0358be-c2ec-47b7-9e31-07888708d8ae" />
<img width="200" height="200" alt="Screenshot 2025-12-12 at 19 56 52" src="https://github.com/user-attachments/assets/064b10bf-6115-45cc-9493-57fb99439e42" />
<img width="200" height="200" alt="Screenshot 2025-12-12 at 19 57 00" src="https://github.com/user-attachments/assets/32ffbb93-842a-4a98-8f83-f3b4f3088ec6" />
<img width="200" height="200" alt="Screenshot 2025-12-12 at 19 57 21" src="https://github.com/user-attachments/assets/def8162a-3eec-48c3-9b79-7e0227046edc" />
<img width="200" height="200" alt="Screenshot 2025-12-12 at 19 57 31" src="https://github.com/user-attachments/assets/ef213adc-a276-452a-9d4d-05f3a1501a4e" />
<img width="200" height="200" alt="Screenshot 2025-12-12 at 19 57 44" src="https://github.com/user-attachments/assets/b29d4272-5097-4713-9ebe-fce9cddbc13b" />
<img width="200" height="200" alt="Screenshot 2025-12-12 at 19 58 33" src="https://github.com/user-attachments/assets/87877977-4aa1-4e4a-97f8-113ea6b7913e" />
<img width="200" height="200" alt="Screenshot 2025-12-12 at 19 58 39" src="https://github.com/user-attachments/assets/77e01c6d-642a-44f9-9f94-f9921d50409c" />
<img width="200" height="200" alt="Screenshot 2025-12-12 at 19 58 49" src="https://github.com/user-attachments/assets/a150dd3a-2c5d-42b1-a764-c70369059985" />
<img width="200" height="200" alt="Screenshot 2025-12-12 at 19 58 57" src="https://github.com/user-attachments/assets/7735af14-99fb-4589-9ded-dbd89337ded6" />

**Docker - Screenshots**

<img width="200" height="200" alt="Screenshot 2025-12-12 at 20 10 59" src="https://github.com/user-attachments/assets/8319b1d4-6ab4-41a5-93a6-70c724018aa6" />
<img width="200" height="200" alt="Screenshot 2025-12-12 at 20 11 14" src="https://github.com/user-attachments/assets/303f4aaf-1013-4ee0-8596-b6a391218a8a" />

**FastAPI - Screenshots**

<img width="200" height="200" alt="Screenshot 2025-12-12 at 20 13 15" src="https://github.com/user-attachments/assets/bd935d41-b27e-43a2-97ed-0f6fee6b08bc" />


## Project Structure

```text
Glass_Identification/
├── src/
│   ├── data_prep.py      # Preprocessing pipeline
│   ├── features.py       # Feature engineering
│   ├── infer.py          # Inference utilities
│   ├── train.py          # Model training
├── app.py                # Streamlit frontend
├── api.py                # FastAPI backend
├── main.py               # Orchestrator
├── models/               # Trained artifacts and metrics
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.app
├── requirements.txt
├── README.md
└── .github/workflows/ci.yml
Installation (Local)
bash
Copy code
git clone https://github.com/YOUR-USERNAME/Glass_Identification.git
cd Glass_Identification
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Running the System
Recommended
Train the model if needed and launch the Streamlit UI:

bash
Copy code
python main.py
Run services manually
bash
Copy code
uvicorn api:app --reload --port 8000
streamlit run app.py
Docker (Local)
bash
Copy code
docker compose up --build
Access:

API → http://localhost:8000

UI → http://localhost:8501

FastAPI Example
http
Copy code
POST /predict
json
Copy code
{
  "RI": 1.52,
  "Na": 13.2,
  "Mg": 3.6,
  "Al": 1.2,
  "Si": 72.3,
  "K": 0.4,
  "Ca": 8.5,
  "Ba": 0.0,
  "Fe": 0.1
}
json
Copy code
{ "predicted_type": 3 }
Tech Stack
ML & Data

pandas, numpy, scikit-learn, imbalanced-learn, joblib

Backend

FastAPI, Uvicorn

Frontend

Streamlit, Matplotlib

DevOps

Docker, Docker Compose

GitHub Actions

GitHub Container Registry

Summary
This project demonstrates applied machine learning engineering beyond notebooks, including ensemble modeling, reproducible inference, API-driven architecture, containerized deployment, and CI/CD automation.

