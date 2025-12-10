Glass Identification – End-to-End ML System
🔬 Machine Learning • 🧱 FastAPI Backend • 🎨 Streamlit Frontend • 📦 Docker • 🤖 CI/CD with GitHub Actions

This project is a complete end-to-end ML system built using the UCI Glass Dataset.
It includes:

Data preprocessing + feature engineering

Model training pipeline

Stacking ensemble classifier

Interactive Streamlit UI

FastAPI REST inference service

Docker-based deployment (API + APP containers)

Automated CI/CD pipeline via GitHub Actions

🚀 Features
🔧 End-to-End ML Pipeline

Data cleaning, winsorization, feature engineering

Multiple model training: RF, Gradient Boosting, Bagging, AdaBoost

Final Stacking Ensemble model saved to artifacts (models/)

Metrics logged: accuracy, F1-score, confusion matrix

🌐 FastAPI Backend

/predict endpoint for model inference

/health endpoint for uptime monitoring

Serves pre-processing, scaling, and ensemble model predictions

🖥 Streamlit Frontend

Beautiful dashboard with tabs:

Interactive Data Exploration

System Architecture Overview

Model Performance & Final Insights

Adjustable sliders for all chemical components

Displays prediction, probability distribution, and previous inputs history

🐳 Dockerized Deployment

Two independent containers:

glass-api → FastAPI backend

glass-app → Streamlit interface

Orchestrated using docker-compose.

🤖 CI/CD with GitHub Actions

Builds both Docker images (API + UI)

Pushes them to GitHub Container Registry (GHCR)

Syntax checks, linting, and validation

📁 Project Structure
Glass_Identification/
│
├── src/
│   ├── data_prep.py          # Preprocessing pipeline
│   ├── features.py           # Feature engineering helpers
│   ├── infer.py              # Inference utilities
│   ├── train.py              # Model training script
│
├── app.py                    # Streamlit frontend
├── api.py                    # FastAPI backend
├── main.py                   # Orchestrator (train / predict / UI / API)
│
├── models/
│   ├── stacking_model.joblib
│   ├── scaler.joblib
│   ├── feature_columns.joblib
│   ├── metrics.json
│
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.app
│
├── requirements.txt
├── README.md
└── .github/workflows/ci.yml  # CI/CD pipeline

🛠 Installation (Local Development)
1️⃣ Clone the repository
git clone https://github.com/YOUR-USERNAME/Glass_Identification.git
cd Glass_Identification

2️⃣ Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

▶️ Running the System Locally
🌐 Start Streamlit App
streamlit run app.py


This launches the UI at:

http://localhost:8501

🚀 Start FastAPI Backend
uvicorn api:app --reload --port 8000


API docs:

http://localhost:8000/docs

🐳 Running with Docker (Recommended)
Build and start both containers
docker-compose up --build


This runs:

API → http://localhost:8000

Streamlit UI → http://localhost:8501

Stop
docker-compose down

🤖 CI/CD Pipeline (GitHub Actions)

Your workflow:

Builds Docker images using Dockerfile.api and Dockerfile.app

Tags images as:

ghcr.io/<owner>/glass-identification-api:latest

ghcr.io/<owner>/glass-identification-app:latest

Pushes them to GitHub Container Registry

Runs Python syntax checks

File: .github/workflows/ci.yml

This means every push to main automatically rebuilds + publishes your containers.

🧬 FastAPI Endpoints
Health Check
GET /health
→ { "status": "ok" }

Predict
POST /predict
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

Response:
{
  "predicted_type": 3
}

📊 Streamlit Dashboard
Tab 1 — Interactive Feature Exploration

Adjust RI, Na, Mg, Al, Si, K, Ca, Ba, Fe via sliders

Predict glass type

View probability breakdown

See last 5 predictions with timestamps

Tab 2 — System Architecture

Explains:

Data pipeline

Modeling pipeline

Serving pipeline

CI/CD pipeline

Tech stack

Tab 3 — Final Results

Best model summary

Confusion matrix

F1-score per class

Key conclusions & recommendations

🧱 Tech Stack
Backend

FastAPI

scikit-learn

Pandas / NumPy

Frontend

Streamlit

Matplotlib / Seaborn

Deployment

Docker

Docker Compose

GitHub Actions (CI/CD)

GitHub Container Registry (GHCR)

📌 Future Improvements

Add monitoring with Prometheus

Add authentication for API

Deploy to DigitalOcean / AWS / Fly.io

Add model retraining pipeline

Add batch inference jobs

🤝 Contributing

Contributions are welcome!
Please open an issue or submit a pull request.

📄 License

MIT License.
Feel free to use and modify this project.

🎉 Final Note

You now have a production-ready professional ML system with:

✔ Streamlit UI
✔ FastAPI backend
✔ Stacking model
✔ Docker deployment
✔ Full CI/CD pipeline