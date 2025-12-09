import sys
import time
from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd
import requests
import seaborn as sns

import streamlit as st
import matplotlib.pyplot as plt
import subprocess

from src.data_prep import load_data, full_preprocess, split_scale_smote
from src.infer import preprocess_single_row

MODELS_DIR = Path("models")
import os
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="Glass Identification – End-to-End ML System Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧪 Glass Identification – End-to-End ML System Dashboard")

# ---------------------------------------------------
# Helpers
# ---------------------------------------------------
@st.cache_data
def get_processed_data():
    df = load_data()
    df = full_preprocess(df)
    return df


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / "stacking_model.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    feature_columns = joblib.load(MODELS_DIR / "feature_columns.joblib")
    with open(MODELS_DIR / "metrics.json") as f:
        metrics = json.load(f)
    return model, scaler, feature_columns, metrics


def start_fastapi():
    subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api:app", "--reload"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1)


def api_alive():
    try:
        r = requests.get(API_URL + "/health", timeout=1)
        return r.status_code == 200
    except Exception:
        return False


def explain_feature(name: str) -> str:
    mapping = {
        "RI": "Refractive index – higher values usually indicate denser glass.",
        "Na": "Sodium content – affects melting point and strength.",
        "Mg": "Magnesium – influences durability and thermal resistance.",
        "Al": "Aluminium oxide – improves hardness and structural stability.",
        "Si": "Silicon oxide (SiO₂) – main glass former.",
        "K": "Potassium – acts as a flux, reducing melting temperature.",
        "Ca": "Calcium – increases chemical durability.",
        "Ba": "Barium – improves clarity and adds weight.",
        "Fe": "Iron – affects coloration (green/brown tint).",
    }
    return mapping.get(name, "")



# ---------------------------------------------------
# Load data & model
# ---------------------------------------------------
df = get_processed_data()
model, scaler, feature_columns, metrics = load_artifacts()

# ---------------------------------------------------
# Sidebar controls
# ---------------------------------------------------
st.sidebar.header("⚙ Backend Controls")

if st.sidebar.button("Start FastAPI server"):
    start_fastapi()
    st.sidebar.success("FastAPI started at http://127.0.0.1:8000/docs")

st.sidebar.markdown(
    "**FastAPI status:** " + ("🟢 Running" if api_alive() else "🔴 Stopped")
)

# ---------------------------------------------------
# Tabs
# ---------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["🔎 Interactive Data Exploration", "🏗 System Architecture", "📊 Final Model & Project Conclusion"]
)

# ===================================================
# TAB 1 – INTERACTIVE FEATURE EXPLORATION
# ===================================================
# TAB 1 – INTERACTIVE FEATURE EXPLORATION
# ===================================================
with tab1:
    st.subheader("🔎 Explore Feature Effects on Prediction")

    st.write(
        "Use the sliders to adjust each chemical component. "
        "Click **Predict** to see how the model classifies the glass type. "
        "Changing sliders alone will **not** change the result until you click Predict again."
    )

    base_cols = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]

    # ------- Session state: prediction + history + reset counter -------
    if "last_output" not in st.session_state:
        st.session_state["last_output"] = None  # dict with pred, proba, payload
    if "history" not in st.session_state:
        st.session_state["history"] = []       # list of dicts (last 5)
    if "reset_counter" not in st.session_state:
        st.session_state["reset_counter"] = 0  # increments on each reset

    reset_counter = st.session_state["reset_counter"]

    # Defaults from data (medians) – for initial slider values
    defaults = {col: float(df[col].median()) for col in base_cols}

    # ---------------- SLIDERS ----------------
    col1, col2, col3 = st.columns(3)
    inputs = {}

    for i, col_name in enumerate(base_cols):
        col = col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3

        if col_name in df.columns:
            col_min = float(df[col_name].min())
            col_max = float(df[col_name].max())
        else:
            col_min, col_max = 0.0, 10.0

        default_val = defaults[col_name]

        # Use a key that depends on reset_counter so widgets are recreated
        slider_key = f"slider_{col_name}_{reset_counter}"

        # ===== RI SPECIAL HANDLING (scale ×100 for UX) =====
        if col_name == "RI":
            scaled_min = round(col_min * 100, 2)
            scaled_max = round(col_max * 100, 2)
            scaled_default = round(default_val * 100, 2)

            scaled_value = col.slider(
                f"{col_name} – {explain_feature(col_name)} (scaled ×100)",
                min_value=scaled_min,
                max_value=scaled_max,
                value=scaled_default,
                step=0.1,
                key=slider_key,
            )

            inputs["RI"] = scaled_value / 100.0
            continue
        # ==================================================

        inputs[col_name] = col.slider(
            f"{col_name} – {explain_feature(col_name)}",
            min_value=col_min,
            max_value=col_max,
            value=default_val,
            step=0.01,
            key=slider_key,
        )

    # ------------- BUTTON ROW: PREDICT + RESET -------------
    btn1, btn2 = st.columns(2)
    with btn1:
        predict_button = st.button("Predict Glass Type")
    with btn2:
        reset_button = st.button("Reset Inputs")

    # ---------------- RESET LOGIC ----------------
    if reset_button:
        # Increment counter so all slider keys change on next run
        st.session_state["reset_counter"] += 1
        # Clear only the current prediction; keep history
        st.session_state["last_output"] = None
        st.rerun()

    # ---------------- PREDICTION LOGIC ----------------
    if predict_button:
        payload = {k: float(v) for k, v in inputs.items()}
        pred_type = None
        X_scaled = None

        # Try API first
        if api_alive():
            try:
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
                if resp.status_code == 200:
                    pred_type = int(resp.json().get("predicted_type"))
                    st.success(f"🎯 **Predicted Glass Type (via API): {pred_type}**")
                else:
                    st.error(f"API error: {resp.status_code} – {resp.text}")
            except Exception as e:
                st.error(f"API call failed: {e}")

        # Local fallback or for probability plot
        if pred_type is None:
            st.info("Using local model pipeline (API not available).")
            df_proc = preprocess_single_row(payload)
            X_scaled = scaler.transform(df_proc)
            pred_type = int(model.predict(X_scaled)[0])
        else:
            # If we predicted via API, recompute pipeline locally for proba
            df_proc = preprocess_single_row(payload)
            X_scaled = scaler.transform(df_proc)

        proba = model.predict_proba(X_scaled)[0]
        classes = model.classes_
        # Add timestamp
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save full output
        st.session_state["last_output"] = {
            "predicted_type": pred_type,
            "payload": payload,
            "proba": proba.tolist(),
            "classes": classes.tolist(),
            "timestamp": timestamp,
        }

        # Save into history
        record = {**payload, "predicted_type": pred_type, "timestamp": timestamp}

        history = st.session_state["history"]
        history.append(record)
        st.session_state["history"] = history[-5:]

    # =============== DISPLAY LAST RESULT (PERSISTS ON SLIDER MOVE) ===============
    last = st.session_state["last_output"]

    if last is not None:
        st.markdown("---")
        st.markdown("## 🧾 Latest Prediction Summary")

        st.success(f"🎯 **Predicted Glass Type: {last['predicted_type']}**")

        # Probability bar chart
        prob_df = pd.DataFrame(
            {"Glass Type": last["classes"], "Probability": last["proba"]}
        )
        st.markdown("### 🔍 Prediction Confidence")
        st.bar_chart(prob_df.set_index("Glass Type"))

        # Current input + prediction
        st.markdown("### 📄 Input Values & Prediction")
        result_row = {**last["payload"], "Predicted Type": last["predicted_type"]}
        st.dataframe(pd.DataFrame([result_row]))

    # =============== DISPLAY LAST 5 RECORDS ===============
    if st.session_state["history"]:
        st.markdown("---")
        st.markdown("## 🕐 Last 5 Predictions")

        hist_df = pd.DataFrame(st.session_state["history"])

        # Column ordering
        cols = ["timestamp"] + base_cols + ["predicted_type"]
        cols = [c for c in cols if c in hist_df.columns]
        hist_df = hist_df[cols]

        # Chronological index (0 = oldest)
        hist_df["Record #"] = list(range(len(hist_df)))

        # Reverse for display (latest on top)
        hist_df = hist_df.iloc[::-1]

        # Move index column to the left
        hist_df = hist_df[["Record #"] + [c for c in hist_df.columns if c != "Record #"]]

        st.dataframe(hist_df)

# ===================================================
# TAB 2 – SYSTEM ARCHITECTURE (Reordered Professionally)
# ===================================================
with tab2:
    st.subheader("🏗 System Architecture Overview")

    # ========================
    # 1️⃣ HIGH-LEVEL ARCHITECTURE
    # ========================
    st.markdown("## 1️⃣ High-Level System Architecture")

    st.code(
        """
+------------------------+            +------------------------+
|    User / Analyst      |            |   CI/CD Automation     |
|  - Browser UI          |            | - Training Pipeline    |
+-----------+------------+            | - Metrics Logging      |
            |                         +-----------+------------+
            |                                     |
            v                                     v
+------------------------+            +------------------------+
|   Streamlit Frontend   | <------->  |      FastAPI Backend   |
|  - Input Sliders       |   HTTP     |  - /predict, /health   |
|  - Results + History   |            |  - Realtime inference  |
+-----------+------------+            +-----------+------------+
            |                                     |
            v                                     v
    +-------------------+               +----------------------+
    | Preprocessing &   |               |  Stacking Model      |
    | Feature Engineering               |  + Scaler + Columns  |
    +-------------------+               +----------------------+
            |
            v
    +-------------------+
    |   Raw Glass Data  |
    +-------------------+
"""
    )

    st.markdown("---")

    # ================================
    # 2️⃣ SERVING LAYER (BACKEND + FRONTEND)
    # ================================
    st.markdown("## 2️⃣ Serving Layer (Backend + Frontend)")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### 🔌 Backend — FastAPI Service")
        st.markdown(
            """
- Hosts the ML model and exposes REST API endpoints:
  - **`/predict`** → Returns predicted glass type.
  - **`/health`** → Lightweight check for liveness.
- Entire preprocessing pipeline reused via `preprocess_single_row()`.
- Ensures **consistent transformations** between training and real-time inference.
- Uses saved:
  - `stacking_model.joblib`
  - `scaler.joblib`
  - `feature_columns.joblib`
"""
        )

    with col_r:
        st.markdown("### 🖥 Frontend — Streamlit Application")
        st.markdown(
            """
- Interactive UI for exploring model predictions.
- Features:
  - Input sliders for each chemical attribute.
  - Predict button → explicitly triggers inference.
  - Persistent result + confidence chart.
  - Full history (last 5 predictions with timestamp).
- Auto-fallback:
  - If FastAPI is offline, app uses local inference path.
- Clean separation → UI + server independent.
"""
        )

    st.markdown("---")

    # ================================
    # 3️⃣ END-TO-END ML PIPELINE
    # ================================
    st.markdown("## 3️⃣ End-to-End Machine Learning Pipeline")

    st.markdown(
        """
### 🔹 **Data Layer**
- Source: UCI Glass Identification Dataset.
- Loaded via `load_data()` from `/data/glass.xlsx`.

### 🔹 **Preprocessing Steps**
1. Replace missing/zero values using **median imputation**.  
2. Remove duplicates + constant features.  
3. Apply **winsorisation** for robust outlier handling.  
4. Execute all transformations inside `full_preprocess()` (training & inference unified).

### 🔹 **Feature Engineering**
- Chemical ratios: **Na/Ca**, **Mg/Ca**  
- Oxide interaction: **Al × Si**  
- Non-linear RI features: `RI²` and quantile bins (one-hot encoded)  
- Ensures the model captures domain-specific chemistry patterns.

### 🔹 **Train/Test Split + Scaling + Balancing**
- Stratified train/test split.  
- `StandardScaler` for feature normalisation.  
- **SMOTE** to oversample minority glass classes.

### 🔹 **Model Training**
Trained and evaluated:
- Random Forest (baseline)
- Tuned Random Forest
- Bagging
- AdaBoost
- Gradient Boosting
- **Final Model: Stacking Ensemble**  
  ➝ (RF Tuned + GB + AdaBoost → Logistic Regression meta-learner)

### 🔹 **Model Saving**
Artifacts stored in `/models/`:
- `stacking_model.joblib`
- `scaler.joblib`
- `feature_columns.joblib`
- `metrics.json` (all evaluation metrics)
"""
    )

    st.markdown("---")

    # ================================
    # 4️⃣ TECH STACK TABLE
    # ================================
    st.markdown("## 4️⃣ Tech Stack")

    st.markdown(
        """
| Layer                | Tools / Technologies Used                                   |
|----------------------|--------------------------------------------------------------|
| Data Processing      | pandas, numpy                                                |
| ML Modelling         | scikit-learn, imbalanced-learn (SMOTE)                       |
| Model Persistence    | joblib, structured `/models` directory                       |
| API Backend          | FastAPI, uvicorn                                             |
| Frontend             | Streamlit                                                    |
| Evaluation           | Classification report, confusion matrix, F1, accuracy        |
| Engineering Practices| Modular code, reusable preprocessing, version-controlled data |
"""
    )
    st.markdown("---")
    st.markdown("## 5️⃣ Containerisation & CI/CD")

    st.markdown(
        """
**Docker**

- The project can be containerised using Docker:
  - One image for the **FastAPI backend** (`/predict`, `/health`).
  - One image for the **Streamlit frontend** (user-facing dashboard).
- This makes the system portable and easy to deploy on any cloud (AWS / GCP / Azure / Render).

**CI/CD (GitHub Actions)**

- A simple CI workflow can:
  - Install dependencies and run unit tests (`pytest`).
  - Build Docker images for the API and the app.
  - Optionally push images to a container registry (e.g. GitHub Container Registry).
- This shows an end-to-end MLOps mindset: not just training models, but packaging and delivering them reliably.
        """
    )

# ===================================================
# TAB 3 – FINAL MODEL & PROJECT CONCLUSION
# ===================================================
with tab3:
    st.subheader("📌 Final Model & Project Conclusion")

    # ---- 1. Compact metric cards ----
    best_acc = metrics["stacking"]["accuracy"]
    rf_tuned_acc = metrics["rf_tuned"]["accuracy"]
    rf_base_acc = metrics["rf_baseline"]["accuracy"]

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric(
            "✅ Final Selected Model",
            "Stacking Ensemble",
            help="Meta-model combining tuned Random Forest, Gradient Boosting and AdaBoost.",
        )
    with col_b:
        st.metric("Best Test Accuracy", f"{best_acc * 100:.2f} %")
    with col_c:
        st.metric("RF (tuned) Accuracy", f"{rf_tuned_acc * 100:.2f} %")

    st.markdown("---")

    # ---- 2. Accuracy summary table for all models ----
    st.markdown("### 📊 Model Comparison Overview")

    summary_df = pd.DataFrame(
        {
            "Model": [
                "RF (baseline)",
                "RF (tuned)",
                "Bagging",
                "AdaBoost",
                "GradientBoosting",
                "Stacking Ensemble (final)",
            ],
            "Test Accuracy": [
                metrics["rf_baseline"]["accuracy"],
                metrics["rf_tuned"]["accuracy"],
                metrics["bagging"]["accuracy"],
                metrics["ada"]["accuracy"],
                metrics["gb"]["accuracy"],
                metrics["stacking"]["accuracy"],
            ],
        }
    )

    st.table(summary_df.style.format({"Test Accuracy": "{:.3f}"}))

    st.markdown("---")

    # ---- 3. Short, interview-style conclusion text ----
    st.markdown("### 🧾 Project Conclusion")

    st.markdown(
        """
- The **Stacking Ensemble** was selected as the final model, achieving around  
  **{best:.1f}% test accuracy**, outperforming the tuned Random Forest and other ensembles.
- The pipeline includes:
  - Data cleaning (zero/NA handling, duplicate & constant column removal, winsorised outliers)
  - **Feature engineering** (ratio features, Al×Si interaction, RI non-linear features)
  - **Class imbalance handling** with SMOTE on the training set
  - Evaluation with accuracy and per-class metrics on a held-out test set.
- The final model is **production-ready**:
  - Persisted using `joblib` under the `models/` folder
  - Served via a **FastAPI** endpoint (`/predict`, `/health`)
  - Exposed to users through an **interactive Streamlit dashboard** with sliders and rich feedback.
        """.format(
            best=best_acc * 100
        )
    )

    st.markdown("### 🚀 What This System Can Do")

    st.markdown(
        """
- Accept a new glass sample’s **chemical composition + RI** and predict the **glass type**.
- Give users **confidence scores** (class probabilities) for each prediction.
- Log and display the **last 5 predictions** for quick inspection and demo purposes.
        """
    )


