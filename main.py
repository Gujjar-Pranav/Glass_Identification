import sys
import subprocess
from pathlib import Path

from src.train import train_all_models

MODELS_DIR = Path("models")


def ensure_trained():
    """
    Train models once if no saved model exists.
    """
    model_path = MODELS_DIR / "stacking_model.joblib"
    if not model_path.exists():
        print("🔁 No trained model found. Training now...")
        train_all_models()
        print("✅ Training finished and model saved.")
    else:
        print(f"✅ Found existing model at {model_path}. Skipping training.")


def launch_streamlit():
    """
    Launch the Streamlit app.
    """
    print("🌐 Launching Streamlit app...")
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    ensure_trained()
    launch_streamlit()
