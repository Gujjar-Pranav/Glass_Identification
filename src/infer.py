import pandas as pd
from pathlib import Path
import joblib

from src.data_prep import load_data, full_preprocess

MODELS_DIR = Path("models")


def load_artifacts():
    """Load model, scaler and feature list from disk."""
    model = joblib.load(MODELS_DIR / "stacking_model.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    feature_columns = joblib.load(MODELS_DIR / "feature_columns.joblib")
    return model, scaler, feature_columns


def preprocess_single_row(sample: dict) -> pd.DataFrame:
    """
    Apply the **full training preprocessing** to a single input sample
    by appending it to the original dataset, then align to feature_columns.
    """

    # 1. Load original training data
    df_train_raw = load_data()

    # 2. Create a one-row DataFrame from the new sample
    df_new = pd.DataFrame([sample])
    df_new["Type"] = 1  # dummy label

    # 3. Concatenate so that preprocessing (IQR, RI bins, etc.) uses train stats
    df_all_raw = pd.concat([df_train_raw, df_new], ignore_index=True)

    # 4. Run full preprocessing on combined data
    df_all = full_preprocess(df_all_raw)

    # 5. Take only the last row (our processed sample)
    df_out = df_all.tail(1)

    # 6. Align with training feature columns (same names + order)
    _, _, feature_columns = load_artifacts()
    df_out = df_out.reindex(columns=feature_columns, fill_value=0)

    return df_out
