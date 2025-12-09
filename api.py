from fastapi import FastAPI
from pydantic import BaseModel,Field

from src.infer import load_artifacts, preprocess_single_row

# FastAPI app instance
app = FastAPI(title="Glass Identification API")

# Load model artifacts once at startup
model, scaler, feature_columns = load_artifacts()


class GlassInput(BaseModel):
    RI: float = Field(..., description="Refractive index", example=1.517)
    Na: float = Field(..., description="Sodium (weight %)", example=13.0)
    Mg: float = Field(..., description="Magnesium (weight %)", example=3.0)
    Al: float = Field(..., description="Aluminium (weight %)", example=1.5)
    Si: float = Field(..., description="Silicon (weight %)", example=72.0)
    K: float = Field(..., description="Potassium (weight %)", example=0.5)
    Ca: float = Field(..., description="Calcium (weight %)", example=8.5)
    Ba: float = Field(..., description="Barium (weight %)", example=0.0)
    Fe: float = Field(..., description="Iron (weight %)", example=0.0)

@app.get("/")
def root():
    return {"message": "Glass Identification API is running"}


@app.get("/health")
def health_check():
    """Simple health endpoint used by Streamlit to check if API is up."""
    return {"status": "ok"}


@app.post("/predict")
def predict_glass(data: GlassInput):
    """
    Predict glass type given chemical composition and RI.
    """
    # Preprocess sample using the same pipeline as training
    df_proc = preprocess_single_row(data.dict())

    # Scale and predict
    X_scaled = scaler.transform(df_proc)
    pred = model.predict(X_scaled)[0]

    return {"predicted_type": int(pred)}
