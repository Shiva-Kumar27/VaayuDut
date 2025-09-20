import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np

app = FastAPI(title="Air Quality Prediction API")

# Load your GRU model
try:
    from tensorflow import keras
    model = keras.models.load_model("final_gru_model.keras")
    print("✅ GRU Model loaded successfully!")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

class ForecastRequest(BaseModel):
    city: str
    horizon_hours: int

@app.get("/")
def root():
    return {
        "message": "Air Quality Prediction API is running!", 
        "model_loaded": model is not None,
        "model_info": {
            "input_shape": str(model.input_shape) if model else None,
            "output_shape": str(model.output_shape) if model else None
        }
    }

@app.post("/predict")
def predict(req: ForecastRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Your model expects: (batch_size, 24, 18)
        timesteps = 24
        n_features = 18
        
        # Create input sequences
        X = []
        
        for pred_hour in range(req.horizon_hours):
            # Create a sequence of 24 timesteps with 18 features each
            sequence = []
            for t in range(timesteps):
                # Example features - replace with your actual features from training
                features = [
                    pred_hour + t,                        # hour feature
                    25.0 + np.random.normal(0, 2),       # temperature
                    60.0 + np.random.normal(0, 5),       # humidity
                    5.0 + np.random.normal(0, 1),        # wind_speed
                    1013.0 + np.random.normal(0, 10),    # pressure
                    *np.random.normal(0, 1, 13)          # 13 more placeholder features
                ]
                sequence.append(features)
            
            X.append(sequence)
        
        # Convert to numpy array
        X_array = np.array(X).astype(np.float32)
        print(f"Input shape: {X_array.shape}")
        
        # Make prediction
        preds = model.predict(X_array)
        print(f"Prediction shape: {preds.shape}")
        print(f"Raw predictions: {preds[:3] if len(preds) > 0 else 'No predictions'}")
        
        # Scale predictions from normalized values to realistic pollution concentrations
        # NO2: typically 20-150 μg/m³, O3: typically 50-250 μg/m³
        no2_scaled = ((preds[:, 0] * 40) + 80).clip(min=10).tolist()
        o3_scaled = ((preds[:, 1] * 60) + 120).clip(min=20).tolist()
        
        print(f"Scaled NO2: {no2_scaled[:3] if len(no2_scaled) > 0 else 'No NO2 data'}")
        print(f"Scaled O3: {o3_scaled[:3] if len(o3_scaled) > 0 else 'No O3 data'}")
        
        return {
            "city": req.city,
            "hour": list(range(req.horizon_hours)),
            "NO2": no2_scaled,
            "O3": o3_scaled,
            "raw_predictions": {
                "NO2_raw": preds[:, 0].tolist(),
                "O3_raw": preds[:, 1].tolist()
            }
        }
        
    except Exception as e:
        print(f"Detailed error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "api_version": "1.0.0"
    }