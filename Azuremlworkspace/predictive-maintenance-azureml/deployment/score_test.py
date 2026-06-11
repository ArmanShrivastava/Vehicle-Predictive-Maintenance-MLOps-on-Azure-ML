"""
Scoring script for vehicle maintenance prediction model
"""
import json
import joblib
import numpy as np
from pathlib import Path

def init():
    """Initialize model on container startup"""
    global model
    model_path = Path(os.getenv("AZUREML_MODEL_DIR")) / "model.pkl"
    model = joblib.load(model_path)

def run(raw_data):
    """
    Make predictions on input data
    
    Args:
        raw_data: Input data as JSON
        
    Returns:
        Predictions as JSON
    """
    try:
        # Parse input
        data = json.loads(raw_data)
        
        # Convert to numpy array
        input_features = np.array(data).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_features)
        prediction_proba = model.predict_proba(input_features)
        
        # Format output
        result = {
            "prediction": int(prediction[0]),
            "probability": {
                "no_maintenance_needed": float(prediction_proba[0][0]),
                "maintenance_needed": float(prediction_proba[0][1])
            }
        }
        
        return json.dumps(result)
        
    except Exception as e:
        return json.dumps({"error": str(e)})
