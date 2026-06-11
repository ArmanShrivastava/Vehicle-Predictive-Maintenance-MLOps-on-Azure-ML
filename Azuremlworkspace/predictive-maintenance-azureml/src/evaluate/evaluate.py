"""
Model evaluation script for predictive maintenance
"""
import argparse
import numpy as np
import joblib
import json
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_auc_score)
from pathlib import Path

def evaluate_model(model_path, test_data_path, output_path):
    """
    Evaluate the trained model on test data
    
    Args:
        model_path: Path to trained model
        test_data_path: Path to test data
        output_path: Path to save evaluation metrics
    """
    # Load model and data
    model = joblib.load(model_path)
    X_test = np.load(f"{test_data_path}/X_test.npy")
    y_test = np.load(f"{test_data_path}/y_test.npy")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    
    # Save metrics
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Model evaluation complete. Metrics: {metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predictive maintenance model")
    parser.add_argument("--model", type=str, default="models/model.pkl", help="Trained model path")
    parser.add_argument("--data", type=str, default="data/processed", help="Test data directory")
    parser.add_argument("--output", type=str, default="metrics/evaluation_metrics.json", help="Output metrics path")
    
    args = parser.parse_args()
    evaluate_model(args.model, args.data, args.output)
