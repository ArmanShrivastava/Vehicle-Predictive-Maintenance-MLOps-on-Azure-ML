"""
Model training script for predictive maintenance
"""
import argparse
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
import json

def train_model(train_data_path, model_output_path, metrics_output_path):
    """
    Train a Random Forest classifier for maintenance prediction
    
    Args:
        train_data_path: Path to processed training data
        model_output_path: Path to save the trained model
        metrics_output_path: Path to save training metrics
    """
    # Load training data
    X_train = np.load(f"{train_data_path}/X_train.npy")
    y_train = np.load(f"{train_data_path}/y_train.npy")
    X_test = np.load(f"{train_data_path}/X_test.npy")
    y_test = np.load(f"{train_data_path}/y_test.npy")
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred))
    }
    
    # Save model and metrics
    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_output_path).parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, f"{model_output_path}/model.pkl")
    
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Model training complete. Metrics: {metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train predictive maintenance model")
    parser.add_argument("--data", type=str, default="data/processed", help="Processed data directory")
    parser.add_argument("--model-output", type=str, default="models", help="Model output directory")
    parser.add_argument("--metrics-output", type=str, default="metrics/training_metrics.json", help="Metrics output path")
    
    args = parser.parse_args()
    train_model(args.data, args.model_output, args.metrics_output)
