"""
Data preparation script for predictive maintenance model
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json

def prepare_data(input_path, output_path, test_size=0.2):
    """
    Prepare and preprocess vehicle maintenance data
    
    Args:
        input_path: Path to raw data CSV
        output_path: Path to save processed data
        test_size: Proportion of data for testing
    """
    # Read data
    df = pd.read_csv(input_path)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.fillna(df.mean(numeric_only=True))
    
    # Feature engineering
    df['high_temp'] = (df['engine_temp'] > 100).astype(int)
    df['low_pressure'] = (df['oil_pressure'] < 45).astype(int)
    df['low_battery'] = (df['battery_voltage'] < 11).astype(int)
    
    # Separate features and target
    X = df.drop('maintenance_needed', axis=1)
    y = df['maintenance_needed']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.select_dtypes(include=[np.number]))
    
    # Split data
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Save processed data
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(f"{output_path}/X_train.npy", X_train)
    np.save(f"{output_path}/X_test.npy", X_test)
    np.save(f"{output_path}/y_train.npy", y_train)
    np.save(f"{output_path}/y_test.npy", y_test)
    
    print(f"Data preparation complete. Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for training")
    parser.add_argument("--input", type=str, default="data/vehicle_maintenance.csv", help="Input data path")
    parser.add_argument("--output", type=str, default="data/processed", help="Output directory path")
    
    args = parser.parse_args()
    prepare_data(args.input, args.output)
