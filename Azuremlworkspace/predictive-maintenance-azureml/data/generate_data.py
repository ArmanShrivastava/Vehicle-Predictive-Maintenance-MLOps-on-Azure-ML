"""
Data generation script for vehicle predictive maintenance
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def generate_synthetic_data(n_samples=1000, output_path="data/vehicle_maintenance.csv"):
    """
    Generate synthetic vehicle maintenance data
    
    Args:
        n_samples: Number of records to generate
        output_path: Path to save the CSV file
    """
    np.random.seed(42)
    
    data = {
        'vehicle_id': range(1, n_samples + 1),
        'engine_temp': np.random.normal(90, 10, n_samples),
        'oil_pressure': np.random.normal(50, 5, n_samples),
        'battery_voltage': np.random.normal(12, 1, n_samples),
        'mileage': np.random.uniform(0, 200000, n_samples),
        'age_years': np.random.uniform(0, 15, n_samples),
        'maintenance_needed': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    df = pd.DataFrame(data)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {n_samples} records to {output_path}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic vehicle maintenance data")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="data/vehicle_maintenance.csv", help="Output file path")
    
    args = parser.parse_args()
    generate_synthetic_data(args.samples, args.output)
