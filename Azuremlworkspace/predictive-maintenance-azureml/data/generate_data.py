import numpy as np
import pandas as pd

np.random.seed(42)
N = 50_000

df = pd.DataFrame({
    "vehicle_id": np.random.randint(1000, 2000, N),
    "engine_temp_c": np.random.normal(90, 8, N),
    "oil_pressure_psi": np.random.normal(40, 6, N),
    "battery_voltage": np.random.normal(12.6, 0.5, N),
    "brake_pad_mm": np.clip(np.random.normal(8, 3, N), 0.5, 15),
    "vibration_rms": np.abs(np.random.normal(0.5, 0.3, N)),
    "mileage_km": np.random.uniform(5_000, 250_000, N),
    "days_since_service": np.random.randint(0, 400, N),
    "ambient_temp_c": np.random.normal(28, 10, N),
    "avg_speed_kmh": np.clip(np.random.normal(55, 20, N), 5, 140),
})

logit = (
    0.06 * (df["engine_temp_c"] - 90)
    + 0.12 * (35 - df["oil_pressure_psi"]).clip(lower=0)
    + 1.5 * (12.0 - df["battery_voltage"]).clip(lower=0)
    + 0.25 * (3 - df["brake_pad_mm"]).clip(lower=0)
    + 2.0 * (df["vibration_rms"] - 0.8).clip(lower=0)
    + 0.004 * df["days_since_service"]
    + 0.000004 * df["mileage_km"]
    - 3.5
)
prob = 1 / (1 + np.exp(-logit))
df["failure_30d"] = (np.random.rand(N) < prob).astype(int)

df.to_csv("data/vehicle_telemetry.csv", index=False)
print(df["failure_30d"].value_counts(normalize=True))