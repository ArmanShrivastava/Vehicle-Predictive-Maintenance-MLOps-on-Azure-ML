# CAR-PREDICTIVE-MAINTENANCE
Predictive Maintenance MLOps Platform that can ingest telemetry from millions of vehicles, train AI models on NVIDIA GPUs, serve predictions in real-time, monitor drift, and automatically retrain itself.
What Business Problem Does It Solve?

 We wants to predict:

Engine failures
Battery failures
Brake wear
Transmission issues
Sensor malfunctions

before they happen.

Instead of:

Vehicle fails
↓
Customer stranded
↓
Service center repair
↓
High cost

The platform does:

Vehicle sends telemetry
↓
AI predicts failure in next 7 days
↓
Customer receives alert
↓
Preventive maintenance scheduled
↓
Failure avoided

The flow 

Vehicles
        │
        ▼
Azure IoT Hub
        │
        ▼
Kafka
        │
        ▼
Spark Streaming
        │
        ▼
Feature Engineering
        │
        ▼
Delta Lake
        │
        ▼
DVC
        │
        ▼
Airflow
        │
        ▼
GPU Training (PyTorch)
        │
        ▼
MLflow
        │
        ▼
TensorRT
        │
        ▼
Triton Server
        │
        ▼
KServe
        │
        ▼
Prediction API
        │
        ▼
Drift Detection
        │
        ▼
Auto Retraining


This project is an end-to-end enterprise MLOps platform for Volkswagen predictive maintenance that ingests vehicle telemetry through Kafka and Spark, trains distributed PyTorch models on NVIDIA A100 GPUs, manages experiments with MLflow, optimizes inference using TensorRT and Triton, serves models via KServe on Kubernetes, and continuously monitors model health and drift using Evidently, Prometheus, Grafana, Loki, Jaeger, and OpenTelemetry, enabling a fully automated retraining lifecycle.




| Component | Purpose |
|-----------|---------|
| Event Hubs (Kafka) | Receives the firehose of vehicle sensor readings. |
| Delta Lake + DVC | Stores telemetry with ACID guarantees; DVC version-stamps every training dataset so any experiment is reproducible. |
| Feast | Serves the exact same features at training and inference, ensuring no training-serving skew. |
| Airflow | Orchestrates the platform by running training pipelines, drift checks, and retraining triggers on schedule or on demand. |
| GPU Pool + PyTorch | Trains machine learning models and scales from zero, so GPU costs are incurred only when training jobs run. |
| MLflow | Tracks experiments and maintains the model registry with quality gates before production deployment. |
| Triton + TensorRT | Provides high-performance GPU inference; TensorRT reduces latency and Triton serves models efficiently. |
| KServe | Wraps Triton with Kubernetes-native model serving, canary deployments, and scale-to-zero capabilities. |
| Evidently + Prometheus | Monitors live predictions, detects data/model drift, and generates alerts when distribution shifts occur. |
| Alertmanager → Airflow | Automates retraining by triggering Airflow workflows when drift alerts are detected. |
| Observability Stack | Prometheus/Grafana (metrics), Loki (logs), Jaeger (traces), and DCGM (GPU monitoring) provide end-to-end visibility. |
| Terraform + ArgoCD | Terraform provisions cloud infrastructure, while ArgoCD enforces GitOps and keeps Kubernetes aligned with Git. |
