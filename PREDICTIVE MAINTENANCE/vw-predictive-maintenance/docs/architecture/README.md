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


