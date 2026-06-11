# scripts/setup/configure-mig.sh
# Run this when switching a node between training and serving mode

#!/bin/bash
set -euo pipefail

NODE_NAME=$1
MODE=$2  # training-mode or serving-mode

echo "Configuring MIG mode: $MODE on node: $NODE_NAME"

# Step 1: Cordon the node — no new pods schedule here
kubectl cordon $NODE_NAME

# Step 2: Drain existing pods gracefully
kubectl drain $NODE_NAME \
  --ignore-daemonsets \
  --delete-emptydir-data \
  --grace-period=60

# Step 3: Apply MIG configuration
kubectl label node $NODE_NAME \
  nvidia.com/mig.config=$MODE \
  --overwrite

# Step 4: Wait for MIG reconfiguration
echo "Waiting for MIG reconfiguration..."
sleep 30

# Step 5: Verify MIG slices
kubectl exec -n gpu-operator \
  $(kubectl get pods -n gpu-operator -l app=nvidia-mig-manager -o jsonpath='{.items[0].metadata.name}') \
  -- nvidia-smi mig -lgip

# Step 6: Uncordon — allow pods back
kubectl uncordon $NODE_NAME

echo "MIG configuration complete. Node ready."