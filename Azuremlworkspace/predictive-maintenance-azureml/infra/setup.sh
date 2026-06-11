#!/bin/bash

# Setup script for Vehicle Predictive Maintenance MLOps on Azure ML

set -e

echo "Starting Azure ML infrastructure setup..."

# Variables
RESOURCE_GROUP="vehicle-maintenance-rg"
WORKSPACE_NAME="vehicle-maintenance-ws"
LOCATION="eastus"
STORAGE_ACCOUNT="vehiclemaintenance"
COMPUTE_NAME="cpu-cluster"

echo "Creating resource group: $RESOURCE_GROUP"
az group create --name $RESOURCE_GROUP --location $LOCATION

echo "Creating Azure ML workspace: $WORKSPACE_NAME"
az ml workspace create \
  --resource-group $RESOURCE_GROUP \
  --name $WORKSPACE_NAME \
  --location $LOCATION

echo "Creating storage account: $STORAGE_ACCOUNT"
az storage account create \
  --resource-group $RESOURCE_GROUP \
  --name $STORAGE_ACCOUNT \
  --location $LOCATION \
  --sku Standard_LRS

echo "Creating compute cluster: $COMPUTE_NAME"
az ml compute create \
  --resource-group $RESOURCE_GROUP \
  --workspace-name $WORKSPACE_NAME \
  --name $COMPUTE_NAME \
  --type amlcompute \
  --min-instances 0 \
  --max-instances 4

echo "Infrastructure setup complete!"
echo "Workspace: $WORKSPACE_NAME"
echo "Resource Group: $RESOURCE_GROUP"
