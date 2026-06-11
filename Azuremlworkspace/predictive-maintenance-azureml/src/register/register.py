"""
Model registration script for Azure ML
"""
import argparse
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential

def register_model(model_path, model_name, model_version, workspace_config):
    """
    Register a trained model to Azure ML
    
    Args:
        model_path: Path to the trained model
        model_name: Name for the registered model
        model_version: Version of the model
        workspace_config: Azure ML workspace configuration
    """
    try:
        # Initialize ML Client
        credential = DefaultAzureCredential()
        ml_client = MLClient.from_config(credential=credential)
        
        # Register model
        model = Model(
            path=model_path,
            name=model_name,
            version=model_version,
            type="custom_model",
            description="Vehicle Predictive Maintenance Model"
        )
        
        registered_model = ml_client.models.create_or_update(model)
        print(f"Model registered successfully: {registered_model.id}")
        
    except Exception as e:
        print(f"Error registering model: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register model to Azure ML")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--name", type=str, default="vehicle-maintenance-model", help="Model name")
    parser.add_argument("--version", type=str, default="1", help="Model version")
    
    args = parser.parse_args()
    register_model(args.model, args.name, args.version, None)
