from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id = "e26f9f9f-61b5-409c-ab99-2a2041c63774",
    resource_group_name = "MLOPS",
    workspace_name= "VehiclePredictiveMaintenanceMLOps",
)

workspace = ml_client.workspaces.get(ml_client.workspace_name)
print(workspace.mlflow_tracking_uri)