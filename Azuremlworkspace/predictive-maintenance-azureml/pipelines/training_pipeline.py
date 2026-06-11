"""
Training pipeline definition for vehicle predictive maintenance
"""
from azure.ai.ml import MLClient, load_component
from azure.ai.ml.dsl import pipeline
from azure.identity import DefaultAzureCredential

@pipeline(
    name="vehicle-maintenance-training",
    description="End-to-end training pipeline for vehicle maintenance prediction",
    default_compute="cpu-cluster"
)
def vehicle_maintenance_pipeline(raw_data_path: str):
    """
    Define the training pipeline
    
    Args:
        raw_data_path: Path to raw vehicle maintenance data
    """
    # Load components
    prep_component = load_component("../components/prep.yml")
    train_component = load_component("../components/train.yml")
    evaluate_component = load_component("../components/evaluate.yml")
    register_component = load_component("../components/register.yml")
    
    # Data preparation step
    prep_job = prep_component(raw_data=raw_data_path)
    
    # Training step
    train_job = train_component(processed_data=prep_job.outputs.processed_data)
    
    # Evaluation step
    eval_job = evaluate_component(
        model=train_job.outputs.model,
        test_data=prep_job.outputs.processed_data
    )
    
    # Registration step
    register_job = register_component(model=train_job.outputs.model)
    
    return {
        "pipeline_output": eval_job.outputs.evaluation_metrics
    }

def submit_pipeline(ml_client: MLClient, subscription_id: str, resource_group: str, 
                   workspace_name: str, raw_data_path: str):
    """
    Submit the pipeline for execution
    
    Args:
        ml_client: Azure ML client
        subscription_id: Azure subscription ID
        resource_group: Azure resource group
        workspace_name: Azure ML workspace name
        raw_data_path: Path to raw data
    """
    # Create pipeline job
    pipeline_job = vehicle_maintenance_pipeline(raw_data_path=raw_data_path)
    
    # Submit job
    submitted_job = ml_client.jobs.create_or_update(pipeline_job)
    print(f"Pipeline submitted: {submitted_job.id}")
    
    return submitted_job

if __name__ == "__main__":
    # Initialize client
    credential = DefaultAzureCredential()
    ml_client = MLClient.from_config(credential=credential)
    
    # Submit pipeline
    raw_data = "azureml://datasets/vehicle-maintenance/versions/1"
    submit_pipeline(ml_client, None, None, None, raw_data)
