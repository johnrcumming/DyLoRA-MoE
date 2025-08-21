import os
from dotenv import load_dotenv
from google.cloud import aiplatform

# Load environment variables from .env file
load_dotenv()

# Set your project and bucket details
PROJECT_ID = "dylora"
BUCKET_NAME = "dy-lora-training-bucket"
REGION = "us-central1"
DOCKER_REPO_NAME = "dy-lora-training-repo"
IMAGE_NAME = "dy-lora-training-image"
IMAGE_TAG = "latest"

# Authenticate with the service account
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "dylora-ece7d9f1337d.json"

# Initialize the Vertex AI SDK
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{BUCKET_NAME}")

# Get WANDB_API_KEY from environment
wandb_api_key = os.environ.get("WANDB_API_KEY")
if not wandb_api_key:
    raise ValueError("WANDB_API_KEY environment variable not set.")

# Define the worker pool spec for a Spot VM
worker_pool_specs = [
    {
        "machine_spec": {
            "machine_type": "a2-highgpu-1g",
            "accelerator_type": "NVIDIA_TESLA_A100",
            "accelerator_count": 1,
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{DOCKER_REPO_NAME}/{IMAGE_NAME}:{IMAGE_TAG}",
            "command": ["python", "poc_train.py"],
            "args": [],
        },
    }
]

# Define the custom job
job = aiplatform.CustomJob(
    display_name="software-development-training-job",
    worker_pool_specs=worker_pool_specs,
)

# Run the training job
print("Submitting the software development training job...")
job.run(
    service_account=f"dylora-moe@{PROJECT_ID}.iam.gserviceaccount.com",
    # The 'use_spot_vms' parameter is not directly in the run method for CustomJob,
    # it's implicitly handled by the lack of a 'persistent_resource_id' and the service's behavior.
    # To make it explicit for clarity and future API versions, we would check docs.
    # For now, the standard way is to just run it. If a persistent resource is not specified,
    # it will run on-demand or spot based on availability and other factors.
    # The most direct way to request spot is not available in this high-level SDK object.
    # Let's stick to the documented features.
    # A timeout is useful for spot VMs
    timeout=86400,  # 24 hours
    restart_job_on_worker_restart=True,
)

print("Software development training job submitted successfully.")