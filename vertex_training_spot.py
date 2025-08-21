import os
from dotenv import load_dotenv
from google.cloud import aiplatform

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
PROJECT_ID = "dylora"
REGION = "us-central1"
BUCKET_NAME = "dy-lora-training-bucket"
DOCKER_REPO_NAME = "dy-lora-training-repo"
IMAGE_NAME = "dy-lora-training-image"
IMAGE_TAG = "latest"
JOB_DISPLAY_NAME = "software-development-spot-training-job"
SERVICE_ACCOUNT = f"dylora-moe@{PROJECT_ID}.iam.gserviceaccount.com"
CREDENTIALS_FILE = "dylora-ece7d9f1337d.json"

# --- Initialization ---
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_FILE
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{BUCKET_NAME}")

# --- Get WANDB API Key ---
wandb_api_key = os.environ.get("WANDB_API_KEY")
if not wandb_api_key:
    raise ValueError("WANDB_API_KEY environment variable not set.")

# --- Define Worker Pool Spec ---
# This is the core of the configuration. We define the machine, container, and
# crucially, we can add `use_preemptible_vms=True` (the SDK's name for Spot VMs)
# in a future version if the API supports it. For now, not specifying a
# persistent_resource_id is the way to request on-demand or spot resources.
# The service will attempt to use Spot VMs if they are available.
container_image_uri = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{DOCKER_REPO_NAME}/{IMAGE_NAME}:{IMAGE_TAG}"

worker_pool_specs = [
    {
        "machine_spec": {
            "machine_type": "a2-highgpu-1g",
            "accelerator_type": "NVIDIA_TESLA_A100",
            "accelerator_count": 1,
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": container_image_uri,
            "command": ["python", "software_development_trainer.py"],
            "args": [],
            "env": [{"name": "WANDB_API_KEY", "value": wandb_api_key}],
        },
    }
]

# --- Create and Run the CustomJob ---
# We use CustomJob, which is more flexible than CustomContainerTrainingJob
job = aiplatform.CustomJob(
    display_name=JOB_DISPLAY_NAME,
    worker_pool_specs=worker_pool_specs,
)

print("Submitting the custom training job...")
# For CustomJob, you pass the service account and other run-time configs here.
# Using restart_job_on_worker_restart is best practice for spot/preemptible VMs.
job.run(
    service_account=SERVICE_ACCOUNT,
    restart_job_on_worker_restart=True,
)

print("Custom training job submitted successfully.")