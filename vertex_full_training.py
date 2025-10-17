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
# Initialize the Vertex AI SDK
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{BUCKET_NAME}")

# Authenticate with the service account
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "dylora-ece7d9f1337d.json"

# Get WANDB_API_KEY from environment
wandb_api_key = os.environ.get("WANDB_API_KEY")
if not wandb_api_key:
    raise ValueError("WANDB_API_KEY environment variable not set.")

# Get HF_TOKEN from environment
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN environment variable not set.")

# Optional: W&B artifact path for checkpoint resumption
# Format: "username/project/artifact:version" (e.g., "johnrcumming/dylo-moe-full-training/best-dylora-model-full:v0")
# Set to None to start fresh training
#WANDB_CHECKPOINT_ARTIFACT = "johnrcumming001/dylo-moe-full-training/best-dylora-model-full:v11"  # Change this to resume from a specific W&B artifact
WANDB_CHECKPOINT_ARTIFACT = None  # Start fresh training

# Define the worker pool spec for a Spot VM
worker_pool_specs = [
    {
        "machine_spec": {
            "machine_type": "a2-highgpu-1g",
            "accelerator_type": "NVIDIA_TESLA_A100",
            # Do Not use H100 GPU as the cost is too high
            #"machine_type": "a3-highgpu-1g", 
            #"accelerator_type": "NVIDIA_H100_80GB",
            "accelerator_count": 1,
        },
        "replica_count": 1,
        "disk_spec": {
            "boot_disk_type": "pd-ssd",  # SSD for faster I/O
            "boot_disk_size_gb": 200,    # Increase from default 100GB to 200GB
        },
        "container_spec": {
            "image_uri": f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{DOCKER_REPO_NAME}/{IMAGE_NAME}:{IMAGE_TAG}",
            "command": [
                "python", "train.py",
                "--datasets", "code_alpaca,mbpp,evol_instruct,code_feedback",  # Large-scale: ~230k examples
                "--bf16", 
                "--num_epochs", "10",
                "--num_experts", "2",
                "--balance_coefficient", "0.01",  # Load balancing loss
                "--cosine_restarts",  # LR scheduler with restarts
                "--train_batch_size", "2",  # Proven stable for MoE multi-expert forward passes
                "--eval_batch_size", "2",  # Conservative for evaluation memory
                "--gradient_accumulation_steps", "64",  # Effective batch = 128
                "--early_stopping_patience", "5",  # Increased patience for large-scale training
            ] + (["--wandb_checkpoint_artifact", WANDB_CHECKPOINT_ARTIFACT] if WANDB_CHECKPOINT_ARTIFACT else []),
            "args": [],
            "env": [
                {"name": "WANDB_API_KEY", "value": wandb_api_key},
                {"name": "HF_TOKEN", "value": hf_token},
            ],
        },
    }
]

# Define the custom job
job = aiplatform.CustomJob(
    display_name="full-software-development-training-job-spot",
    worker_pool_specs=worker_pool_specs,
)

# Run the training job
print("Submitting the software development training job...")
if WANDB_CHECKPOINT_ARTIFACT:
    print(f"📦 Resuming from W&B checkpoint: {WANDB_CHECKPOINT_ARTIFACT}")
else:
    print("🆕 Starting fresh training (no checkpoint)")
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

print("Job finished. Final state:")
print(job.state)