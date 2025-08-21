import os
import subprocess
import google.cloud.aiplatform as aip

# --- Configuration ---
# GCP Project and Service Account Details
PROJECT_ID = "dylora" 
REGION = "us-central1"
SERVICE_ACCOUNT_JSON = "dylora-ece7d9f1337d.json"

# Docker and Artifact Registry Details
DOCKER_REPO_NAME = "dy-lora-training-repo"
IMAGE_NAME = "dy-lora-training-image"
IMAGE_TAG = "latest"
IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{DOCKER_REPO_NAME}/{IMAGE_NAME}:{IMAGE_TAG}"

# Cloud Storage Bucket for Training Outputs
TRAINING_BUCKET = "gs://dy-lora-training-bucket"

# Vertex AI Training Job Details
JOB_DISPLAY_NAME = "dy-lora-custom-training-job"
MACHINE_TYPE = "n1-standard-4"
ACCELERATOR_TYPE = "NVIDIA_TESLA_T4"
ACCELERATOR_COUNT = 1

from typing import List

def run_command(command: List[str]) -> str:
    """Executes a shell command and prints its output."""
    print(f"Executing command: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error executing command: {' '.join(command)}")
        print(f"Stderr: {stderr}")
        raise RuntimeError(stderr)
    print(stdout)
    return stdout

def main():
    """Main function to orchestrate the training pipeline."""
    # --- 1. Authenticate with gcloud ---
    print("Authenticating gcloud with the service account...")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_JSON
    run_command(["gcloud", "auth", "activate-service-account", f"--key-file={SERVICE_ACCOUNT_JSON}"])
    run_command(["gcloud", "config", "set", "project", PROJECT_ID])

    # --- 2. Build and Push Docker Image ---
    print("Building and pushing the Docker image...")
    try:
        run_command(["gcloud", "builds", "submit", "--tag", IMAGE_URI, "."])
    except RuntimeError as e:
        print(f"Docker build and push failed. Please check permissions and configurations.")
        print(e)
        return

    # --- 3. Initialize Vertex AI SDK ---
    aip.init(project=PROJECT_ID, location=REGION, staging_bucket=TRAINING_BUCKET)

    # --- 4. Submit the Training Job ---
    print("Submitting the training job...")
    job = aip.CustomJob(
        display_name=JOB_DISPLAY_NAME,
        worker_pool_specs=[
            {
                "machine_spec": {
                    "machine_type": MACHINE_TYPE,
                    "accelerator_type": ACCELERATOR_TYPE,
                    "accelerator_count": ACCELERATOR_COUNT,
                },
                "replica_count": 1,
                "container_spec": {
                    "image_uri": IMAGE_URI,
                    "command": [],
                    "args": [],
                },
            }
        ],
        base_output_dir=TRAINING_BUCKET,
    )

    try:
        job.run(
            service_account=f"dylora-moe@{PROJECT_ID}.iam.gserviceaccount.com"
        )
        print("Training job submitted successfully.")
        print(f"Training Output directory: \n{job.output_dir}")
    except Exception as e:
        print(f"Failed to submit training job: {e}")

if __name__ == "__main__":
    main()