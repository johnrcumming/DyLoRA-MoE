import os
from google.cloud import aiplatform

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

# Authenticate gcloud with the service account
print("Authenticating gcloud with the service account...")
os.system(f"gcloud auth activate-service-account --key-file={os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")


# Build and push the Docker image using Cloud Build
print("Building and pushing the Docker image...")
os.system(f"gcloud builds submit --tag {REGION}-docker.pkg.dev/{PROJECT_ID}/{DOCKER_REPO_NAME}/{IMAGE_NAME}:{IMAGE_TAG}")

# Define the custom training job
job = aiplatform.CustomContainerTrainingJob(
    display_name="dy-lora-training-job",
    container_uri=f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{DOCKER_REPO_NAME}/{IMAGE_NAME}:{IMAGE_TAG}",
    command=["python", "train.py"],
)

# Run the training job
print("Submitting the training job...")
job.run(
    machine_type="a2-highgpu-1g",
    accelerator_type="NVIDIA_TESLA_A100",
    accelerator_count=1,
)

print("Training job submitted successfully.")