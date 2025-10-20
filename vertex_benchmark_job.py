#!/usr/bin/env python3
"""
Vertex AI Benchmark Job Submission

Submit benchmarking jobs to Google Cloud Vertex AI using the unified entrypoint.
"""

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

# Get environment variables
wandb_api_key = os.environ.get("WANDB_API_KEY")
if not wandb_api_key:
    raise ValueError("WANDB_API_KEY environment variable not set.")

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN environment variable not set.")

def submit_benchmark_job(
    model_path=None,
    wandb_artifact=None,
    max_samples=164,
    benchmarks="humaneval",
    temperature=0.2,
    max_tokens=256,
    job_name="dylora-benchmark-job"
):
    """
    Submit a benchmarking job to Vertex AI.
    
    Args:
        model_path: Local path to model (if benchmarking local model)
        wandb_artifact: W&B artifact path (e.g., "user/project/model:v0") 
        max_samples: Maximum number of samples to evaluate
        benchmarks: Comma-separated list of benchmarks to run
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        job_name: Name for the Vertex AI job
    """
    
    # Build benchmark command
    benchmark_args = [
        "python", "entrypoint.py", "--benchmark",
        "--benchmarks", benchmarks,
        "--max-samples", str(max_samples),
        "--temperature", str(temperature),
        "--max-tokens", str(max_tokens),
    ]
    
    # Add model source
    if wandb_artifact:
        benchmark_args.extend(["--wandb-artifact", wandb_artifact])
    elif model_path:
        benchmark_args.extend(["--model-path", model_path])
    else:
        # Default to base model benchmark
        benchmark_args.extend(["--base-model"])
    
    # Define the worker pool spec
    worker_pool_specs = [
        {
            "machine_spec": {
                # Use A100 for benchmarking (cheaper than H100, sufficient for inference)
                "machine_type": "a2-highgpu-1g",
                "accelerator_type": "NVIDIA_TESLA_A100",
                "accelerator_count": 1,
            },
            "replica_count": 1,
            "disk_spec": {
                "boot_disk_type": "pd-ssd",
                "boot_disk_size_gb": 100,  # Smaller disk for benchmarking
            },
            "container_spec": {
                "image_uri": f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{DOCKER_REPO_NAME}/{IMAGE_NAME}:{IMAGE_TAG}",
                "command": benchmark_args,
                "args": [],
                "env": [
                    {"name": "WANDB_API_KEY", "value": wandb_api_key},
                    {"name": "HF_TOKEN", "value": hf_token},
                ],
            },
        }
    ]
    
    # Create and submit the job
    job = aiplatform.CustomJob(
        display_name=job_name,
        worker_pool_specs=worker_pool_specs,
    )
    
    print(f"Submitting benchmark job: {job_name}")
    print(f"Model source: {'W&B artifact: ' + wandb_artifact if wandb_artifact else 'Local path: ' + (model_path or 'base model')}")
    print(f"Benchmarks: {benchmarks}")
    print(f"Max samples: {max_samples}")
    
    job.run(
        service_account=f"dylora-moe@{PROJECT_ID}.iam.gserviceaccount.com",
        timeout=7200,  # 2 hours should be enough for most benchmarks
        restart_job_on_worker_restart=True,
    )
    
    print("Benchmark job completed!")
    print(f"Final state: {job.state}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Submit benchmark job to Vertex AI")
    parser.add_argument("--model-path", type=str, help="Local path to model")
    parser.add_argument("--wandb-artifact", type=str, help="W&B artifact path (e.g., 'user/project/model:v0')")
    parser.add_argument("--max-samples", type=int, default=164, help="Maximum samples to evaluate")
    parser.add_argument("--benchmarks", type=str, default="humaneval", help="Benchmarks to run")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum tokens to generate")
    parser.add_argument("--job-name", type=str, default="dylora-benchmark-job", help="Job name")
    
    args = parser.parse_args()
    
    submit_benchmark_job(
        model_path=args.model_path,
        wandb_artifact=args.wandb_artifact,
        max_samples=args.max_samples,
        benchmarks=args.benchmarks,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        job_name=args.job_name
    )