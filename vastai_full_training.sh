#!/bin/bash
set -e

# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found"
    exit 1
fi

# Check required environment variables
if [ -z "$WANDB_API_KEY" ]; then
    echo "Error: WANDB_API_KEY not found in .env"
    exit 1
fi

if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN not found in .env"
    exit 1
fi

# Optional: Check for DOCKER_TOKEN (for private repos)
if [ -z "$DOCKER_TOKEN" ]; then
    echo "Warning: DOCKER_TOKEN not found in .env (only needed for private Docker repos)"
fi

# Check if OFFER_ID is provided as argument
if [ -z "$1" ]; then
    echo "Usage: $0 <OFFER_ID>"
    echo "Example: $0 12345678"
    exit 1
fi

OFFER_ID=$1

echo "Creating Vast.ai instance with offer ID: $OFFER_ID"
echo "Using Docker image: johnrcumming001/dylora-moe:latest"

# Create Vast.ai instance with environment variables from .env
# Note: The entrypoint.py will automatically detect Vast.ai and apply appropriate defaults
vastai create instance "$OFFER_ID" \
    --image johnrcumming001/dylora-moe:latest \
    --env "-e WANDB_API_KEY=$WANDB_API_KEY -e HF_TOKEN=$HF_TOKEN -e VAST_INSTANCE_ID=$OFFER_ID" \
    --disk 200 \
    --login "-u johnrcumming001 -p $DOCKER_TOKEN docker.io" \
    --args "--train --datasets code_alpaca,mbpp,evol_instruct,code_feedback --bf16 --num_epochs 10 --num_experts 2"

echo ""
echo "Instance created! The container will:"
echo "  1. Auto-detect Vast.ai platform"
echo "  2. Apply conservative batch sizes for Vast.ai hardware"
echo "  3. Start training with DyLoRA-MoE"
echo ""
echo "To monitor progress:"
echo "  vastai logs <instance_id>"
echo ""
echo "To run benchmarking instead:"
echo "  vastai ssh <instance_id> 'python entrypoint.py --benchmark --model-path ./results/best_model --max-samples 164'"

