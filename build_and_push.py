#!/usr/bin/env python3
"""
Python script to build and push Docker images for DyLoRA-MoE project.
Builds and pushes to GCP Artifact Registry and/or Docker Hub based on command line arguments.
"""

import argparse
import subprocess
import sys
import os
from typing import Union, Any


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_colored(message: str, color: str = Colors.WHITE) -> None:
    """Print a message with color."""
    print(f"{color}{message}{Colors.RESET}")


def run_command(command: list[str], check: bool = True) -> Union[subprocess.CompletedProcess[str], subprocess.CalledProcessError]:
    """Run a shell command and handle errors."""
    try:
        result = subprocess.run(command, check=check, capture_output=False, text=True)
        return result
    except subprocess.CalledProcessError as e:
        if check:
            print_colored(f"Command failed: {' '.join(command)}", Colors.YELLOW)
            raise
        return e


def get_git_hash() -> str:
    """Get the short git hash of the current commit."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        print_colored("Error: Failed to get git hash", Colors.YELLOW)
        sys.exit(1)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build and push Docker images for DyLoRA-MoE project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s                     # Build and push to both registries
  %(prog)s --gcp-only         # Build and push only to GCP Artifact Registry
  %(prog)s --dockerhub-only   # Build and push only to Docker Hub
  %(prog)s --build-only       # Build images but don't push anywhere"""
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--gcp-only",
        action="store_true",
        help="Push only to GCP Artifact Registry"
    )
    group.add_argument(
        "--dockerhub-only",
        action="store_true",
        help="Push only to Docker Hub"
    )
    group.add_argument(
        "--build-only",
        action="store_true",
        help="Build images but don't push to any registry"
    )
    
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip building and only push existing images"
    )
    
    return parser.parse_args()


def main():
    """Main function to build and push Docker images."""
    
    args = parse_arguments()
    
    # Get git hash and set environment variable
    git_hash = get_git_hash()
    os.environ['GIT_HASH'] = git_hash
    
    # Configuration
    PROJECT_ID = "dylora"
    REGION = "us-central1"
    DOCKER_REPO_NAME = "dy-lora-training-repo"
    IMAGE_NAME = "dy-lora-training-image"
    
    # Docker Hub configuration
    DOCKER_HUB_USERNAME = os.environ.get('DOCKER_HUB_USERNAME', 'johnrcumming001')
    DOCKER_HUB_IMAGE = f"{DOCKER_HUB_USERNAME}/dylora-moe"
    
    # Full image URIs
    GCP_IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{DOCKER_REPO_NAME}/{IMAGE_NAME}"
    
    # Build Docker image with appropriate tags based on arguments
    if not args.skip_build:
        print_colored("Building the Docker image for linux/amd64...", Colors.GREEN)
        
        build_command = ["docker", "build", "--platform", "linux/amd64"]
        
        # Add tags based on what we're going to push
        if not args.dockerhub_only:
            build_command.extend(["-t", f"{GCP_IMAGE_URI}:{git_hash}"])
            build_command.extend(["-t", f"{GCP_IMAGE_URI}:latest"])
        
        if not args.gcp_only:
            build_command.extend(["-t", f"{DOCKER_HUB_IMAGE}:{git_hash}"])
            build_command.extend(["-t", f"{DOCKER_HUB_IMAGE}:latest"])
        
        build_command.append(".")
        
        try:
            run_command(build_command)
        except subprocess.CalledProcessError:
            print_colored("Docker build failed", Colors.YELLOW)
            sys.exit(1)
    else:
        print_colored("Skipping build - using existing images", Colors.CYAN)
    
    # Push to Google Cloud Artifact Registry (if not disabled)
    if not args.dockerhub_only and not args.build_only:
        print_colored("Pushing the Docker image to GCP Artifact Registry...", Colors.GREEN)
        
        try:
            run_command(["docker", "push", f"{GCP_IMAGE_URI}:{git_hash}"])
            run_command(["docker", "push", f"{GCP_IMAGE_URI}:latest"])
        except subprocess.CalledProcessError:
            print_colored("Failed to push to GCP Artifact Registry", Colors.YELLOW)
            sys.exit(1)
    
    # Push to Docker Hub (if not disabled)
    if not args.gcp_only and not args.build_only:
        print_colored("Pushing the Docker image to Docker Hub...", Colors.GREEN)
        
        try:
            run_command(["docker", "push", f"{DOCKER_HUB_IMAGE}:{git_hash}"])
            run_command(["docker", "push", f"{DOCKER_HUB_IMAGE}:latest"])
            print_colored("✓ Docker image pushed to Docker Hub successfully.", Colors.GREEN)
        except subprocess.CalledProcessError:
            print_colored("⚠ Warning: Failed to push to Docker Hub (may need to run 'docker login')", Colors.YELLOW)
            if not args.dockerhub_only:
                print_colored("  Continuing anyway - GCP push succeeded.", Colors.YELLOW)
            else:
                sys.exit(1)
    
    # Print summary
    print()
    print_colored("================================", Colors.CYAN)
    if args.build_only:
        print_colored("Docker image build complete!", Colors.CYAN)
    elif args.skip_build:
        print_colored("Docker image push complete!", Colors.CYAN)
    else:
        print_colored("Docker image build and push complete!", Colors.CYAN)
    print_colored("================================", Colors.CYAN)
    
    if not args.dockerhub_only:
        print_colored("GCP Artifact Registry:", Colors.WHITE)
        print_colored(f"  - {GCP_IMAGE_URI}:{git_hash}", Colors.GRAY)
        print_colored(f"  - {GCP_IMAGE_URI}:latest", Colors.GRAY)
        if not args.gcp_only:
            print()
    
    if not args.gcp_only:
        print_colored("Docker Hub:", Colors.WHITE)
        print_colored(f"  - {DOCKER_HUB_IMAGE}:{git_hash}", Colors.GRAY)
        print_colored(f"  - {DOCKER_HUB_IMAGE}:latest", Colors.GRAY)
    
    print_colored("================================", Colors.CYAN)


if __name__ == "__main__":
    main()