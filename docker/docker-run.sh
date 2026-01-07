#!/bin/bash
# Docker run helper script
# Usage: ./docker/docker-run.sh [command] [args...]
# Examples:
#   ./docker/docker-run.sh train cql
#   ./docker/docker-run.sh train mappo experiment.num_episodes=1000
#   ./docker/docker-run.sh eval
#   ./docker/docker-run.sh bash

set -e

# Build image if it doesn't exist
IMAGE_NAME="halite-marl"
if ! docker image inspect $IMAGE_NAME >/dev/null 2>&1; then
    echo "Building Docker image..."
    docker build -t $IMAGE_NAME .
fi

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Handle special commands (train, eval) by routing to their scripts
if [ "$1" = "train" ]; then
    shift  # Remove "train" from arguments
    # Call train.sh with remaining arguments
    docker run --rm -it \
        -v "$PROJECT_ROOT:/app" \
        -v "$PROJECT_ROOT/outputs:/app/outputs" \
        -v "$PROJECT_ROOT/models:/app/models" \
        -v "$PROJECT_ROOT/wandb:/app/wandb" \
        -w /app \
        $IMAGE_NAME \
        ./docker/train.sh "$@"
elif [ "$1" = "eval" ]; then
    shift  # Remove "eval" from arguments
    # Call eval.sh with remaining arguments
    docker run --rm -it \
        -v "$PROJECT_ROOT:/app" \
        -v "$PROJECT_ROOT/outputs:/app/outputs" \
        -v "$PROJECT_ROOT/models:/app/models" \
        -v "$PROJECT_ROOT/wandb:/app/wandb" \
        -w /app \
        $IMAGE_NAME \
        ./docker/eval.sh "$@"
else
    # Run command directly in container
    docker run --rm -it \
        -v "$PROJECT_ROOT:/app" \
        -v "$PROJECT_ROOT/outputs:/app/outputs" \
        -v "$PROJECT_ROOT/models:/app/models" \
        -v "$PROJECT_ROOT/wandb:/app/wandb" \
        -w /app \
        $IMAGE_NAME \
        "$@"
fi



