# Docker Setup for Halite MARL

This directory contains Docker configuration and one-liner scripts for training and evaluation.

## Quick Start

### Build Docker Image

```bash
docker build -t halite-marl .
```

### One-Liner Training

Train using the Docker helper script:

```bash
# Train CQL
./docker/docker-run.sh train cql

# Train MAPPO with custom config
./docker/docker-run.sh train mappo experiment.num_episodes=1000

# Train IQL on GPU (if available)
./docker/docker-run.sh train iql device=cuda

# Train IPPO
./docker/docker-run.sh train ippo
```

Or use the scripts directly inside the container:

```bash
# Enter container
./docker/docker-run.sh bash

# Then run training
./docker/train.sh cql
./docker/train.sh mappo experiment.num_episodes=1000
```

### One-Liner Evaluation

```bash
# Run evaluation
./docker/docker-run.sh eval evaluate-model --model-folder models/mappo_98137A

# With additional options
./docker/docker-run.sh eval evaluate-model --model-folder models/mappo_98137A --checkpoint-name ep18000.pt
```

### Direct Python Commands

You can also run Python commands directly:

```bash
# Training
./docker/docker-run.sh python -m src.training.train_cql
./docker/docker-run.sh python -m src.training.train_mappo

# Evaluation
./docker/docker-run.sh python -m src.eval.evaluate_model --model-folder models/mappo_98137A
```

## Local Scripts (without Docker)

If you have the environment set up locally, you can use the scripts directly:

```bash
# Training
./docker/train.sh cql
./docker/train.sh mappo experiment.num_episodes=1000

# Evaluation
./docker/eval.sh
```

## Volume Mounts

The `docker-run.sh` script automatically mounts:
- Project root (`/app`)
- `outputs/` directory (for checkpoints and logs)
- `models/` directory (for saved models)
- `wandb/` directory (for Weights & Biases logs)

All outputs will be saved to your local filesystem.

## Dependencies

The Docker image uses pinned dependencies from `requirements-lock.txt` for reproducible builds. To update dependencies:

1. Update `requirements.txt` with new versions
2. Install and test locally
3. Run `pip freeze | grep -E "(numpy|gymnasium|torch|...)"` to get pinned versions
4. Update `requirements-lock.txt`
5. Rebuild Docker image





