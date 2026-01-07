#!/bin/bash
# One-liner training script wrapper
# Usage: ./docker/train.sh [algorithm] [hydra_overrides...]
# Examples:
#   ./docker/train.sh cql
#   ./docker/train.sh mappo experiment.num_episodes=1000
#   ./docker/train.sh iql device=cuda

set -e

ALGORITHM=${1:-cql}
shift

# Map algorithm to training script
case $ALGORITHM in
    cql)
        SCRIPT="src.training.train_cql"
        ;;
    iql)
        SCRIPT="src.training.train_iql"
        ;;
    ippo)
        SCRIPT="src.training.train_ippo"
        ;;
    mappo)
        SCRIPT="src.training.train_mappo"
        ;;
    *)
        echo "Error: Unknown algorithm '$ALGORITHM'"
        echo "Supported algorithms: cql, iql, ippo, mappo"
        exit 1
        ;;
esac

# Run training with any additional Hydra overrides
python -m $SCRIPT "$@"



