#!/bin/bash
# Evaluation script wrapper
# Usage:
#   ./docker/eval.sh evaluate-model --model-folder <folder> [options...]

set -e

# Check if first argument is a subcommand
if [ "$1" = "evaluate-model" ]; then
    shift
    python -m src.eval.evaluate_model "$@"

else
    echo "Usage:"
    echo "  ./docker/eval.sh evaluate-model --model-folder <folder> [options...]"
    exit 1
fi

