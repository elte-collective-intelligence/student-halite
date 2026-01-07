# Hydra Configuration System

This directory contains Hydra configuration files for reproducible training experiments.

## Structure

```
configs/
├── config.yaml              # Main config file (default)
├── env/
│   └── halite.yaml          # Environment configuration
├── algo/
│   ├── centralized_qlearning.yaml
│   ├── iql.yaml
│   ├── ippo.yaml
│   └── mappo.yaml
└── experiment/
    ├── default.yaml         # Default experiment settings
    └── short.yaml           # Short experiment for testing
```

## Usage

### Basic Usage

Train with default configuration:
```bash
# From src/algo/ directory
python train_centralized_qlearning.py
```

### Override Configuration

Override specific parameters from command line:
```bash
# Change algorithm
python train_centralized_qlearning.py algo=iql

# Change experiment type
python train_centralized_qlearning.py experiment=short

# Override specific parameters
python train_centralized_qlearning.py algo.lr=0.0002 experiment.num_episodes=500

# Override environment
python train_centralized_qlearning.py env.grid_size=[10,10] env.num_agents=3

# Change device
python train_centralized_qlearning.py device=cuda
```

### Train Different Algorithms

```bash
# Centralized Q-Learning
python train_centralized_qlearning.py algo=centralized_qlearning

# Independent Q-Learning
python train_centralized_qlearning.py algo=iql algo.shared_network=true

# Independent PPO
python train_ippo.py algo=ippo algo.shared_weights=false

# Multi-Agent PPO
python train_mappo.py algo=mappo algo.shared_actor=true
```

### Custom Configuration

Create a new config file (e.g., `configs/experiment/custom.yaml`):
```yaml
num_episodes: 2000
log_frequency: 20
save_frequency: 200
```

Then use it:
```bash
python train_centralized_qlearning.py experiment=custom
```

### Output Location

Hydra saves outputs (logs, configs) to:
```
outputs/YYYY-MM-DD_HH-MM-SS/
```

The working directory is preserved, so models are saved to the configured paths.

## Configuration Files

### Environment (`env/halite.yaml`)
- `num_agents`: Number of agents
- `grid_size`: Grid dimensions [height, width]
- `max_steps`: Maximum steps per episode
- `generator`: 'uniform' or 'original'

### Algorithm (`algo/*.yaml`)
Each algorithm has its specific hyperparameters:
- Learning rates
- Discount factors
- Buffer sizes
- Network sharing options
- etc.

### Experiment (`experiment/*.yaml`)
- `num_episodes`: Total training episodes
- `log_frequency`: How often to print progress
- `save_frequency`: How often to save checkpoints
- `model_save_path`: Where to save final model
- `checkpoint_path`: Where to save checkpoints

## Reproducibility

Set a fixed seed:
```bash
python train_centralized_qlearning.py seed=42
```

The seed is used for:
- NumPy random number generator
- PyTorch random number generator
- CUDA (if available)

## Tips

1. **Quick testing**: Use `experiment=short` for quick validation
2. **Override multiple values**: Combine multiple overrides with spaces
3. **Save configs**: Hydra automatically saves the used config in the output directory
4. **Resume training**: Load checkpoints from the checkpoint directory


