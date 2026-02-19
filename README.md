# GNN-Network

A multi-agent reinforcement learning (MARL) system that combines **Graph Neural Networks (GNNs)** with **Deep Q-Learning (DQN)** to train cooperative agents to navigate a grid environment and collect goals. Agents share relational information through a Transformer-based GNN, enabling coordinated decision-making.

## Overview

Eight agents operate on a 31×31 grid and must collectively collect 43 goals. Each agent observes nearby entities (other agents and goals) as a graph, which is processed by a GNN to produce Q-values for action selection. The system uses Double DQN with Prioritized Experience Replay (PER) for stable and sample-efficient training.

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING LOOP (main.py)                  │
│       Episodes → Actions → Env Steps → Experience           │
└──────────────────┬──────────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
   ┌──────────────┐    ┌──────────────┐
   │  GR_QNetwork │    │ ReplayBuffer │
   │  (DQN + GNN) │    │    (PER)     │
   └──────┬───────┘    └──────────────┘
          │
   ┌──────▼────────────────────────────┐
   │       TransformerConvNet          │
   │  EmbedConv → TransformerConv ×2   │
   │  → Graph Filtering → Q-Values     │
   └──────┬────────────────────────────┘
          │
   ┌──────▼──────────────────┐
   │     MultiGridEnv        │
   │  31×31 grid, 8 agents   │
   │  43 goals, actions: LRUD│
   └─────────────────────────┘
```

## Architecture

### GNN (gnn.py)

The `TransformerConvNet` processes each agent's local graph:

1. **EmbedConv** — Custom message-passing layer. Concatenates node features with learned entity-type embeddings and edge attributes (distances), then passes through an MLP.
2. **TransformerConv ×2** — Multi-head (3) self-attention over the graph. Agents learn to weight neighbors by relevance.
3. **Graph Filtering (`processAdj`)** — Dynamically prunes the graph:
   - Keeps top-4 closest agent neighbors
   - Prunes agent-goal edges beyond distance 4.5
   - Zeroes non-agent-to-non-agent connections

The `GNNBase` appends the GNN output to the agent's local observation, then applies two FC layers to produce 4 Q-values (one per action).

### DQN (network.py)

- **Double DQN**: Action selection uses the local network; value estimation uses the target network.
- **Prioritized Experience Replay (PER)**: Samples transitions proportional to `|TD-error|^α`. Importance-sampling weights correct for bias. Beta is annealed from 0.4 → 1.0.
- **Soft target updates**: Target network parameters are slowly blended toward local network weights using `τ = 0.001`.

### Environment (minigrid/)

- **Grid**: 31×31 with wall borders. Agents are placed along the edges; goals are randomly distributed.
- **Observations** (per agent):
  - *Agent obs* (8 values): Agent position + positions of 3 nearest uncollected goals
  - *Node obs*: Feature vectors for all visible entities (relative position, goal info, entity type)
- **Adjacency matrix**: Pairwise distances between all entities, filtered in `processAdj`
- **Rewards**: `+10` for collecting a goal, `-5` for an invalid move
- **Episode ends**: After 175 steps or when all 43 goals are collected

## Project Structure

```
GNN-Network/
├── main.py               # Training loop and entry point
├── args.py               # All hyperparameters and configuration
├── network.py            # DQN, Double DQN, PER replay buffer
├── gnn.py                # TransformerConvNet and GNNBase
├── logger.py             # Weights & Biases logging
├── utils.py              # Constants, enums, helpers
├── script.sh             # SLURM batch job submission script
└── minigrid/
    ├── environment.py    # MultiGridEnv (core RL environment)
    ├── grid.py           # Grid data structure
    ├── world.py          # World objects (Agent, Goal, Obstacle, Wall)
    └── rendering.py      # Pygame rendering utilities
```

## Requirements

- Python 3.8+
- PyTorch
- PyTorch Geometric
- Gymnasium
- NumPy
- Pygame
- Weights & Biases (`wandb`)

Install dependencies:

```bash
pip install torch torch-geometric gymnasium numpy pygame wandb
```

## Usage

**Run training:**

```bash
python main.py
```

**Submit to SLURM cluster (HPC):**

```bash
sbatch script.sh
```

The SLURM script requests 1 A100 GPU, 50 GB RAM, and a 72-hour wall time.

## Configuration

All settings are in `args.py`. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `grid_size` | 31 | Grid dimensions |
| `num_agents` | 8 | Number of cooperative agents |
| `num_goals` | 43 | Goals to collect per episode |
| `total_steps` | 500,000 | Training budget |
| `max_steps` | 175 | Max steps per episode |
| `eps_start` / `eps_end` | 1.0 / 0.01 | Epsilon-greedy exploration range |
| `eps_percentage` | 90% | % of training over which epsilon decays |
| `lr` | 0.0005 | Adam learning rate |
| `gamma` | 0.99 | Discount factor |
| `tau` | 0.001 | Target network soft-update rate |
| `batch_size` | 64 | Replay buffer sample size |
| `buffer_size` | 100,000 | Experience replay capacity |
| `double_dqn` | True | Enable Double DQN |
| `priority_replay` | True | Enable PER |
| `gnn_hidden_size` | 16 | GNN output dimension |
| `gnn_num_heads` | 3 | Transformer attention heads |
| `gnn_layer_N` | 2 | Number of TransformerConv layers |
| `agent_connections` | 4 | Max agent-agent graph connections |
| `max_edge_dist` | 4.5 | Max agent-goal edge distance |
| `device` | `cuda` | Compute device |
| `render` | False | Enable Pygame visualization |
| `logger` | True | Enable W&B logging |
| `debug` | False | Manual interactive control mode |

## Outputs

- **Model checkpoints**: Saved as `{title}.pt` (PyTorch state dict) at every 1% of training progress.
- **W&B metrics**: Episodes, steps, epsilon, average reward, loss, goals collected, grid coverage, throughput (steps/sec). Project name: `GNN-NODE-500k`.

## Algorithm Summary

```
Double DQN target:
  target = r + γ · Q_target(s', argmax_a Q_local(s', a))
  loss   = importance_weight · (Q_local(s, a) − target)²

Prioritized replay:
  priority          = |TD-error| + ε
  sampling_prob    ∝ priority^α       (α = 0.5)
  importance_weight = (1 / (N · priority))^β   (β: 0.4 → 1.0)
```
