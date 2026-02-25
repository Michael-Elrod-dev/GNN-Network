# GNN-MAPPO

Graph Neural Network Multi-Agent Proximal Policy Optimization for cooperative grid navigation.

This project takes the discrete 31×31 grid environment from the GNN project and replaces the Double DQN algorithm with **Graph MAPPO** — a Centralized Training, Decentralized Execution (CTDE) on-policy PPO algorithm adapted from [InforMARL](https://github.com/calimero-is-near/InforMARL). Both projects share the same GNN backbone (EmbedConv + TransformerConv).

---

## Task

8 agents navigate a 31×31 grid and must collectively collect 43 goal objects. Each episode lasts 175 steps. Agents receive a sparse reward of **+10** for collecting a goal and a penalty of **-5** for an invalid move (e.g. walking into a wall). There are no obstacles.

---

## Algorithm: Graph MAPPO (GR-MAPPO)

MAPPO is a multi-agent extension of PPO. The "graph" variant uses a GNN to process the graph-structured observation (agents and goals as nodes, distances as edge weights) before passing features into the policy MLP.

**CTDE:** During training, the critic sees a *centralized* observation (all 8 agents' local observations concatenated → 64-dim vector). During execution, each actor only uses its own local observation (8-dim) plus its GNN embedding.

### Training flow (one episode)

```
8 parallel envs  ─→  collect 175 steps each  ─→  GraphReplayBuffer
                                                        │
                                              compute GAE returns
                                                        │
                                          10 PPO epochs × 1 mini-batch
                                               ┌────────┴────────┐
                                           actor loss         critic loss
                                        (clipped ratio)    (Huber + ValueNorm)
```

---

## Architecture

### GNN Backbone (`gnn.py`)

Each agent observes up to 51 entities (8 agents + 43 goals) as a graph. The backbone has two stages:

1. **EmbedConv** — A message-passing layer that looks up a learned entity-type embedding (agent=0, goal=1) for each neighbour, concatenates it with the neighbour's features and the edge distance, then passes through linear layers.
2. **TransformerConv** — 1 + 2 = 3 attention-based graph convolution layers.

**Actor** uses `graph_aggr='node'`: extracts only the ego-agent's node embedding → **(batch, 16)**.
**Critic** uses `graph_aggr='global'`: mean-pools all node embeddings → **(batch, 16)**.

### Actor (`GR_Actor` in `actor_critic.py`)

```
node_obs (B, 51, 9)  ──→  GNNBase (aggr=node)  ──→  (B, 16)
                                                          │
obs (B, 8)  ────────────────────────────── cat ──→  (B, 24)
                                                          │
                                                    MLPBase  ──→  (B, 64)
                                                          │
                                                    ACTLayer ──→  actions (B, 1)
                                                                   log_probs (B, 1)
```

### Critic (`GR_Critic` in `actor_critic.py`)

```
node_obs (B, 51, 9)  ──→  GNNBase (aggr=global) ──→  (B, 16)
                                                           │
cent_obs (B, 64)  ──────────────────────────── cat ──→  (B, 80)
                                                           │
                                                     MLPBase  ──→  (B, 64)
                                                           │
                                                     Linear(1) ──→  values (B, 1)
```

### Node features (9 per entity)

| Feature | Description |
|---|---|
| `rel_pos_x` | x offset from ego agent |
| `rel_pos_y` | y offset from ego agent |
| `goal1_rel_x` | x offset to nearest goal 1 |
| `goal1_rel_y` | y offset to nearest goal 1 |
| `goal2_rel_x` | x offset to nearest goal 2 |
| `goal2_rel_y` | y offset to nearest goal 2 |
| `goal3_rel_x` | x offset to nearest goal 3 |
| `goal3_rel_y` | y offset to nearest goal 3 |
| `entity_type` | 0 = agent, 1 = goal |

---

## File Structure

```
GNN-MAPPO/
├── main.py             Entry point (if __name__ == "__main__" guard required on Windows)
├── args.py             All hyperparameters (env + PPO + GNN)
├── runner.py           Training loop, rollout collection, make_env() factory
├── buffer.py           On-policy rollout buffer (T×R×A tensors)
├── policy.py           GR_MAPPOPolicy: wraps actor+critic, Adam optimizers
├── trainer.py          GR_MAPPO: PPO gradient update (clip, GAE, ValueNorm)
├── actor_critic.py     GR_Actor + GR_Critic + ACTLayer + Categorical + MLPBase
├── gnn.py              EmbedConv + TransformerConvNet + GNNBase
├── utils.py            ValueNorm + env constants + NN/training utilities
├── env_wrappers.py     GraphSubprocVecEnv + GraphDummyVecEnv
├── logger.py           WandB logger
└── minigrid/
    ├── __init__.py
    ├── environment.py  MultiGridEnv (numpy-only, auto-reset, gym spaces)
    ├── grid.py         Grid tile rendering
    ├── world.py        WorldObj, Agent, Goal, Wall, Obstacle
    └── rendering.py    Pixel rendering helpers
```

---

## Requirements

```
torch >= 2.0
torch_geometric
numpy
gym
wandb
cloudpickle
```

Install PyTorch Geometric following the [official guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for your CUDA version. For everything else:

```bash
pip install wandb cloudpickle gym
```

---

## Running

From inside the `GNN-MAPPO/` directory:

```bash
python main.py
```

To disable WandB logging (offline/debug runs), open `args.py` and set:

```python
self.use_wandb = False
```

To run on CPU:

```python
self.device = "cpu"
```

### Quick smoke test (no GPU, no WandB)

```bash
python -c "
import sys; sys.path.insert(0, '.')
from args import Args
from runner import Runner

args = Args()
args.device = 'cpu'
args.n_rollout_threads = 1
args.episode_length = 5
args.ppo_epoch = 1
args.num_env_steps = 15
args.use_wandb = False

runner = Runner(args)
runner.run()
runner.envs.close()
print('OK')
"
```

### Loading a saved checkpoint

```python
self.model_dir = "checkpoints/GNN-MAPPO"
```

Checkpoints are saved every `save_interval` (default 10) episodes under `checkpoints/GNN-MAPPO/`.

---

## Key Hyperparameters (`args.py`)

### Environment

| Parameter | Default | Description |
|---|---|---|
| `num_agents` | 8 | Number of agents |
| `num_goals` | 43 | Number of goal objects |
| `grid_size` | 31 | Grid width and height |
| `episode_length` | 175 | Steps per episode |
| `reward_goal` | 10 | Reward for collecting a goal |
| `penalty_invalid_move` | -5 | Penalty for hitting a wall |
| `max_edge_dist` | 4.5 | Max distance for a graph edge to exist |

### PPO

| Parameter | Default | Description |
|---|---|---|
| `n_rollout_threads` | 8 | Parallel environments |
| `num_env_steps` | 2,000,000 | Total environment steps |
| `ppo_epoch` | 10 | PPO update epochs per rollout |
| `clip_param` | 0.2 | PPO clipping ε |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE λ |
| `lr` / `critic_lr` | 7e-4 | Actor / critic learning rates |
| `entropy_coef` | 0.01 | Entropy bonus coefficient |
| `use_valuenorm` | True | Normalize value targets with EMA |
| `use_huber_loss` | True | Huber loss for critic (delta=10) |

### GNN

| Parameter | Default | Description |
|---|---|---|
| `node_obs_shape` | 9 | Features per entity node |
| `gnn_hidden_size` | 16 | GNN output channels |
| `gnn_num_heads` | 3 | Attention heads (averaged, not concatenated) |
| `gnn_layer_N` | 2 | Number of TransformerConv layers after first |
| `embed_hidden_size` | 16 | EmbedConv output channels |
| `embedding_size` | 12 | Entity-type embedding dimension |
| `actor_graph_aggr` | `"node"` | Ego-node embedding for actor |
| `critic_graph_aggr` | `"global"` | Mean-pool for critic |

### MLP

| Parameter | Default | Description |
|---|---|---|
| `hidden_size` | 64 | MLP hidden / output width |
| `layer_N` | 1 | Extra MLP hidden layers |

---

## Tensor Shape Reference

| Quantity | Shape | Notes |
|---|---|---|
| `obs` per rollout | (176, 8, 8, 8) | T+1, R, A, obs_dim |
| `node_obs` per rollout | (176, 8, 8, 51, 9) | T+1, R, A, entities, node_feat |
| `adj` per rollout | (176, 8, 8, 51, 51) | T+1, R, A, entities, entities |
| Actor input obs | (64, 8) | R×A flattened |
| Actor GNN output | (64, 16) | ego-node embedding |
| Actor MLP input | (64, 24) | cat(obs, gnn) |
| Actor output | (64, 1) + (64, 1) | actions, log_probs |
| Critic input cent_obs | (64, 64) | 8 agents × 8 obs |
| Critic GNN output | (64, 16) | global mean-pool |
| Critic MLP input | (64, 80) | cat(cent_obs, gnn) |
| Critic output | (64, 1) | values |
| PPO train batch | (11,200, ...) | 175 × 8 × 8 |

---

## WandB Metrics

| Metric | Description |
|---|---|
| `Value Loss` | Critic Huber loss |
| `Policy Loss` | Actor PPO clip loss |
| `Entropy` | Action distribution entropy |
| `Actor Grad Norm` | Gradient norm for actor |
| `Critic Grad Norm` | Gradient norm for critic |
| `Goals Collected` | Mean goals collected per episode |
| `Goals %` | Goals collected / total goals |
| `Grid Seen %` | Fraction of grid cells visited |
| `FPS` | Environment steps per second |

---

## Windows Notes

- **`if __name__ == "__main__"`** in `main.py` is mandatory. Python's multiprocessing uses `spawn` mode on Windows, which re-imports the main module in each subprocess — without the guard, the subprocesses would each try to spawn more subprocesses, causing an infinite fork bomb.
- All environment outputs are **numpy arrays** (no torch tensors). CUDA tensors cannot cross subprocess pipe boundaries on Windows.
- If you hit issues with 8 parallel subprocesses, set `n_rollout_threads = 1` in `args.py` to switch to `GraphDummyVecEnv` (single-process, no spawning).

---

## Acknowledgements

- Environment adapted from the companion **GNN** project in this repo.
- PPO algorithm and GNN backbone adapted from [InforMARL](https://github.com/calimero-is-near/InforMARL) (Nayak et al., 2023).
