import os
import time


class Args:
    def __init__(self) -> None:
        # Env Parameters
        self.env = None
        self.num_agents = 8
        self.num_goals = 43
        self.num_obstacles = 0
        self.grid_size = 31
        self.action_size = 4
        self.agent_connections = 4
        self.agent_view_size = 9

        # DQN Parameters
        self.prio_e = 0.1
        self.prio_a = 0.5
        self.prio_b = 0.4
        self.double_dqn = True
        self.priority_replay = True

        # Training parameters
        self.total_steps = 500000
        self.episode_steps = 175
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_percentage = 0.90
        self.seed = int(time.time())

        # Network Parameters
        self.buffer_size = 100000  # Replay buffer size
        self.batch_size = 64  # Sample batch size
        self.gamma = 0.99  # Discount factor
        self.tau = 0.001  # Soft update of target parameters
        self.lr = 0.0005  # Learning rate
        self.update_step = 4  # Update the network

        # GNN Parameters
        self.gnn_hidden_size = 16
        self.gnn_num_heads = 3
        self.gnn_concat_heads = False
        self.node_obs_shape = 8
        self.edge_dim = 1
        self.num_embeddings = 2
        self.embedding_size = 12
        self.gnn_layer_N = 2
        self.gnn_use_ReLU = True
        self.graph_aggr = "node"
        self.global_aggr_type = "mean"
        self.embed_hidden_size = 16
        self.embed_layer_N = 1
        self.use_orthogonal = True
        self.embed_use_ReLU = True
        self.use_feat_norm = True
        self.embed_add_self_loop = False
        self.full_features = True
        self.max_edge_dist = 4.5

        # Run Parameters
        self.device = "cuda"
        self.load_policy = False  # Evaluate a learned policy
        self.logger = True  # Log training data to Wandb
        self.render = False  # Render the environment
        self.debug = False  # Allow manual action inputs

        # Reward Parameters
        self.reward_goal = 10
        self.penalty_goal = 0
        self.penalty_obstacle = 0
        self.penalty_invalid_move = -5

        # Derived Parameters
        self.title = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
        self.eps_decay = self.calc_eps_decay(self.eps_start, self.eps_end, self.total_steps, self.eps_percentage)

    # Calculate the rate that ε should decay
    def calc_eps_decay(self, eps_start: float, eps_end: float, n_steps: int, eps_percentage: float) -> float:
        effective_steps = n_steps * eps_percentage
        decrement_per_step = (eps_start - eps_end) / effective_steps
        return decrement_per_step
