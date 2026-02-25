import os
import time


class Args:
    def __init__(self) -> None:

        self.env = None
        self.num_agents = 15
        self.num_goals = 76
        self.num_obstacles = 0
        self.grid_size = 41
        self.action_size = 4
        self.agent_connections = 3
        self.agent_view_size = 9

        self.n_rollout_threads = 4
        self.episode_length = 200
        self.num_env_steps = 500_000
        self.ppo_epoch = 10
        self.num_mini_batch = 4
        self.clip_param = 0.2
        self.value_loss_coef = 1.0
        self.entropy_coef = 0.01
        self.max_grad_norm = 10.0
        self.huber_delta = 10.0
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.lr = 7e-4
        self.critic_lr = 7e-4
        self.opti_eps = 1e-5
        self.weight_decay = 0.0
        self.use_linear_lr_decay = True
        self.data_chunk_length = 10

        self.use_gae = True
        self.use_valuenorm = True
        self.use_popart = False
        self.use_clipped_value_loss = True
        self.use_huber_loss = True
        self.use_max_grad_norm = True
        self.use_recurrent_policy = False
        self.use_naive_recurrent_policy = False
        self.recurrent_N = 1
        self.use_value_active_masks = True
        self.use_policy_active_masks = True
        self.use_centralized_V = True
        self.use_cent_obs = True
        self.use_proper_time_limits = False
        self.use_feature_normalization = True
        self.use_ReLU = True
        self.layer_N = 1
        self.gain = 0.01
        self.stacked_frames = 1
        self.hidden_size = 64

        self.gnn_hidden_size = 16
        self.gnn_num_heads = 3
        self.gnn_concat_heads = False
        self.node_obs_shape = 9
        self.edge_dim = 1
        self.num_embeddings = 2
        self.embedding_size = 12
        self.gnn_layer_N = 2
        self.gnn_use_ReLU = True
        self.actor_graph_aggr = "node"
        self.critic_graph_aggr = "global"
        self.global_aggr_type = "mean"
        self.embed_hidden_size = 16
        self.embed_layer_N = 1
        self.use_orthogonal = True
        self.embed_use_ReLU = True
        self.embed_add_self_loop = False
        self.max_edge_dist = 4.5
        self.split_batch = True
        self.max_batch_size = 128
        self.full_features = False

        self.seed = int(time.time())
        self.device = "cuda"
        self.save_interval = 10
        self.log_interval = 1
        self.use_wandb = False
        self.use_render = True
        self.model_dir = None

        self.reward_goal = 10
        self.penalty_goal = 0
        self.penalty_obstacle = 0
        self.penalty_invalid_move = -5

        self.title = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
