import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional

from utils import init, get_clones, check
from gnn import GNNBase


class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return super().log_prob(actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class Categorical(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int, use_orthogonal: bool = True, gain: float = 0.01):
        super(Categorical, self).__init__()
        init_method = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x: Tensor, available_actions=None, temperature: float = 1.0):
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = torch.finfo(x.dtype).min
        if temperature != 1.0:
            x = x / temperature
        return FixedCategorical(logits=x)


class ACTLayer(nn.Module):
    def __init__(self, action_dim: int, inputs_dim: int, use_orthogonal: bool, gain: float):
        super(ACTLayer, self).__init__()
        self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)

    def forward(
        self,
        x: Tensor,
        available_actions: Optional[Tensor] = None,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[Tensor, Tensor]:
        action_logits = self.action_out(x, available_actions, temperature)
        actions = action_logits.mode() if deterministic else action_logits.sample()
        action_log_probs = action_logits.log_probs(actions)
        return actions, action_log_probs

    def evaluate_actions(
        self,
        x: Tensor,
        action: Tensor,
        available_actions: Optional[Tensor] = None,
        active_masks: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        action_logits = self.action_out(x, available_actions)
        action_log_probs = action_logits.log_probs(action)
        if active_masks is not None:
            dist_entropy = (action_logits.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum()
        else:
            dist_entropy = action_logits.entropy().mean()
        return action_log_probs, dist_entropy


class MLPLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, layer_N: int, use_orthogonal: bool, use_ReLU: bool):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = nn.ReLU() if use_ReLU else nn.Tanh()
        init_method = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_
        gain = nn.init.calculate_gain("relu" if use_ReLU else "tanh")

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc_h = nn.Sequential(init_(nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc2 = get_clones(self.fc_h, self._layer_N)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x


class MLPBase(nn.Module):
    def __init__(self, args, override_obs_dim: Optional[int] = None):
        super(MLPBase, self).__init__()
        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size

        obs_dim = override_obs_dim
        if obs_dim is None:
            raise ValueError("override_obs_dim must be provided for graph-based models")

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(obs_dim, self.hidden_size, self._layer_N, self._use_orthogonal, self._use_ReLU)

    def forward(self, x: Tensor) -> Tensor:
        if self._use_feature_normalization:
            x = self.feature_norm(x)
        return self.mlp(x)


class RNNLayer(nn.Module):
    def __init__(self, inputs_dim, outputs_dim, recurrent_N, use_orthogonal):
        super(RNNLayer, self).__init__()
        self._recurrent_N = recurrent_N
        self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=recurrent_N)
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                if use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        self.norm = nn.LayerNorm(outputs_dim)

    def forward(self, x: Tensor, hxs: Tensor, masks: Tensor) -> Tuple[Tensor, Tensor]:
        x, hxs = self.rnn(
            x.unsqueeze(0),
            (hxs * masks.repeat(1, self._recurrent_N).unsqueeze(-1)).transpose(0, 1).contiguous(),
        )
        x = x.squeeze(0)
        hxs = hxs.transpose(0, 1)
        x = self.norm(x)
        return x, hxs


def minibatchGenerator(obs, node_obs, adj, agent_id, max_batch_size: int):
    num_minibatches = obs.shape[0] // max_batch_size + 1
    for i in range(num_minibatches):
        yield (
            obs[i * max_batch_size : (i + 1) * max_batch_size],
            node_obs[i * max_batch_size : (i + 1) * max_batch_size],
            adj[i * max_batch_size : (i + 1) * max_batch_size],
            agent_id[i * max_batch_size : (i + 1) * max_batch_size],
        )


class GR_Actor(nn.Module):
    def __init__(
        self,
        args,
        obs_dim: int,
        node_obs_dim: int,
        edge_dim: int,
        action_dim: int,
        device=torch.device("cpu"),
        split_batch: bool = False,
        max_batch_size: int = 32,
    ):
        super(GR_Actor, self).__init__()
        self.hidden_size = args.hidden_size
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.split_batch = split_batch
        self.max_batch_size = max_batch_size
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.gnn_base = GNNBase(args, node_obs_dim, edge_dim, args.actor_graph_aggr)
        gnn_out_dim = self.gnn_base.out_dim
        mlp_in_dim = gnn_out_dim + obs_dim

        self.base = MLPBase(args, override_obs_dim=mlp_in_dim)

        if self._use_recurrent_policy or self._use_naive_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_dim, self.hidden_size, self._use_orthogonal, self._gain)

        self.to(device)

    def forward(
        self,
        obs,
        node_obs,
        adj,
        agent_id,
        rnn_states,
        masks,
        available_actions=None,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        obs = check(obs).to(**self.tpdv)
        node_obs = check(node_obs).to(**self.tpdv)
        adj = check(adj).to(**self.tpdv)
        agent_id = check(agent_id).to(**self.tpdv).long()
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if self.split_batch and obs.shape[0] > self.max_batch_size:
            actor_features_list = []
            for batch in minibatchGenerator(obs, node_obs, adj, agent_id, self.max_batch_size):
                ob, nob, ad, ag = batch
                nbd = self.gnn_base(nob, ad, ag)
                feat = torch.cat([ob, nbd], dim=1)
                actor_features_list.append(self.base(feat))
            actor_features = torch.cat(actor_features_list, dim=0)
        else:
            nbd_features = self.gnn_base(node_obs, adj, agent_id)
            actor_features = torch.cat([obs, nbd_features], dim=1)
            actor_features = self.base(actor_features)

        if self._use_recurrent_policy or self._use_naive_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic, temperature)
        return actions, action_log_probs, rnn_states

    def evaluate_actions(
        self,
        obs,
        node_obs,
        adj,
        agent_id,
        rnn_states,
        action,
        masks,
        available_actions=None,
        active_masks=None,
    ) -> Tuple[Tensor, Tensor]:
        obs = check(obs).to(**self.tpdv)
        node_obs = check(node_obs).to(**self.tpdv)
        adj = check(adj).to(**self.tpdv)
        agent_id = check(agent_id).to(**self.tpdv).long()
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        if self.split_batch and obs.shape[0] > self.max_batch_size:
            actor_features_list = []
            for batch in minibatchGenerator(obs, node_obs, adj, agent_id, self.max_batch_size):
                ob, nob, ad, ag = batch
                nbd = self.gnn_base(nob, ad, ag)
                feat = torch.cat([ob, nbd], dim=1)
                actor_features_list.append(self.base(feat))
            actor_features = torch.cat(actor_features_list, dim=0)
        else:
            nbd_features = self.gnn_base(node_obs, adj, agent_id)
            actor_features = torch.cat([obs, nbd_features], dim=1)
            actor_features = self.base(actor_features)

        if self._use_recurrent_policy or self._use_naive_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None,
        )
        return action_log_probs, dist_entropy


class GR_Critic(nn.Module):
    def __init__(
        self,
        args,
        cent_obs_dim: int,
        node_obs_dim: int,
        edge_dim: int,
        device=torch.device("cpu"),
        split_batch: bool = False,
        max_batch_size: int = 32,
    ):
        super(GR_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.split_batch = split_batch
        self.max_batch_size = max_batch_size
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.use_cent_obs = args.use_cent_obs

        init_method = nn.init.orthogonal_ if self._use_orthogonal else nn.init.xavier_uniform_

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.gnn_base = GNNBase(args, node_obs_dim, edge_dim, args.critic_graph_aggr)
        gnn_out_dim = self.gnn_base.out_dim

        if args.critic_graph_aggr == "node":
            gnn_out_dim *= args.num_agents

        mlp_in_dim = gnn_out_dim
        if self.use_cent_obs:
            mlp_in_dim += cent_obs_dim

        self.base = MLPBase(args, override_obs_dim=mlp_in_dim)

        if self._use_recurrent_policy or self._use_naive_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(
        self,
        cent_obs,
        node_obs,
        adj,
        agent_id,
        rnn_states,
        masks,
    ) -> Tuple[Tensor, Tensor]:
        cent_obs = check(cent_obs).to(**self.tpdv)
        node_obs = check(node_obs).to(**self.tpdv)
        adj = check(adj).to(**self.tpdv)
        agent_id = check(agent_id).to(**self.tpdv).long()
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if self.split_batch and cent_obs.shape[0] > self.max_batch_size:
            critic_features_list = []
            for batch in minibatchGenerator(cent_obs, node_obs, adj, agent_id, self.max_batch_size):
                co, nob, ad, ag = batch
                nbd = self.gnn_base(nob, ad, ag)
                feat = torch.cat([co, nbd], dim=1) if self.use_cent_obs else nbd
                critic_features_list.append(self.base(feat))
            critic_features = torch.cat(critic_features_list, dim=0)
        else:
            nbd_features = self.gnn_base(node_obs, adj, agent_id)
            if self.use_cent_obs:
                critic_features = torch.cat([cent_obs, nbd_features], dim=1)
            else:
                critic_features = nbd_features
            critic_features = self.base(critic_features)

        if self._use_recurrent_policy or self._use_naive_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        values = self.v_out(critic_features)
        return values, rnn_states
