import copy
import torch
import torch.nn as nn
import torch_geometric.loader as loader

from torch import Tensor
from typing import Callable
from torch_geometric.data import Data
from torch_geometric.typing import OptPairTensor
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import MessagePassing, TransformerConv


def init(module: nn.Module, weight_init: Callable, bias_init: Callable, gain: float = 1) -> nn.Module:
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class EmbedConv(MessagePassing):
    def __init__(
        self,
        input_dim: int,
        num_embeddings: int,
        embedding_size: int,
        hidden_size: int,
        layer_n: int,
        use_orthogonal: bool,
        use_relu: bool,
        use_layer_norm: bool,
        add_self_loop: bool,
        edge_dim: int = 0,
    ) -> None:
        super(EmbedConv, self).__init__(aggr="add")
        self._layer_n = layer_n
        self._add_self_loops = add_self_loop
        active_func = [nn.Tanh(), nn.ReLU()][use_relu]
        layer_norm = [nn.Identity(), nn.layer_norm(hidden_size)][use_layer_norm]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(["tanh", "relu"][use_relu])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.entity_embed = nn.Embedding(num_embeddings, embedding_size)
        self.lin1 = nn.Sequential(
            init_(nn.Linear(input_dim + embedding_size + edge_dim + 3, hidden_size)), active_func, layer_norm
        )
        self.lin_h = nn.Sequential(init_(nn.Linear(hidden_size, hidden_size)), active_func, layer_norm)
        self.lin2 = get_clones(self.lin_h, self._layer_n)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None) -> Tensor:
        x: OptPairTensor = (x, x)
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        node_feat_j = x_j[..., :-1]
        entity_type_j = x_j[..., -1].long()
        entity_embed_j = self.entity_embed(entity_type_j)
        node_feat = torch.cat([node_feat_j, entity_embed_j, edge_attr], dim=-1)
        x = self.lin1(node_feat)

        for i in range(self._layer_n):
            x = self.lin2[i](x)
        return x


class TransformerConvNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_embeddings: int,
        embedding_size: int,
        hidden_size: int,
        num_heads: int,
        concat_heads: bool,
        layer_n: int,
        use_relu: bool,
        graph_aggr: str,
        global_aggr_type: str,
        embed_hidden_size: int,
        embed_layer_n: int,
        embed_use_orthogonal: bool,
        embed_use_relu: bool,
        embed_use_layer_norm: bool,
        embed_add_self_loop: bool,
        max_edge_dist: float,
        num_agents: int,
        agent_connections: int,
        edge_dim: int = 1,
    ) -> None:
        super(TransformerConvNet, self).__init__()
        self.active_func = [nn.Tanh(), nn.ReLU()][use_relu]
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        self.edge_dim = edge_dim
        self.max_edge_dist = max_edge_dist
        self.graph_aggr = graph_aggr
        self.global_aggr_type = global_aggr_type
        self.num_agents = num_agents
        self.agent_connections = agent_connections
        self.embed_layer = EmbedConv(
            input_dim=input_dim,
            num_embeddings=num_embeddings,
            embedding_size=embedding_size,
            hidden_size=embed_hidden_size,
            layer_n=embed_layer_n,
            use_orthogonal=embed_use_orthogonal,
            use_relu=embed_use_relu,
            use_layer_norm=embed_use_layer_norm,
            add_self_loop=embed_add_self_loop,
            edge_dim=edge_dim,
        )
        self.gnn1 = TransformerConv(
            in_channels=embed_hidden_size,
            out_channels=hidden_size,
            heads=num_heads,
            concat=concat_heads,
            beta=False,
            dropout=0.0,
            edge_dim=edge_dim,
            bias=True,
            root_weight=True,
        )
        self.gnn2 = nn.ModuleList()

        for _ in range(layer_n):
            self.gnn2.append(self.add_tc_layer(self.get_in_channels(hidden_size), hidden_size))

    def forward(self, node_obs: Tensor, adj: Tensor, agent_id: Tensor) -> Tensor:
        batch_size = node_obs.shape[0]
        data_list = []

        for i in range(batch_size):
            current_agent_id = agent_id[i]
            agent_nodes = node_obs[i]

            # Process adjacency matrix
            edge_index, edge_attr = self.process_adj(adj[i], current_agent_id)

            # Print filtered node features (all nodes that remain after filtering)
            print("\nFILTERED NODE FEATURES:")
            print(f"Agent {current_agent_id.item()} filtered node_obs shape: {agent_nodes.shape}")
            print(f"edge_attr:\n{edge_attr}")

            if len(edge_attr.shape) == 1:
                edge_attr = edge_attr.unsqueeze(1)
            data_list.append(Data(x=agent_nodes, edge_index=edge_index, edge_attr=edge_attr))

        # Create batch
        loader_data = loader.DataLoader(data_list, shuffle=False, batch_size=batch_size)
        data = next(iter(loader_data))
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch

        if self.edge_dim is None:
            edge_attr = None

        # Process through layers
        x = self.embed_layer(x, edge_index, edge_attr)
        x = self.active_func(self.gnn1(x, edge_index, edge_attr))

        for layer in self.gnn2:
            x = self.active_func(layer(x, edge_index, edge_attr))

        # Convert back to dense batch
        x, _ = to_dense_batch(x, batch)

        # Gather node features
        x = self.gather_node_feats(x, agent_id)

        return x

    def add_tc_layer(self, in_channels: int, out_channels: int) -> TransformerConv:
        return TransformerConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=self.num_heads,
            concat=self.concat_heads,
            beta=False,
            dropout=0.0,
            edge_dim=self.edge_dim,
            root_weight=True,
        )

    def get_in_channels(self, out_channels: int) -> int:
        return out_channels + (self.num_heads - 1) * self.concat_heads * out_channels

    def process_adj(self, adj: Tensor, agent_id: Tensor) -> tuple[Tensor, Tensor]:
        assert adj.dim() == 2  # Should be a 2D matrix
        assert adj.size(0) == adj.size(1)  # Should be square

        modified_adj = adj.clone()

        # Get the agent-to-agent distance submatrix
        agent_distances = modified_adj[: self.num_agents, : self.num_agents]

        # Get distances from ego agent to other agents
        other_agent_distances = agent_distances[agent_id].clone()
        other_agent_distances[agent_id] = float("inf")  # Make sure we don't select self

        # Handle cases with any number of agents
        num_other_agents = self.num_agents - 1
        if num_other_agents > 0:
            # Find up to (self.agent_connections) closest agents, but no more than available
            k = min(self.agent_connections, num_other_agents)
            closest_agents = torch.topk(other_agent_distances, k=k, largest=False).indices

            # Zero out ALL agent-to-agent connections first
            modified_adj[: self.num_agents, : self.num_agents] = 0

            # Restore original distances for closest agents
            for close_id in closest_agents:
                # Keep original distances instead of setting to 1
                modified_adj[agent_id, close_id] = adj[agent_id, close_id]
                modified_adj[close_id, agent_id] = adj[close_id, agent_id]
        else:
            # If there's only one agent, zero out agent-to-agent connections
            modified_adj[: self.num_agents, : self.num_agents] = 0

        # Zero out all non-agent to non-agent connections
        modified_adj[self.num_agents :, self.num_agents :] = 0

        # Handle agent-to-goal connections with distance threshold
        agent_to_goals = modified_adj[: self.num_agents, self.num_agents :]
        goals_to_agent = modified_adj[self.num_agents :, : self.num_agents]

        # Zero out connections beyond max_edge_dist
        agent_to_goals[agent_to_goals > self.max_edge_dist] = 0
        goals_to_agent[goals_to_agent > self.max_edge_dist] = 0

        # Convert to edge index format for PyG
        index = modified_adj.nonzero(as_tuple=True)
        edge_attr = modified_adj[index]  # Get the actual distance values

        return torch.stack(index, dim=0), edge_attr

    @staticmethod
    def gather_node_feats(x: Tensor, idx: Tensor) -> Tensor:
        if x.shape[0] == 1:
            return x[0, idx, :]

        batch_size, _, feature_size = x.shape
        idx_expanded = idx.view(batch_size, 1, 1).expand(-1, 1, feature_size)
        return torch.gather(x, dim=1, index=idx_expanded).squeeze(1)


class GNNBase(nn.Module):
    def __init__(self, args) -> None:
        super(GNNBase, self).__init__()
        self.args = args
        self.input_dim = args.node_obs_shape
        self.num_actions = args.action_size
        self.hidden_dim = args.gnn_hidden_size
        self.heads = args.gnn_num_heads
        self.concat = args.gnn_concat_heads

        self.gnn = TransformerConvNet(
            input_dim=args.node_obs_shape,
            edge_dim=args.edge_dim,
            num_embeddings=args.num_embeddings,
            embedding_size=args.embedding_size,
            hidden_size=args.gnn_hidden_size,
            num_heads=args.gnn_num_heads,
            concat_heads=args.gnn_concat_heads,
            layer_n=args.gnn_layer_n,
            use_relu=args.gnn_use_relu,
            graph_aggr=args.graph_aggr,
            global_aggr_type=args.global_aggr_type,
            embed_hidden_size=args.embed_hidden_size,
            embed_layer_n=args.embed_layer_n,
            embed_use_orthogonal=args.use_orthogonal,
            embed_use_relu=args.embed_use_relu,
            embed_use_layer_norm=args.use_feat_norm,
            embed_add_self_loop=args.embed_add_self_loop,
            max_edge_dist=args.max_edge_dist,
            num_agents=args.num_agents,
            agent_connections=args.agent_connections,
        )

        self.fc1 = nn.Linear(self.out_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim, self.num_actions)

    def forward(self, obs: Tensor, node_obs: Tensor, adj: Tensor, agent_id: Tensor) -> Tensor:
        x = self.gnn(node_obs, adj, agent_id)
        if obs.shape[0] > 1:
            x = torch.cat((obs, x), dim=-1)
        else:
            x = torch.cat((obs, x.unsqueeze(0)), dim=-1)
        x = self.fc1(x)
        x = self.relu(x)
        q_values = self.fc2(x)
        return q_values

    @property
    def out_dim(self) -> int:
        return self.args.gnn_hidden_size + self.args.node_obs_shape

    def count_layers_and_params(self) -> tuple[dict, int]:
        def count_layers(module, module_name=""):
            if isinstance(module, nn.Linear):
                return {"linear": 1}, module_name
            elif isinstance(module, nn.Embedding):
                return {"embedding": 1}, module_name
            elif isinstance(module, TransformerConv):
                return {"transformer": 1}, module_name
            elif isinstance(module, EmbedConv):
                embed_layers = {"embedding": 1, "linear": 0}
                for child in module.children():
                    if isinstance(child, nn.Linear):
                        embed_layers["linear"] += 1
                    elif isinstance(child, nn.Sequential):
                        for subchild in child:
                            if isinstance(subchild, nn.Linear):
                                embed_layers["linear"] += 1
                return embed_layers, module_name
            else:
                layer_count = {}
                layer_names = []
                for name, child in module.named_children():
                    child_count, child_names = count_layers(child, f"{module_name}.{name}" if module_name else name)
                    for key, value in child_count.items():
                        layer_count[key] = layer_count.get(key, 0) + value
                    layer_names.extend(child_names if isinstance(child_names, list) else [child_names])
                return layer_count, layer_names

        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        total_layers, layer_names = count_layers(self)
        total_params = count_params(self)

        print("Detailed Layer and Parameter Count:")
        print("===================================")

        # Count layers and params for GNN
        gnn_layers, gnn_layer_names = count_layers(self.gnn, "gnn")
        gnn_params = count_params(self.gnn)
        print("GNN Layers:")
        for layer_type, count in gnn_layers.items():
            print(f"  - {layer_type.capitalize()}: {count}")
        print(f"GNN Parameters: {gnn_params}")
        print("-----------------------------------")

        # Count layers and params for fully connected layers
        fc_layers = 2
        fc_params = count_params(self.fc1) + count_params(self.fc2)
        print(f"Fully Connected Layers: {fc_layers}")
        print("  - fc1")
        print("  - fc2")
        print(f"Fully Connected Parameters: {fc_params}")
        print("-----------------------------------")

        # Total counts
        print("Total Layers:")
        for layer_type, count in total_layers.items():
            print(f"  - {layer_type.capitalize()}: {count}")
        print(f"Total Parameters: {total_params}")

        return total_layers, total_params

    def get_gnn_structure(self) -> str:
        def get_structure(module, indent=0):
            if isinstance(module, (nn.Linear, nn.Conv2d, TransformerConv, EmbedConv, nn.Embedding)):
                return f"{'  ' * indent}{module.__class__.__name__}: {module}\n"
            else:
                structure = f"{'  ' * indent}{module.__class__.__name__}:\n"
                for name, child in module.named_children():
                    structure += get_structure(child, indent + 1)
                return structure

        gnn_structure = get_structure(self.gnn)

        full_structure = "GNNBase Structure:\n"
        full_structure += "==================\n"
        full_structure += "GNN:\n"
        full_structure += gnn_structure
        full_structure += "Fully Connected Layers:\n"
        full_structure += f"  fc1: {self.fc1}\n"
        full_structure += f"  fc2: {self.fc2}\n"

        return full_structure
