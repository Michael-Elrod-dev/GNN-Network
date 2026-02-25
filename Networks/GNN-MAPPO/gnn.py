import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, TransformerConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import add_self_loops
from typing import List, Tuple, Union, Optional
from torch_geometric.typing import OptPairTensor, OptTensor

from utils import init, get_clones


class EmbedConv(MessagePassing):
    def __init__(
        self,
        input_dim: int,
        num_embeddings: int,
        embedding_size: int,
        hidden_size: int,
        layer_N: int,
        use_orthogonal: bool,
        use_ReLU: bool,
        use_layerNorm: bool,
        add_self_loop: bool,
        edge_dim: int = 0,
    ):
        super(EmbedConv, self).__init__(aggr="add")
        self._layer_N = layer_N
        self._add_self_loops = add_self_loop
        self.active_func = nn.ReLU() if use_ReLU else nn.Tanh()
        self.layer_norm = nn.LayerNorm(hidden_size) if use_layerNorm else nn.Identity()
        self.init_method = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_

        self.entity_embed = nn.Embedding(num_embeddings, embedding_size)
        self.lin1 = nn.Linear(input_dim + embedding_size + edge_dim, hidden_size)

        self.layers = nn.ModuleList()
        for _ in range(layer_N):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(self.active_func)
            self.layers.append(self.layer_norm)

        self._initialize_weights()

    def _initialize_weights(self):
        gain = nn.init.calculate_gain("relu" if isinstance(self.active_func, nn.ReLU) else "tanh")
        self.init_method(self.lin1.weight, gain=gain)
        nn.init.constant_(self.lin1.bias, 0)
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                self.init_method(layer.weight, gain=gain)
                nn.init.constant_(layer.bias, 0)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ):
        if self._add_self_loops and edge_attr is None:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: Tensor, edge_attr: Optional[Tensor]):
        node_feat_j = x_j[:, :-1]
        entity_type_j = x_j[:, -1].long()
        entity_embed_j = self.entity_embed(entity_type_j)

        if edge_attr is not None:
            node_feat = torch.cat([node_feat_j, entity_embed_j, edge_attr], dim=1)
        else:
            node_feat = torch.cat([node_feat_j, entity_embed_j], dim=1)

        x = self.lin1(node_feat)
        x = self.active_func(x)
        x = self.layer_norm(x)

        for layer in self.layers:
            x = layer(x)

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
        layer_N: int,
        use_ReLU: bool,
        graph_aggr: str,
        global_aggr_type: str,
        embed_hidden_size: int,
        embed_layer_N: int,
        embed_use_orthogonal: bool,
        embed_use_ReLU: bool,
        embed_use_layerNorm: bool,
        embed_add_self_loop: bool,
        max_edge_dist: float,
        edge_dim: int = 1,
    ):
        super(TransformerConvNet, self).__init__()
        self.activation = nn.ReLU() if use_ReLU else nn.Tanh()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        self.edge_dim = edge_dim
        self.max_edge_dist = max_edge_dist
        self.graph_aggr = graph_aggr
        self.global_aggr_type = global_aggr_type

        self.embed_layer = EmbedConv(
            input_dim=input_dim - 1,
            num_embeddings=num_embeddings,
            embedding_size=embedding_size,
            hidden_size=embed_hidden_size,
            layer_N=embed_layer_N,
            use_orthogonal=embed_use_orthogonal,
            use_ReLU=embed_use_ReLU,
            use_layerNorm=embed_use_layerNorm,
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
        for _ in range(layer_N):
            in_channels = hidden_size * num_heads if concat_heads else hidden_size
            self.gnn2.append(
                TransformerConv(
                    in_channels=in_channels,
                    out_channels=hidden_size,
                    heads=num_heads,
                    concat=concat_heads,
                    beta=False,
                    dropout=0.0,
                    edge_dim=edge_dim,
                    root_weight=True,
                )
            )

    def forward(self, batch):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        x = self.embed_layer(x, edge_index, edge_attr)
        x = self.activation(self.gnn1(x, edge_index, edge_attr))
        for gnn in self.gnn2:
            x = self.activation(gnn(x, edge_index, edge_attr))

        if self.graph_aggr == "node":
            return x
        elif self.graph_aggr == "global":
            if self.global_aggr_type == "mean":
                return global_mean_pool(x, batch.batch)
            elif self.global_aggr_type == "max":
                return global_max_pool(x, batch.batch)
            elif self.global_aggr_type == "add":
                return global_add_pool(x, batch.batch)
        raise ValueError(f"Invalid graph_aggr: {self.graph_aggr}")

    @staticmethod
    def process_adj(adj: Tensor, max_edge_dist: float) -> Tuple[Tensor, Tensor]:
        assert adj.dim() in (2, 3)
        assert adj.size(-1) == adj.size(-2)

        connect_mask = ((adj < max_edge_dist) & (adj > 0)).float()
        adj = adj * connect_mask

        if adj.dim() == 3:
            batch_size, num_nodes, _ = adj.shape
            edge_index = adj.nonzero(as_tuple=False)
            edge_attr = adj[edge_index[:, 0], edge_index[:, 1], edge_index[:, 2]]
            batch = edge_index[:, 0] * num_nodes
            edge_index = torch.stack([batch + edge_index[:, 1], batch + edge_index[:, 2]], dim=0)
        else:
            edge_index = adj.nonzero(as_tuple=False).t().contiguous()
            edge_attr = adj[edge_index[0], edge_index[1]]

        edge_attr = edge_attr.unsqueeze(1) if edge_attr.dim() == 1 else edge_attr
        return edge_index, edge_attr


class GNNBase(nn.Module):
    def __init__(
        self,
        args,
        node_obs_shape: int,
        edge_dim: int,
        graph_aggr: str,
    ):
        super(GNNBase, self).__init__()
        self.args = args

        self.gnn = TransformerConvNet(
            input_dim=node_obs_shape,
            edge_dim=edge_dim,
            num_embeddings=args.num_embeddings,
            embedding_size=args.embedding_size,
            hidden_size=args.gnn_hidden_size,
            num_heads=args.gnn_num_heads,
            concat_heads=args.gnn_concat_heads,
            layer_N=args.gnn_layer_N,
            use_ReLU=args.gnn_use_ReLU,
            graph_aggr=graph_aggr,
            global_aggr_type=args.global_aggr_type,
            embed_hidden_size=args.embed_hidden_size,
            embed_layer_N=args.embed_layer_N,
            embed_use_orthogonal=args.use_orthogonal,
            embed_use_ReLU=args.embed_use_ReLU,
            embed_use_layerNorm=args.use_feature_normalization,
            embed_add_self_loop=args.embed_add_self_loop,
            max_edge_dist=args.max_edge_dist,
        )

        self.out_dim = args.gnn_hidden_size * (args.gnn_num_heads if args.gnn_concat_heads else 1)

    def forward(self, node_obs: Tensor, adj: Tensor, agent_id: Tensor) -> Tensor:
        batch_size, num_nodes, _ = node_obs.shape
        edge_index, edge_attr = TransformerConvNet.process_adj(adj, self.gnn.max_edge_dist)

        x = node_obs.view(-1, node_obs.size(-1))
        batch = torch.arange(batch_size, device=node_obs.device).repeat_interleave(num_nodes)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

        x = self.gnn(data)

        if self.gnn.graph_aggr == "node":
            x = x.view(batch_size, num_nodes, -1)
            agent_id = agent_id.long()
            x = x.gather(1, agent_id.unsqueeze(-1).expand(-1, -1, x.size(-1))).squeeze(1)

        return x
