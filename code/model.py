import math
import torch
from torch import nn
from torch_geometric.nn import MeanSubtractionNorm
import torch.nn.functional as F
from torch.nn.modules.module import Module
torch.backends.cudnn.enabled = False
from utilss import *
import typing
from typing import Optional, Tuple, Union
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (Adj, NoneType, OptTensor, PairTensor, SparseTensor)
from torch_geometric.utils import softmax
if typing.TYPE_CHECKING:
    from typing import overload
else:
    from torch.jit import _overload_method as overload



class GKformer(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 heads,
                 dropout_trans,
                 num_gcn_layers,
                 num_transformer_layers,
                 dropout,
                 num_r
    ):
        super(GKformer, self).__init__()
        self.num_gcn_layers = num_gcn_layers
        self.num_transformer_layers = num_transformer_layers
        self.num_r = num_r

        self.GCNConv = nn.ModuleList(
            [GCNLayer(in_channels if i == 0 else hidden_channels, hidden_channels) for i in range(num_gcn_layers)])
        # self.TransformerConv = nn.ModuleList(
        #     # [TransformerConv(hidden_channels, hidden_channels, heads, dropout_trans) for j in range(num_transformer_layers)])
        #     [TransformerConv(hidden_channels, hidden_channels, heads) for j in range(num_transformer_layers)])
        # self.TransformerConv = TransformerConv(in_channels, hidden_channels, heads)
        self.TransformerConv = TransformerConv(hidden_channels, hidden_channels, heads)
        self.KanConv = FastKANLayer(hidden_channels, hidden_channels)

        self.dropout = dropout
        #self.norm = BatchNorm(hidden_channels)
        self.norm = MeanSubtractionNorm()
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.decoder = InnerProductDecoder(out_channels, num_r)

    def forward(self, x, adj, index, attr):
        #x_res = self.lin(x)
        x = F.dropout(x, self.dropout, training=self.training)
        for GConv in self.GCNConv:
            x = GConv(x, index, attr)

        # for TransformerConv in self.TransformerConv:
        #     x = TransformerConv(x, index)
        x = self.TransformerConv(x, index)

        x = self.KanConv(x)

        # x = F.relu(x)
        x = self.norm(x)
        #x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin(x)

        x = F.dropout(x, self.dropout)
        x = self.norm(x)
        x = self.decoder(x)
        output = F.sigmoid(x)

        return output



class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / (self.weight.size(1) ** 0.5)
        self.weight.data.uniform_(-stdv, stdv)
    def forward(self, x, edge_index, edge_weight):
        num_nodes = x.size(0)
        adj = torch.zeros(num_nodes, num_nodes, device=x.device)
        src, dst = edge_index
        adj[src, dst] = edge_weight
        adj = adj + torch.eye(num_nodes, device=x.device)
        deg = torch.sum(adj, dim=0)
        deg_inv_sqrt = deg.pow(-0.5)
        adj = deg_inv_sqrt[None, :] * adj * deg_inv_sqrt[:, None]

        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        return output

class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)
    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)
class RadialBasisFunction(nn.Module):
    def __init__(
            self,
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 8,
            denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)
    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)
class FastKANLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 8,
            use_base_update: bool = True,
            base_activation=F.silu,
            spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, time_benchmark=False):
        if not time_benchmark:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret


class InnerProductDecoder(Module):
    def __init__(self, input_dim, num_r):
        super(InnerProductDecoder, self).__init__()
        self.weight = nn.Parameter(torch.empty(size=(input_dim, input_dim)))  # 建立一个w权重，用于对特征数进行线性变化
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)  # 对权重矩阵进行初始化
        self.num_r = num_r
    def forward(self, inputs):
        M = inputs[0:self.num_r, :]
        D = inputs[self.num_r:, :]
        M = torch.mm(M, self.weight)
        D = torch.t(D)  # 转置
        x = torch.mm(M, D)
        # x = torch.reshape(x, [-1])  # 转化为行向量
        return x



class TransformerConv(MessagePassing):

    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int,
        concat: bool = True,
        # concat: bool = False,
        beta: bool = False,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    @overload
    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        return_attention_weights: NoneType = None,
    ) -> Tensor:
        pass

    @overload
    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Tensor,
        edge_attr: OptTensor = None,
        return_attention_weights: bool = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        pass

    @overload
    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        edge_index: SparseTensor,
        edge_attr: OptTensor = None,
        return_attention_weights: bool = None,
    ) -> Tuple[Tensor, SparseTensor]:
        pass

    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        return_attention_weights: Optional[bool] = None,
    ) -> Union[
            Tensor,
            Tuple[Tensor, Tuple[Tensor, Tensor]],
            Tuple[Tensor, SparseTensor],
    ]:

        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor,
        #                  edge_attr: OptTensor)
        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
            key_j = key_j + edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            out = out + edge_attr

        out = out * alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


# class TransformerConv(MessagePassing):
#     _alpha: OptTensor
#
#     def __init__(
#         self,
#         in_channels: Union[int, Tuple[int, int]],
#         out_channels: int,
#         heads: int,
#         dropout_trans: float,
#         beta: bool = False,
#         bias: bool = True,
#         root_weight: bool = True,
#         **kwargs,
#     ):
#         kwargs.setdefault('aggr', 'add')
#         super().__init__(node_dim=0, **kwargs)
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.heads = heads
#         self.beta = beta and root_weight
#         self.root_weight = root_weight
#         self.dropout = dropout_trans
#
#         in_channels = (in_channels, in_channels)
#         self.lin_key = Linear(in_channels[0], heads * out_channels)
#         self.lin_query = Linear(in_channels[1], heads * out_channels)
#         self.lin_value = Linear(in_channels[0], heads * out_channels)
#
#         self.lin_skip = Linear(in_channels[1], heads * out_channels, bias=bias)
#         self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
#
#
#     def forward(  # noqa: F811
#         self,
#         x: Union[Tensor, PairTensor],
#         edge_index: Adj,
#     ):
#
#         H, C = self.heads, self.out_channels
#         if isinstance(x, Tensor):
#             x = (x, x)
#         query = self.lin_query(x[1]).view(-1, H, C)
#         key = self.lin_key(x[0]).view(-1, H, C)
#         value = self.lin_value(x[0]).view(-1, H, C)
#         out = self.propagate(edge_index, query=query, key=key, value=value)
#
#         out = out.view(-1, self.heads * self.out_channels)
#         x_r = self.lin_skip(x[1])
#         beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
#         beta = beta.sigmoid()
#         out = beta * x_r + (1 - beta) * out
#
#         return out
#
#     def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor, index: Tensor, ptr: OptTensor, size_i: Optional[int]):
#
#         alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
#         alpha = softmax(alpha, index, ptr, size_i)
#         self._alpha = alpha
#         alpha = F.dropout(alpha, p=self.dropout, training=self.training)
#
#         out = value_j
#         out = out * alpha.view(-1, self.heads, 1)
#         return out
#
#     def __repr__(self) -> str:
#         return (f'{self.__class__.__name__}({self.in_channels}, '
#                 f'{self.out_channels}, heads={self.heads})')
