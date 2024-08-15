from torch import nn
import torch
from torch import Tensor
from torch_geometric import nn as gnn
from torch_geometric.utils import to_dense_batch
import rff
from torch_geometric.typing import Adj, OptTensor
import numpy as np


def fmm_modulate_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    styles: torch.Tensor,
    activation: str = "demod",
) -> torch.Tensor:
    points_num, c_in = x.shape
    c_out, c_in = weight.shape
    rank = styles.shape[0] // (c_in + c_out)

    assert styles.shape[0] % (c_in + c_out) == 0
    assert len(styles.shape) == 1

    # Now, we need to construct a [c_out, c_in] matrix
    left_matrix = styles[: c_out * rank]  # [left_matrix_size]
    right_matrix = styles[c_out * rank :]  # [right_matrix_size]

    left_matrix = left_matrix.view(c_out, rank)  # [c_out, rank]
    right_matrix = right_matrix.view(rank, c_in)  # [c_out, rank]

    # Imagine, that the output of `self.affine` (in SynthesisLayer) is N(0, 1)
    # Then, std of weights is sqrt(rank). Converting it back to N(0, 1)
    modulation = left_matrix @ right_matrix / np.sqrt(rank)  # [c_out, c_in]

    if activation == "tanh":
        modulation = modulation.tanh()
    elif activation == "sigmoid":
        modulation = modulation.sigmoid() - 0.5

    W = weight * (modulation + 1.0)  # [c_out, c_in]
    if activation == "demod":
        W = W / (W.norm(dim=1, keepdim=True) + 1e-8)  # [c_out, c_in]
    W = W.to(dtype=x.dtype)

    x = x.view(points_num, c_in, 1)
    out = torch.matmul(W, x)  # [num_rays, c_out, 1]
    out = out.view(points_num, c_out)  # [num_rays, c_out]

    return out


class SynthesisLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        w_dim,
        channels_last=False,
        activation=nn.LeakyReLU(inplace=True),
        rank=10,
    ):
        super().__init__()

        self.w_dim = w_dim
        self.affine = nn.Linear(self.w_dim, (in_channels + out_channels) * rank)

        memory_format = (
            torch.channels_last if channels_last else torch.contiguous_format
        )
        self.weight = torch.nn.Parameter(
            torch.randn([out_channels, in_channels]).to(memory_format=memory_format)
        )
        self.bias = torch.nn.Parameter(torch.zeros([1, out_channels]))
        self.activation = activation

    def forward(self, x, w):
        styles = self.affine(w).squeeze(0)

        x = fmm_modulate_linear(
            x=x, weight=self.weight, styles=styles, activation="demod"
        )

        x = self.activation(x.add_(self.bias))
        return x
    


class PointGNNConv(gnn.MessagePassing):
    r"""The PointGNN operator from the `"Point-GNN: Graph Neural Network for
    3D Object Detection in a Point Cloud" <https://arxiv.org/abs/2003.01251>`_
    paper.
    """

    def __init__(
        self,
        channels,
        z_dim,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "max")
        super().__init__(**kwargs)

        self.mlp_f = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channels // 2, channels),
            nn.LeakyReLU(inplace=True),
        )

        self.mlp_h = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channels // 2, 3),
            nn.Tanh(),
        )

        self.mlp_g = nn.ModuleList(
            [
                SynthesisLayer(channels + 3, channels, z_dim),
                SynthesisLayer(channels, channels, z_dim),
                SynthesisLayer(channels, channels, z_dim),
            ]
        )

    def forward(self, x: Tensor, pos: Tensor, edge_index: Adj, w: Tensor) -> Tensor:
        out = self.propagate(edge_index, x=x, pos=pos)
        for i, layer in enumerate(self.mlp_g):
            out = self.mlp_g(out, w)
        return x + out

    def message(self, pos_j: Tensor, pos_i: Tensor, x_i: Tensor,
                x_j: Tensor) -> Tensor:
        delta = self.mlp_h(x_i)
        e = torch.cat([pos_j - pos_i + delta, x_j], dim=-1)
        return self.mlp_f(e)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  mlp_h={self.mlp_h},\n"
            f"  mlp_g={self.mlp_g},\n"
            f")"
        )


class Generator(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Generator, self).__init__()
        self.z_dim = 128
        channels=128

        self.encoder = rff.layers.GaussianEncoding(
            sigma=10.0, input_size=3, encoded_size=channels // 2
        )

        self.global_conv = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channels, 512),
            nn.LeakyReLU(inplace=True),
        )

        self.tail = nn.Sequential(
            nn.Linear(channels + 512, channels),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channels, channels // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channels // 2, 3),
            nn.Tanh(),
        )

        self.conv1 = PointGNNConv(128, self.z_dim)
        self.conv2 = PointGNNConv(128, self.z_dim)

    def forward(self, pos, edge_index, batch, styles):
        results = []
        for style in styles:
            x = self.encoder(pos)
            x, _ = self.conv1(x, pos, edge_index, style)
            x, _ = self.conv2(x, pos, edge_index, style)
            # x, _ = self.conv3(x, pos, edge_index, style)
            
            h = gnn.global_max_pool(x, batch)
            h = self.global_conv(h)
            h = h.repeat(x.size(0), 1)

            x = torch.cat([x, h], dim=1)

            x = self.tail(x)
            results.append(x)
        return torch.stack(results, dim=0).permute(0, 2, 1)
