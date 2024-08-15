from torch import nn
import torch
from torch import Tensor
from torch_geometric import nn as gnn
from torch_geometric.utils import to_dense_batch
import rff
from torch_geometric.typing import Adj, OptTensor
import numpy as np


class AdaptivePointNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = gnn.InstanceNorm(in_channel)
        self.affine = nn.Linear(style_dim, in_channel * 2)

        self.affine.weight.data.normal_()
        self.affine.bias.data.zero_()

        self.affine.bias.data[:in_channel] = 1
        self.affine.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.affine(style)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = (gamma * out).add_(beta)

        return out
    

class StyleLinearLayer(nn.Module):
    def __init__(self, in_dim, w_dim, out_dim, noise=False):
        super().__init__()
        self.in_dim = in_dim
        self.w_dim = w_dim
        self.out_dim = out_dim
        self.noise = noise
        self.activation = nn.LeakyReLU(inplace=True)

        self.linear = nn.Linear(in_dim, out_dim)
        self.adain = AdaptivePointNorm(out_dim, w_dim)
        self.noise_strength = nn.Parameter(torch.zeros(1)) if noise else None

    def forward(self, x, w):
        x = self.linear(x)

        if self.noise:
            noise = torch.randn(1, x.size(1), device=x.device)
            noise = noise * self.noise_strength
            x.add_(noise)

        x = self.activation(x)
        x = self.adain(x, w)

        return x


class GNNConv(nn.Module):
    def __init__(self, channels, aggr="max"):
        super().__init__()

        self.mlp_h = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channels, 3),
            nn.Tanh(),
        )

        self.mlp_f = nn.Sequential(
            nn.Linear(channels + 3, channels),
            nn.LeakyReLU(inplace=True),
        )

        self.mlp_g = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channels, channels),
        )

        self.network = gnn.PointGNNConv(self.mlp_h, self.mlp_f, self.mlp_g, aggr=aggr)

    def forward(self, x, pos, edge_index):
        return self.network(x, pos, edge_index)
    

class SyntheticBlock(nn.Module):
    def __init__(self, in_channels, out_channels, styles):
        super().__init__()
        self.add_noise = True

        self.gnn_conv = GNNConv(in_channels)
        self.adaptive_norm = AdaptivePointNorm(in_channels, styles)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

        if self.add_noise:
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))

    def forward(self, h, pos, edge_index, style):
        h = self.gnn_conv(h, pos, edge_index)

        if self.add_noise:
            noise = torch.randn_like(h) * self.noise_strength
            h = h + noise

        h = self.leaky_relu(h)
        h = self.adaptive_norm(h, style)
        return h


class Generator(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Generator, self).__init__()
        self.z_dim = 128
        channels = 128

        self.encoder = rff.layers.GaussianEncoding(
            sigma=10.0, input_size=3, encoded_size=channels // 2
        )

        self.style = nn.Sequential(
            nn.Linear(self.z_dim + 3, self.z_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.z_dim, self.z_dim),
            nn.LeakyReLU(inplace=True),
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

        self.conv1 = SyntheticBlock(128, 128, self.z_dim)
        self.conv2 = SyntheticBlock(128, 128, self.z_dim)

    def forward(self, pos, edge_index, batch, styles):
        results = []
        for style in styles:
            # broadcast style to all points
            style = style.repeat(pos.size(0), 1)
            style = torch.cat([style, pos], dim=1)
            style = self.style(style)

            x = self.encoder(pos)
            x = self.conv1(x, pos, edge_index, style)
            x = self.conv2(x, pos, edge_index, style)
            # x, _ = self.conv3(x, pos, edge_index, style)

            h = gnn.global_max_pool(x, batch)
            h = self.global_conv(h)
            h = h.repeat(x.size(0), 1)

            x = torch.cat([x, h], dim=1)

            x = self.tail(x)
            results.append(x)
        return torch.stack(results, dim=0).permute(0, 2, 1)
