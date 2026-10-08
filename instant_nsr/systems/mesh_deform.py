"""Mesh deformation and loss helpers used by the post-training stage."""

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, in_dim: int, num_freqs: int, log_scale: bool = True):
        super().__init__()
        if log_scale:
            freqs = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        else:
            freqs = torch.linspace(1.0, 2.0 ** (num_freqs - 1), num_freqs)
        self.register_buffer('freqs', freqs)
        self.out_dim = in_dim * (1 + 2 * num_freqs)

    def forward(self, x: Tensor) -> Tensor:
        parts = [x]
        for freq in self.freqs:
            parts.extend((torch.sin(freq * x), torch.cos(freq * x)))
        return torch.cat(parts, dim=-1)


class MeshDeformNet(nn.Module):
    """Fourier encoded position and time to vertex displacement."""

    def __init__(self, pos_freq: int = 6, time_freq: int = 4, hidden: int = 64):
        super().__init__()
        self.pos_enc = PositionalEncoding(3, pos_freq)
        self.time_enc = PositionalEncoding(1, time_freq)
        in_dim = self.pos_enc.out_dim + self.time_enc.out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )

    def forward(self, xyz: Tensor, t: Tensor) -> Tensor:
        time = t.expand(xyz.shape[0], 1)
        return self.net(torch.cat((self.pos_enc(xyz), self.time_enc(time)), dim=-1))


def masked_l1_loss(pred: Tensor, gt: Tensor, mask: Tensor) -> Tensor:
    return ((pred - gt).abs() * mask).sum() / mask.sum().clamp_min(1e-8)


def build_unique_edges(faces: Tensor) -> Tensor:
    """Return unique undirected edges from triangular faces."""
    indices = faces.long().cpu()
    edges = torch.cat((indices[:, [0, 1]], indices[:, [1, 2]], indices[:, [2, 0]]), dim=0)
    return torch.unique(torch.sort(edges, dim=1).values, dim=0)


def laplacian_smoothness_loss(verts: Tensor, edges: Tensor) -> Tensor:
    count = verts.shape[0]
    src, dst = edges[:, 0], edges[:, 1]
    ones = torch.ones(len(src), device=verts.device, dtype=verts.dtype)
    degree = torch.zeros(count, device=verts.device, dtype=verts.dtype)
    degree.index_add_(0, src, ones)
    degree.index_add_(0, dst, ones)
    neighbors = torch.zeros_like(verts)
    neighbors.index_add_(0, src, verts[dst])
    neighbors.index_add_(0, dst, verts[src])
    laplacian = verts - neighbors / degree.clamp_min(1.0)[:, None]
    return (laplacian ** 2).mean()
