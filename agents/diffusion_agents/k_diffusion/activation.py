import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim


def init_alpha_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def init_beta_index(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


"""
   自定义激活函数 PPRsigELU
"""


class PPRsigELU(nn.Module):
    def __init__(self):
        super(PPRsigELU, self).__init__()

        self.alpha = torch.nn.Parameter(torch.Tensor(1))
        init_alpha_half(self.alpha)

        self.beta = torch.nn.Parameter(torch.Tensor(1))
        init_beta_index(self.beta)

    def forward(self, x):
        return torch.where(x > 1.0, x * torch.sigmoid(x) * self.alpha + x,
                           torch.where(x <= 0.0, self.beta * (torch.exp(x) - 1.0), x))


"""
   自定义激活函数 MetaAconC
"""


class MetaAconC(nn.Module):
    def __init__(self, c1, r=8):
        super().__init__()
        c2 = max(r, c1 // r)
        self.p1 = nn.Parameter(torch.randn(1, 1, c1))
        self.p2 = nn.Parameter(torch.randn(1, 1, c1))
        self.fc1 = nn.Linear(c1, c2)
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x):
        y = x.mean(dim=1, keepdims=True)
        beta = torch.sigmoid(self.fc2(self.fc1(y)))
        dpx = (self.p1 - self.p2) * x
        return dpx * torch.sigmoid(beta * dpx) + self.p2 * x


"""
   自定义激活函数 AconC
"""


class AconC(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.p1 = nn.Parameter(torch.randn(1, 1, c1))
        self.p2 = nn.Parameter(torch.randn(1, 1, c1))
        self.beta = nn.Parameter(torch.ones(1, 1, c1))

    def forward(self, x):
        dpx = (self.p1 - self.p2) * x
        return dpx * torch.sigmoid(self.beta * dpx) + self.p2 * x


"""
   自定义激活函数 SwiGlu
"""

class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden_dim)
        self.fc2 = nn.Linear(embed_size, ff_hidden_dim)
        self.fc3 = nn.Linear(ff_hidden_dim, embed_size)

    def forward(self, x):
        gate = torch.sigmoid(self.fc1(x))
        transformed = torch.relu(self.fc2(x))
        output = gate * transformed
        return self.fc3(output)


class SwiGlu(nn.Module):
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.swish = nn.Linear(in_features, hidden_size)
        self.gate = nn.Linear(in_features, hidden_size)
        self.proj = nn.Linear(hidden_size, in_features)

    def forward(self, x):
        x1 = self.swish(x)
        swish1 = F.silu(x1)
        gate1 = self.gate(x)
        out = swish1 * gate1
        projection = self.proj(out)
        return projection
