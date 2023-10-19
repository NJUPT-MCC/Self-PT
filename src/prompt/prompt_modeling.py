# # The codes are from https://github.com/NJUPT-MCC/Self-PT

# # Self-PT: Adaptive Self-Prompt Tuning for Low-Resource Visual Question Answering. ACM MM 2023

import torch
import torch.nn as nn

class InputPrompts(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.prompt_len = config.prompt_len
        self.input_dim = config.input_dim
        self.mid_dim = config.mid_dim

        self.prefix_tokens = torch.arange(self.prompt_len).long()
        self.prefix_embedding = nn.Sequential(
            nn.Embedding(self.prompt_len, self.input_dim),
            nn.Linear(self.input_dim, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.input_dim),
        )

    def get_prompt(self, bsz, device):
        input_tokens = self.prefix_tokens.unsqueeze(0).expand(bsz, -1).to(device) # (B, L)
        prefix_prompt = self.prefix_embedding(input_tokens) # (B, L, d_model * n_heads * n_layer)
        
        return prefix_prompt


from torch.nn.parameter import Parameter
import torch
from torch import nn
from torch import Tensor
from torch.nn import init
import math
from torch.nn import functional as F


class PHMLayer(nn.Module):
    """
    Self-PT: Adaptive Self-Prompt Tuning for Low-Resource Visual Question Answering
    """
    def __init__(self, n, in_features, out_features):
        super(PHMLayer, self).__init__()
        self.n = n
        self.in_features = in_features
        self.out_features = out_features

        self.bias = Parameter(torch.Tensor(out_features))

        self.a = torch.zeros((n, n, n))
        self.a = Parameter(torch.nn.init.xavier_uniform_(self.a))

        self.s = torch.zeros((n, self.out_features // n, self.in_features // n))
        self.s = Parameter(torch.nn.init.xavier_uniform_(self.s))

        self.weight = torch.zeros((self.out_features, self.in_features))

        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def kronecker_product1(self, a, b):
        siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
        res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
        siz0 = res.shape[:-4]
        out = res.reshape(siz0 + siz1)
        return out

    def forward(self, input: Tensor) -> Tensor:
        self.weight = torch.sum(self.kronecker_product1(self.a, self.s), dim=0)
        input = input.type(dtype=self.weight.type())
        return F.linear(input, weight=self.weight, bias=self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.a, a=math.sqrt(5))
        init.kaiming_uniform_(self.s, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.placeholder)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)


