import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import math

class embedding(nn.Module):
    def __init__(self, seg_len, d_model):
        super(embedding, self).__init__()
        self.seg_len = seg_len

        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x):
        batch, ts_len, ts_dim = x.shape
        x_embed = self.linear(x)
        return x_embed