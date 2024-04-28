import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from embed import embedding

from math import ceil

class Customformer(nn.Module):
    def __init__(self, seg_len, d_model=64, device=torch.device('cuda:0')):
        super(Customformer, self).__init__()
        self.seg_len = seg_len
        self.out_len = 52
        # Embedding
        self.enc_value_embedding = embedding(seg_len, d_model)
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, seg_len, d_model))
        self.pre_norm = nn.LayerNorm(d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=4, batch_first=True)
        
    def forward(self, x_seq):
        base=0
        batch_size = x_seq.shape[0]
        x_seq = self.enc_value_embedding(x_seq)
        x_seq = self.pre_norm(x_seq)

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d d -> (repeat b) ts_d d', repeat = batch_size)
        predict_y = self.transformer(x_seq, dec_in)


        return base + predict_y[:, :self.out_len, :]