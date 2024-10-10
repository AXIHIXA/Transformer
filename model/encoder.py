import torch

from .layers import *


class EncoderLayer(torch.nn.Module):
    r"""
    One Layer in encoder stack.
    """
    def __init__(self, d_model, ffn_depth, n_heads, dropout):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.norm1 = LayerNorm(normalized_shape=d_model)
        self.ffn = PositionwiseFeedForward(d_model=d_model, ffn_depth=ffn_depth, dropout=dropout)
        self.norm2 = LayerNorm(normalized_shape=d_model)

    def forward(self, x, src_mask):
        r"""
        Forward.
        :param x: (batch_size, seq_len, d_model,)
        :param src_mask:
        :return:
        """
        residual = x
        x = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + residual)

        residual = x
        x = self.ffn(x)
        x = self.norm2(x + residual)

        return x


class Encoder(torch.nn.Module):
    r"""
    Encoder stack.
    """

    def __init__(self, d_model, ffn_depth, n_heads, n_layers, dropout):
        super().__init__()

        self.layers = torch.nn.ModuleList([
            EncoderLayer(d_model=d_model, ffn_depth=ffn_depth, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, src_mask):
        for sub_layer in self.layers:
            x = sub_layer(x, src_mask)

        return x
