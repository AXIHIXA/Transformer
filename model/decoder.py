import torch

from .layers import *


class DecoderLayer(torch.nn.Module):
    r"""
    One layer in decoder stack.
    """

    def __init__(self, d_model, ffn_depth, n_heads, dropout):
        super().__init__()

        self.self_attn = MultiheadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.norm1 = LayerNorm(normalized_shape=d_model)

        self.enc_dec_attention = MultiheadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.norm2 = LayerNorm(normalized_shape=d_model)

        self.ffn = PositionwiseFeedForward(d_model=d_model, ffn_depth=ffn_depth, dropout=dropout)
        self.norm3 = LayerNorm(normalized_shape=d_model)

    def forward(self, dec, enc, tgt_mask, src_mask):
        residual = dec
        x = self.self_attn(q=dec, k=dec, v=dec, mask=tgt_mask)
        x = self.norm1(x + residual)

        # Cross attention.
        if enc is not None:
            residual = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
            x = self.norm2(x + residual)

        residual = x
        x = self.ffn(x)
        x = self.norm3(x + residual)

        return x


class Decoder(torch.nn.Module):
    def __init__(self, d_model, ffn_depth, n_heads, n_layers, dropout):
        super().__init__()

        self.layers = torch.nn.ModuleList([
            DecoderLayer(
                    d_model=d_model,
                    ffn_depth=ffn_depth,
                    n_heads=n_heads,
                    dropout=dropout,
            )
            for _ in range(n_layers)
        ])

    def forward(self, tgt, enc_src, tgt_mask, src_mask):
        for sub_layer in self.layers:
            tgt = sub_layer(tgt, enc_src, tgt_mask, src_mask)

        return tgt
