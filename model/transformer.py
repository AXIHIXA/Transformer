import torch

import utils

from .decoder import Decoder
from .encoder import Encoder
from .layers import InputEmbedding


class Transformer(torch.nn.Module):
    r"""
    Vanilla Transformer model.
    """

    def __init__(self,
                 src_pad_idx,
                 tgt_pad_idx,
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model,
                 n_head,
                 max_len,
                 ffn_depth,
                 n_layers,
                 dropout,
                 ) -> None:
        """
        Transformer.
        :param src_pad_idx: Padding index for the source sequences.
        :param tgt_pad_idx: Padding index for the target sequences.
        :param src_vocab_size: Vocabulary size of the source language dictionary.
        :param tgt_vocab_size: Vocabulary size of the target language dictionary.
        :param d_model: Dimensionality of the model.
        :param n_head: Number of attention heads.
        :param max_len: Maximum sequence length.
        :param ffn_depth: Depth of the feed-forward network.
        :param n_layers: Number of layers in the encoder and decoder.
        :param dropout: Dropout probability.
        """

        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.src_embedding = InputEmbedding(d_model=d_model,
                                            max_len=max_len,
                                            vocab_size=src_vocab_size,
                                            dropout=dropout)

        self.tgt_embedding = InputEmbedding(d_model=d_model,
                                            dropout=dropout,
                                            max_len=max_len,
                                            vocab_size=tgt_vocab_size)

        self.encoder = Encoder(d_model=d_model,
                               n_heads=n_head,
                               ffn_depth=ffn_depth,
                               dropout=dropout,
                               n_layers=n_layers)

        self.decoder = Decoder(d_model=d_model,
                               n_heads=n_head,
                               ffn_depth=ffn_depth,
                               dropout=dropout,
                               n_layers=n_layers)

        self.linear = torch.nn.Linear(d_model, tgt_vocab_size, bias=False)

        self.__initialize_parameters()

    def forward(self, src, tgt):
        encoded_src = self.encode(src)
        decoder_mask = utils.make_pad_mask(src, self.src_pad_idx)
        out = self.decode(tgt, encoded_src, decoder_mask)
        return out

    def encode(self, src):
        return self.encoder(self.src_embedding(src), utils.make_pad_mask(src, self.src_pad_idx))

    def decode(self, tgt, memory, memory_mask):
        out = self.decoder(self.tgt_embedding(tgt), memory, utils.make_tgt_mask(tgt, self.tgt_pad_idx), memory_mask)
        out = self.linear(out)
        return out

    def __initialize_parameters(self):
        for p in self.parameters():
            if 1 < p.dim():
                torch.nn.init.xavier_uniform_(p)
