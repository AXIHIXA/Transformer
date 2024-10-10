import math

import torch

import utils


class InputEmbedding(torch.nn.Module):
    """
    Input embedding.
    """
    def __init__(self, vocab_size, d_model, max_len, dropout):
        super().__init__()
        self.input_embedding = torch.nn.Embedding(vocab_size, d_model)
        self.positional_embedding = utils.PositionalEncoding(d_model, max_len)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        r"""
        Input embedding.
        :param x: (batch_size, seqlen,)
        :return: (batch_size, seqlen, d_model,)
        """
        tok_emb = self.input_embedding(x)       # (batch_size, seqlen, d_model,)
        pos_emb = self.positional_embedding(x)  # (seq_len, d_model,)
        return self.dropout(tok_emb + pos_emb)


class LayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps: int = 1e-5, elementwise_affine: bool = True):
        r"""
        Learnable LayerNorm.
        :param normalized_shape:
        :param eps:
        :param elementwise_affine: Whether to use elementwise affine transformation.
        """
        super().__init__()

        self.normalized_shape = normalized_shape
        self.eps: float = eps
        self.elementwise_affine: bool = elementwise_affine

        if self.elementwise_affine:
            self.gamma = torch.nn.Parameter(torch.ones(normalized_shape))
            self.beta = torch.nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.gamma = None
            self.beta = None

    def forward(self, x):
        """
        :param x: (batch_size, ..., self.normalized_shape,)
        :return: Same shape.
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x_normalized = (x - mean) / (std + self.eps)

        if self.elementwise_affine:
            x_normalized = self.gamma * x_normalized + self.beta

        return x_normalized


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(q, k, v, mask=None):
        batch_size, n_heads, seqlen, embed_dim = k.size()
        attn_output = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(embed_dim)

        if mask is not None:
            attn_output = attn_output.masked_fill(mask.logical_not(), float('-inf'))

        attn_output = torch.nn.functional.softmax(attn_output, dim=-1)
        attn_output = torch.matmul(attn_output, v)

        return attn_output


class MultiheadAttention(torch.nn.Module):
    r"""
    Multihead attention.
    """
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_head = n_heads
        self.attention = ScaledDotProductAttention()
        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        q, k, v = self.__split(q), self.__split(k), self.__split(v)
        out = self.attention(q, k, v, mask=mask)
        out = self.__concat(out)
        out = self.out_proj(out)
        return self.dropout(out)

    def __split(self, tensor):
        batch_size, length, d_model = tensor.size()
        d_k = d_model // self.n_head
        return tensor.view(batch_size, length, self.n_head, d_k).transpose(1, 2)

    @staticmethod
    def __concat(tensor):
        batch_size, head, length, d_key = tensor.size()
        d_model = head * d_key

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


class PositionwiseFeedForward(torch.nn.Module):
    r"""
    Just MLP.
    """
    def __init__(self, d_model, ffn_depth, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = torch.nn.Linear(d_model, ffn_depth)
        self.linear2 = torch.nn.Linear(ffn_depth, d_model)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)
