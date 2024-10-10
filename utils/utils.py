import math
import typing

import torch

import config
import tokenizer


def make_pad_mask(seq, pad_idx):
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seqlen,)
    return mask.to(seq.device)


def make_causal_mask(seq):
    _, seqlen = seq.size()
    mask = torch.tril(torch.ones((seqlen, seqlen), device=seq.device)).bool()  # (seq_len, seq_len,)
    return mask


def make_tgt_mask(tgt, pad_idx):
    tgt_pad_mask = make_pad_mask(tgt, pad_idx)  # (batch_size, 1, 1, seqlen,)
    tgt_sub_mask = make_causal_mask(tgt)  # (seqlen, seqlen,)
    tgt_mask = tgt_pad_mask & tgt_sub_mask.unsqueeze(0).unsqueeze(0)  # (batch_size, 1, seqlen, seqlen,)
    return tgt_mask


class PositionalEncoding(torch.nn.Module):
    r"""
    Positinoal encoding.
    """
    def __init__(self, d_model, max_len):
        super().__init__()

        # Initialize position encoding matrix with shape (max_len, d_model,).
        pe = torch.zeros(max_len, d_model)

        # Create a tensor with shape (max_len, 1,) with position indices.
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute the div_term with shape (d_model // 2,) for the sin and cos functions.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sin and cos to even and odd locations.
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # No grads needed, does not need to register as paremeter.
        self.register_buffer("pe", pe)

    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.pe[:seq_len, :]  # (seqlen, d_model,)


class LabelSmoothingLoss(torch.nn.Module):
    r"""
    Label Smoothing Loss replaces Cross Entropy loss.
    Degrades to Cross Entropy if label_smoothing == 0.
    """
    def __init__(self, ignore_index, label_smoothing):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, pred, target):
        smoothing = self.label_smoothing
        pad_idx = self.ignore_index
        classes = pred.shape[-1]

        if smoothing == 0:
            return torch.nn.functional.cross_entropy(pred, target, ignore_index=pad_idx)

        log_probs = torch.nn.functional.log_softmax(pred, dim=-1)

        with torch.no_grad():
            one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
            one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (classes - 1)
            mask = torch.nonzero(target == pad_idx)

            if 0 < mask.dim():
                one_hot.index_fill_(0, mask.squeeze(), 0.0)

        return torch.mean(torch.sum(-one_hot * log_probs, dim=-1))


class WarmupScheduler(torch.optim.lr_scheduler.LRScheduler):
    r"""
    Re-implementation of Attention Is All You Need.
    """

    def __init__(self,
                 optimizer,
                 d_model: int,
                 warmup_step: int,
                 last_epoch: int = -1,
                 ) -> None:
        # Assign these fields prior to calling super().__init__()!
        self.d_model = d_model
        self.warmup_step = warmup_step
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            self.d_model ** -0.5 *
            min((self.last_epoch + 1) ** -0.5,
                (self.last_epoch + 1) * self.warmup_step ** -1.5)
        ]


@torch.no_grad()
def greedy_search(model, memory, memory_mask, max_len, sos_idx, eos_idx, pad_idx):
    batch_size, seq_len, d_model = memory.shape
    ys = torch.ones(batch_size, 1, dtype=torch.long, device=memory.device).fill_(sos_idx)
    ended = torch.zeros(batch_size, dtype=torch.bool, device=memory.device)

    for i in range(max_len - 1):
        logits = model.decode(ys, memory, memory_mask)[:, -1]
        next_words = torch.argmax(logits, dim=1)

        ys = torch.cat([ys, next_words.unsqueeze(1)], dim=1)
        ended = ended | (next_words == eos_idx)
        ys[ended & (ys[:, -1] != eos_idx), -1] = pad_idx

        if ended.all():
            break

    # Reached max length...
    if i == max_len - 2:
        ys[~ended, -1] = eos_idx
        ys[ended, -1] = pad_idx

    return ys


@torch.no_grad()
def translate_sentence(
        sentences: typing.Union[list[str], str],
        model,
        src_tokenizer,
        tgt_tokenizer,
        max_len=config.max_len,
) -> list[str]:
    r"""
    Translate the input sentence with greedy search algorithm.
    :param sentences:
    :param model:
    :param src_tokenizer:
    :param tgt_tokenizer:
    :param max_len:
    :return: Translated sentences.
    """
    if isinstance(sentences, str):
        sentences = [sentences]

    device = next(model.parameters()).device

    sos_idx = tgt_tokenizer.token_to_id(tokenizer.SOS_TOKEN)
    eos_idx = tgt_tokenizer.token_to_id(tokenizer.EOS_TOKEN)
    pad_idx = tgt_tokenizer.token_to_id(tokenizer.PAD_TOKEN)

    src_tensor = torch.LongTensor([encoding.ids for encoding in src_tokenizer.encode_batch(sentences)]).to(device)
    memory = model.encode(src_tensor)
    memory_mask = make_pad_mask(src_tensor, src_tokenizer.token_to_id(tokenizer.PAD_TOKEN))

    tgt_tokens = greedy_search(model,
                               memory,
                               memory_mask,
                               max_len,
                               sos_idx,
                               eos_idx,
                               pad_idx)

    return [''.join(s) for s in tgt_tokenizer.decode_batch(tgt_tokens.cpu().numpy())]
