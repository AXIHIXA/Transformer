import os

import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torchviz import make_dot

import config
import utils
from data import load_data
from model import Transformer, MultiheadAttention
from tokenizer import PAD_TOKEN


def dump_mha_computation_graph() -> None:
    q = torch.empty([32, 103, 512], dtype=torch.float, device=config.device)
    k = torch.empty([32, 103, 512], dtype=torch.float, device=config.device)
    v = torch.empty([32, 103, 512], dtype=torch.float, device=config.device)
    mha = MultiheadAttention(config.d_model, config.n_heads, config.dropout).to(config.device)
    attn_out = mha(q, k, v)
    make_dot(attn_out, show_attrs=True, params=dict(mha.named_parameters())).render('mha_computation_graph', format='pdf')


def profile_model() -> None:
    device = config.device

    src_tokenizer, tgt_tokenizer, test_loader = load_data(config.src_lang, config.tgt_lang, ['test'])
    dataset = test_loader.dataset

    model = Transformer(src_pad_idx=src_tokenizer.token_to_id(PAD_TOKEN),
                        tgt_pad_idx=tgt_tokenizer.token_to_id(PAD_TOKEN),
                        src_vocab_size=src_tokenizer.get_vocab_size(),
                        tgt_vocab_size=tgt_tokenizer.get_vocab_size(),
                        d_model=config.d_model,
                        n_head=config.n_heads,
                        max_len=config.max_len,
                        ffn_depth=config.ffn_depth,
                        n_layers=config.n_layers,
                        dropout=config.dropout).to(device)

    # TODO: Replace with latest state dict.
    state_dict = torch.load(os.path.join(config.checkpoint_dir, 'en_de_5.pth'))
    model.load_state_dict(state_dict['model'])

    num_samples = 1
    sample_lst = dataset[torch.randint(0, len(dataset), (num_samples,))]['translation']
    src_sentences = [sentence[config.src_lang] for sentence in sample_lst]
    tgt_sentences = [sentence[config.tgt_lang] for sentence in sample_lst]

    # Training.
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 profile_memory=True,
                 record_shapes=True) as prof:
        with record_function('model_train'):
            for batch in test_loader:
                src, tgt = batch
                tgt, gt = tgt[:, :-1], tgt[:, 1:]
                src = src.to(device)
                tgt = tgt.to(device)
                gt = gt.to(device)
                output = model(src, tgt)

    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

    # Inference.
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 profile_memory=True,
                 record_shapes=True) as prof:
        with record_function('model_inference'):
            utils.translate_sentence(src_sentences, model, src_tokenizer, tgt_tokenizer)

    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))


def main() -> None:
    # dump_mha_computation_graph()
    profile_model()


if __name__ == '__main__':
    main()
