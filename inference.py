import os

import torch

import config
import utils
from data import load_data
from model import Transformer
import tokenizer


def main() -> None:
    device = config.device

    src_tokenizer, tgt_tokenizer, test_loader = load_data(config.src_lang, config.tgt_lang, ['test'])
    dataset = test_loader.dataset

    model = Transformer(
            src_pad_idx=src_tokenizer.token_to_id(tokenizer.PAD_TOKEN),
            tgt_pad_idx=tgt_tokenizer.token_to_id(tokenizer.PAD_TOKEN),
            src_vocab_size=src_tokenizer.get_vocab_size(),
            tgt_vocab_size=tgt_tokenizer.get_vocab_size(),
            d_model=config.d_model,
            n_head=config.n_heads,
            max_len=config.max_len,
            ffn_depth=config.ffn_depth,
            n_layers=config.n_layers,
            dropout=config.dropout
    ).to(device)

    # TODO: Replace with latest state dict.
    state_dict = torch.load(os.path.join(config.checkpoint_dir, 'en_de_5.pth'))
    model.load_state_dict(state_dict['model'])

    num_samples = 5
    sample_lst = dataset[torch.randint(0, len(dataset), (num_samples,))]['translation']
    src_sentences = [sentence[config.src_lang] for sentence in sample_lst]
    tgt_sentences = [sentence[config.tgt_lang] for sentence in sample_lst]

    for i in range(num_samples):
        print(f'Source: \"{"".join(src_sentences[i])}\"')
        print(f'Ground Truth: \"{"".join(tgt_sentences[i])}\"')
        print(f'Translation: \"{"".join(utils.translate_sentence(src_sentences, model, src_tokenizer, tgt_tokenizer))}\"')
        print()


if __name__ == '__main__':
    main()
