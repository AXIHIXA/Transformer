import typing

import datasets
import torch
from torch.utils.data import DataLoader

import config
import tokenizer


def load_data(src_lang: str, tgt_lang: str, splits: typing.Optional[list[str]] = None):
    """
    Load IWSLT 2017 dataset with pre-trained tokenizers.
    :param src_lang:
    :param tgt_lang:
    :param splits:   Choose one or multiple from "train_epoch", "test" and "validation".
                     If not speficied, all splits will be loaded.
    :return: src_tokenizer, tgt_tokenizer, list[splits of dataset].
    """
    if sorted((src_lang, tgt_lang)) != ['de', 'en']:
        raise ValueError("Available language options are ('de','en') and ('en', 'de')")

    all_splits = ['train', 'validation', 'test']

    if splits is None:
        splits = all_splits
    elif not set(splits).issubset(all_splits):
        raise ValueError(f'Splits should only contain some of {all_splits}')

    dataset = datasets.load_dataset('iwslt2017', f'iwslt2017-{src_lang}-{tgt_lang}', trust_remote_code=True)

    # TODO: Revert to actual size when really training something.
    for split in splits:
        dataset[split] = dataset[split].select(range(config.dataset_size[split]))

    src_tokenizer = tokenizer.get_tokenizer(src_lang)
    tgt_tokenizer = tokenizer.get_tokenizer(tgt_lang)

    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for item in batch:
            src_batch.append(item['translation'][src_lang])
            tgt_batch.append(item['translation'][tgt_lang])

        src_batch = src_tokenizer.encode_batch(src_batch)
        tgt_batch = tgt_tokenizer.encode_batch(tgt_batch)

        src_tensor = torch.LongTensor([item.ids for item in src_batch])
        tgt_tensor = torch.LongTensor([item.ids for item in tgt_batch])

        if src_tensor.shape[-1] < tgt_tensor.shape[-1]:
            src_tensor = torch.nn.functional.pad(src_tensor,
                                                 [0, tgt_tensor.shape[-1] - src_tensor.shape[-1]],
                                                 value=src_tokenizer.token_to_id(tokenizer.PAD_TOKEN))
        else:
            tgt_tensor = torch.nn.functional.pad(tgt_tensor,
                                                 [0, src_tensor.shape[-1] - tgt_tensor.shape[-1]],
                                                 value=tgt_tokenizer.token_to_id(tokenizer.PAD_TOKEN))

        return src_tensor, tgt_tensor

    dataloaders = [
        DataLoader(dataset[split],
                   batch_size=config.batch_size,
                   collate_fn=collate_fn,
                   shuffle=split == 'train')
        for split in splits
    ]

    return src_tokenizer, tgt_tokenizer, *dataloaders
