import os

from tokenizers import Tokenizer
from tokenizers.models import BPE

import config


SOS_TOKEN = '[SOS]'
EOS_TOKEN = '[EOS]'
PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'


special_tokens = [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN]


def get_tokenizer(lang):
    tokenizer_path = os.path.join(config.tokenizer_dir, f'tokenizer-{lang}.json')
    assert os.path.exists(tokenizer_path)
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN)).from_file(str(tokenizer_path))
    return tokenizer
