import os
from pathlib import Path

import torch

# Global task settings (English-to-German translation).
src_lang = 'en'
tgt_lang = 'de'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Reproductibility.
torch.manual_seed(3407)

# model parameter setting (Transformer base).
max_len = 512
d_model = 512
n_layers = 6
n_heads = 8
ffn_depth = 2048
dropout = 0.1

# training setting.
batch_size = 32
update_freq = 16
epochs = 20
eps_ls = 0.1  # eps for label smoothing.
warmup_step = 4000
clip = 1

# For demo purpose, limit data size.
dataset_size = {'train': 1024, 'validation': 128, 'test': 512}

# optimizer parameter setting.
betas = (0.9, 0.98)
adam_eps = 1e-9

# path.
project_root_dir = str(Path(__file__).parent.parent.resolve())
checkpoint_dir = os.path.join(project_root_dir, 'checkpoints')
tokenizer_dir = os.path.join(project_root_dir, 'tokenizer', 'IWSLT17')

# inference.
num_beams = 3
top_k = 30
top_p = 0.7
temperature = 1.0
length_penalty = 0.7
