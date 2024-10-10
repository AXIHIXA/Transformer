# Demo Transformer for English-to-German Translation

## Introduction

- A demo transformer (vanilla version from Attention Is All You Need paper) with frontends (trained tokenizer, dataset utilities and emcoding/decoding layers) tailered for English-to-German translation tasks.  
- Trained for 5 epoches over 1024 sentences and observed a drop in training error. 

## Dependencies

- pytorch (from conda)
- tokenizer (from PyPI)
- datasets (from PyPI)
- torchviz (from PyPI)

## Usage

### Train

```bash
python train.py
```

### Inference

```bash
python inference.py
```
