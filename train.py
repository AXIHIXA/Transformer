import os

import torch

import config
from data import load_data
from tokenizer import PAD_TOKEN
from model import Transformer
from utils import WarmupScheduler
from utils import LabelSmoothingLoss


def train_epoch(epoch, model, device,
                train_dataloader, tgt_tokenizer,
                criterion, optimizer, scheduler) -> float:
    model.train()
    total_loss = 0
    step = 0
    optimizer.zero_grad()

    for batch_id, batch in enumerate(train_dataloader):
        # Split batch.
        src, tgt = batch
        tgt, gt = tgt[:, :-1], tgt[:, 1:]
        src = src.to(device)
        tgt = tgt.to(device)
        gt = gt.to(device)

        # (batch_size, seqlen - 1, tgt_vocab_size,)
        output = model(src, tgt)

        # (batch_size * (seqlen - 1), tgt_vocab_size,)
        output = output.contiguous().view(-1, tgt_tokenizer.get_vocab_size())
        loss = criterion(output, gt.contiguous().view(-1))
        loss.backward()

        if (step + 1) % config.update_freq == 0 or (step + 1) == len(train_dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        step += 1
        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)

    return avg_loss


def train(restore_epoch: int = 0) -> None:
    device = config.device

    src_tokenizer, tgt_tokenizer, train_dataloader = load_data(config.src_lang, config.tgt_lang, ['train'])

    model = Transformer(
            src_pad_idx=src_tokenizer.token_to_id(PAD_TOKEN),
            tgt_pad_idx=tgt_tokenizer.token_to_id(PAD_TOKEN),
            src_vocab_size=src_tokenizer.get_vocab_size(),
            tgt_vocab_size=tgt_tokenizer.get_vocab_size(),
            d_model=config.d_model,
            n_head=config.n_heads,
            max_len=config.max_len,
            ffn_depth=config.ffn_depth,
            n_layers=config.n_layers,
            dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), betas=config.betas, eps=config.adam_eps)
    scheduler = WarmupScheduler(optimizer, config.d_model, config.warmup_step)
    criterion = LabelSmoothingLoss(ignore_index=tgt_tokenizer.token_to_id(PAD_TOKEN), label_smoothing=config.eps_ls)

    restore_ckpt_path = os.path.join(config.checkpoint_dir, f'en_de_{restore_epoch}.pth')

    if restore_epoch != 0:
        assert os.path.exists(restore_ckpt_path)
        ckpt = torch.load(restore_ckpt_path)
        assert ckpt['epoch'] == restore_epoch
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])

    for epoch in range(restore_epoch, config.epochs):
        avg_train_loss = train_epoch(epoch,
                                     model,
                                     device,
                                     train_dataloader,
                                     tgt_tokenizer,
                                     criterion,
                                     optimizer,
                                     scheduler)

        print(f'Epoch {epoch + 1}/{config.epochs}, Training Loss: {avg_train_loss: .4f}')
        checkpoint_path = os.path.join(config.checkpoint_dir, f'en_de_{epoch + 1}.pth')

        save_dict = {
            'epoch':     epoch + 1,
            'model':     model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }

        torch.save(save_dict, checkpoint_path)


def main() -> None:
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    train()


if __name__ == '__main__':
    main()
