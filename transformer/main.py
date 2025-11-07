import os
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
dir_stat = os.path.join(os.path.dirname(__file__), "results", timestamp, "statistics")
dir_ckpt = os.path.join(os.path.dirname(__file__), "results", timestamp, "checkpoints")
os.makedirs(dir_stat, exist_ok=True)
os.makedirs(dir_ckpt, exist_ok=True)

import logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename=f"{dir_stat}/output.log",
    filemode="a"
)
logger = logging.getLogger(__name__)

import math
import time
from conf import *

from torch import nn, optim
from torch.optim import Adam

from data import *
from models.model.transformer import Transformer
from utils.bleu import idx_to_word, get_bleu
from utils.epoch_timer import epoch_time


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output_reshape, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        logger.info(f"Step: {round((i / len(iterator)) * 100, 2):.2f}% , loss: {loss.item()}")

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            total_bleu = []
            for j in range(batch_size):
                try:
                    trg_words = idx_to_word(batch.trg[j], loader.target.vocab)
                    output_words = output[j].max(dim=1)[1]
                    output_words = idx_to_word(output_words, loader.target.vocab)
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except:
                    pass

            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / len(iterator), batch_bleu


def run(total_epoch, best_loss):
    train_losses, valid_losses, bleus = [], [], []
    
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, clip)
        valid_loss, bleu = evaluate(model, valid_iter, criterion)
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        bleus.append(bleu)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), f"{dir_ckpt}/model-{valid_loss}.pt")

        f = open(f"{dir_stat}/train_loss.txt", 'w')
        f.write(str(train_losses))
        f.close()

        f = open(f"{dir_stat}/bleu.txt", 'w')
        f.write(str(bleus))
        f.close()

        f = open(f"{dir_stat}/valid_loss.txt", 'w')
        f.write(str(valid_losses))
        f.close()

        logger.info(f"Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s")
        logger.info(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}")
        logger.info(f"\tValid Loss: {valid_loss:.3f} |  Valid PPL: {math.exp(valid_loss):7.3f}")
        logger.info(f"\tBLEU Score: {bleu:.3f}")


if __name__ == "__main__":
    model = Transformer(src_pad_idx=src_pad_idx,
                        trg_pad_idx=trg_pad_idx,
                        trg_sos_idx=trg_sos_idx,
                        d_model=d_model,
                        enc_voc_size=enc_voc_size,
                        dec_voc_size=dec_voc_size,
                        max_len=max_len,
                        ffn_hidden=ffn_hidden,
                        n_head=n_heads,
                        n_layers=n_layers,
                        drop_prob=drop_prob,
                        device=device).to(device)
    
    logger.info(f"The model has {count_parameters(model):,} trainable parameters")
    model.apply(initialize_weights)
    optimizer = Adam(params=model.parameters(),
                    lr=init_lr,
                    weight_decay=weight_decay,
                    eps=adam_eps)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                    verbose=True,
                                                    factor=factor,
                                                    patience=patience)

    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
    
    run(total_epoch=epoch, best_loss=math.inf)
