import logging
import os
import time
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import wandb
from conf import (
    batch_size,
    depth,
    device,
    dim_head,
    dropout,
    emb_dropout,
    heads,
    image_size,
    lr,
    mlp_dim,
    n_epochs,
    num_classes,
    patch_size,
)
from data import test_loader, train_loader
from models.model.vision_transformer import VisionTransformer
from utils.util import progress_bar

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
dir_stat = os.path.join(os.path.dirname(__file__), "results", timestamp, "statistics")
dir_ckpt = os.path.join(os.path.dirname(__file__), "results", timestamp, "checkpoints")
os.makedirs(dir_stat, exist_ok=True)
os.makedirs(dir_ckpt, exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename=f"{dir_stat}/output.log",
    filemode="a",
)
logger = logging.getLogger(__name__)

wandb.init(
    project="algorithm-vision-transformer",
    name=timestamp,
    config={
        "epoch": n_epochs,
        "batch_size": batch_size,
        "d_model": mlp_dim,
        "n_layers": depth,
        "n_heads": heads,
        "lr": lr,
        "dropout": dropout,
        "optimizer": "Adam",
    },
)

criterion = nn.CrossEntropyLoss()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# train
def train():
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.amp.autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=False
        ):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(
            batch_idx,
            len(train_loader),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        )
        logger.info(
            f"Step: {round((batch_idx / len(train_loader)) * 100, 2):.2f}% , loss: {loss.item()}"
        )
    return train_loss / (batch_idx + 1)


# Validation
def test(dir_stat, dir_ckpt):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(
                batch_idx,
                len(test_loader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (test_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
            )

    # Save checkpoint.
    acc = 100.0 * correct / total
    if acc > best_acc:
        print("Saving models...")
        state = {
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        ckpt_file = os.path.join(dir_ckpt, f"vit_cifar10_patch{patch_size}_best.t7")
        torch.save(state, ckpt_file)
        best_acc = acc

    # Log
    content = (
        time.ctime()
        + " "
        + f"Epoch {epoch+1}, lr: {optimizer.param_groups[0]['lr']:.7f}, "
        + f"val loss: {test_loss:.5f}, acc: {(acc):.5f}"
    )
    print(content)
    logger.info(content)
    return test_loss, acc


if __name__ == "__main__":
    best_acc = 0
    start_epoch = 0
    net = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim_head,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        emb_dropout=emb_dropout,
    )
    logger.info(f"The model has {count_parameters(net):,} trainable parameters")
    print(f"Using device: {device}")
    logger.info(f"Using device: {device}")
    # make parallel
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    scaler = torch.amp.GradScaler(enabled=False)

    # net.cuda()
    for epoch in range(start_epoch, n_epochs):
        start = time.time()
        train_loss = train()
        val_loss, acc = test(dir_stat, dir_ckpt)

        # step cosine scheduling
        scheduler.step(epoch - 1)

        logger.info(f"Epoch: {epoch + 1}")
        logger.info(f"\tTrain Loss: {train_loss:.3f}")
        logger.info(f"\tValid Loss: {val_loss:.3f}")
        logger.info(f"\tAccuracy: {acc:.3f}")

        wandb.log(
            {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "valid/loss": val_loss,
                "valid/acc": acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
