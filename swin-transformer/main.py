import logging
import os
import time
from datetime import datetime

import torch
import torch.distributed as dist
import wandb
from conf import (
    batch_size,
    betas,
    cutmix,
    cutmix_minmax,
    device,
    downscaling_factors,
    eps,
    head_dim,
    heads,
    hidden_dim,
    in_channels,
    label_smoothing,
    layers,
    lr,
    min_lr,
    mixup,
    mixup_mode,
    mixup_prob,
    mixup_switch_prob,
    n_epochs,
    num_classes,
    warmup_epochs,
    warmup_lr,
    weight_decay,
    window_size,
)
from data import dataloader_train, dataloader_val
from models.model.swin_transformer import SwinTransformer
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.utils import AverageMeter, NativeScaler, accuracy
from torch import optim as optim
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
    project="algorithm-swin-transformer",
    name=timestamp,
    config={
        "epoch": n_epochs,
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "layers": layers,
        "heads": heads,
        "head_dim": head_dim,
        "window_size": window_size,
        "downscaling_factors": downscaling_factors,
        "num_classes": num_classes,
        "n_heads": heads,
        "optimizer": "AdamW",
    },
)


if mixup > 0.0:
    # smoothing is handled with mixup label transform
    criterion = SoftTargetCrossEntropy()
elif label_smoothing:
    criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
else:
    criterion = torch.nn.CrossEntropyLoss()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(
    model, criterion, dataloader, optimizer, epoch, device, loss_scaler, max_norm=0, mixup_fn=None
):
    model.train()

    num_steps = len(dataloader)
    time_meter = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    total = 0
    correct = 0

    end = time.time()
    for batch_idx, (samples, targets) in enumerate(dataloader):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        targets_true = targets.clone()

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.amp.autocast("cuda"):
            outputs = model(samples)
            loss = criterion(outputs, targets)

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets_true).sum().item()

        optimizer.zero_grad()

        is_second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        grad_norm = loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=is_second_order,
        )

        lr_scheduler.step_update(epoch * num_steps + batch_idx)

        loss_scale_value = loss_scaler.state_dict()["scale"]
        torch.cuda.synchronize()

        loss_meter.update(loss.item(), samples.size(0))
        # loss_scaler return None if not update
        if grad_norm is not None:
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        time_meter.update(time.time() - end)
        end = time.time()

        progress_bar(
            batch_idx,
            len(dataloader),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (loss_meter.avg, 100.0 * correct / total, correct, total),
        )
        logger.info(
            f"Step: {round((batch_idx / len(dataloader)) * 100, 2):.2f}% , loss: {loss.item()}"
        )
    return {
        "loss": loss_meter.avg,
        "grad_norm": norm_meter.avg,
        "loss_scale": scaler_meter.avg,
        "time": time_meter.avg,
    }


def evaluate(model, dataloader, device, optimizer, loss_scaler, dir_ckpt):
    criterion = torch.nn.CrossEntropyLoss()
    global best_acc
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    correct = 0
    total = 0

    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(
                device, non_blocking=True
            )

            # compute output
            outputs = model(inputs)

            # measure accuracy and record loss
            loss = criterion(outputs, targets)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

            # update meters
            loss_meter.update(loss.item(), targets.size(0))
            acc1_meter.update(acc1.item(), targets.size(0))
            acc5_meter.update(acc5.item(), targets.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(
                batch_idx,
                len(dataloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (loss_meter.avg, 100.0 * correct / total, correct, total),
            )

    # Save checkpoint.
    acc = acc1_meter.avg
    if acc > best_acc:
        print("Saving models...")
        state = {
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": loss_scaler.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        ckpt_file = os.path.join(dir_ckpt, "ckpt_swin_best.pth")
        torch.save(state, ckpt_file)
        best_acc = acc

    # Log
    content = (
        time.ctime()
        + f"Epoch {epoch+1}, val loss: {loss_meter.avg:.5f}, "
        + f"acc@1: {(acc1_meter.avg):.5f}, acc@5: {(acc5_meter.avg):.5f}"
    )
    print(content)
    logger.info(content)
    return {
        "loss": loss_meter.avg,
        "acc1": acc1_meter.avg,
        "acc5": acc5_meter.avg,
        "time": batch_time.avg,
    }


if __name__ == "__main__":
    best_acc = 0.0
    # model
    net = SwinTransformer(
        hidden_dim=hidden_dim,
        layers=layers,
        heads=heads,
        channels=in_channels,
        num_classes=num_classes,
        head_dim=head_dim,
        window_size=window_size,
        downscaling_factors=downscaling_factors,
        relative_pos_embedding=True,
    ).to(device)

    logger.info(f"The model has {count_parameters(net):,} trainable parameters")
    print(f"Using device: {device}")
    logger.info(f"Using device: {device}")

    # mixup
    mixup_fn = None
    mixup_active = mixup > 0 or cutmix > 0.0 or cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=mixup,
            cutmix_alpha=cutmix,
            cutmix_minmax=cutmix_minmax,
            prob=mixup_prob,
            switch_prob=mixup_switch_prob,
            mode=mixup_mode,
            label_smoothing=label_smoothing,
            num_classes=num_classes,
        )

    # linear scale the learning rate according to total batch size
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    linear_scaled_lr = lr * batch_size * window_size / 512.0
    linear_scaled_warmup_lr = warmup_lr * batch_size * window_size / 512.0
    linear_scaled_min_lr = min_lr * batch_size * window_size / 512.0
    lr = linear_scaled_lr
    warmup_lr = linear_scaled_warmup_lr
    min_lr = linear_scaled_min_lr

    # optimizer
    optimizer = optim.AdamW(
        net.parameters(), eps=eps, betas=betas, lr=lr, weight_decay=weight_decay
    )
    loss_scaler = NativeScaler()

    # scheduler
    num_steps = int(n_epochs * len(dataloader_train))
    warmup_steps = int(warmup_epochs * len(dataloader_train))
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_steps - warmup_steps,
        lr_min=min_lr,
        warmup_lr_init=warmup_lr,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
        warmup_prefix=True,
    )

    # training loop
    for epoch in range(n_epochs):
        if torch.distributed.is_initialized():
            dataloader_train.sampler.set_epoch(epoch)
        lr_scheduler.step(epoch + 1)

        train_stats = train(
            net,
            criterion,
            dataloader_train,
            optimizer,
            epoch,
            device,
            loss_scaler,
            max_norm=5.0,
            mixup_fn=mixup_fn,
        )

        eval_stats = evaluate(net, dataloader_val, device, optimizer, loss_scaler, dir_ckpt)

        logger.info(f"Epoch: {epoch + 1}")
        logger.info(f"\tTrain Loss: {train_stats['loss']:.3f}")

        wandb.log(
            {
                "epoch": epoch + 1,
                "train/loss": train_stats["loss"],
                "train/grad_norm": train_stats["grad_norm"],
                "train/loss_scale": train_stats["loss_scale"],
                "train/time": train_stats["time"],
                "lr": optimizer.param_groups[0]["lr"],
                "valid/loss": eval_stats["loss"],
                "valid/acc1": eval_stats["acc1"],
                "valid/acc5": eval_stats["acc5"],
                "valid/time": eval_stats["time"],
            }
        )
