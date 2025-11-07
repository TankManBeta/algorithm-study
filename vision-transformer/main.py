import os
import csv
import time
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from conf import *
from datetime import datetime
from utils.util import progress_bar
from data import train_loader, test_loader
from models.model.vision_transformer import VisionTransformer


criterion = nn.CrossEntropyLoss()

# train
def train(epoch):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=False):
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

        progress_bar(batch_idx, len(train_loader), "Loss: %.3f | Acc: %.3f%% (%d/%d)" % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1)


# Validation
def test(epoch, dir_stat, dir_ckpt):
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
            progress_bar(batch_idx, len(test_loader), "Loss: %.3f | Acc: %.3f%% (%d/%d)" % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print("Saving..")
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
    content = time.ctime() + ' ' + f"Epoch {epoch}, lr: {optimizer.param_groups[0]['lr']:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}"
    print(content)
    log_file = os.path.join(dir_stat, f"log_vit_cifar10_patch{patch_size}.txt")
    with open(log_file, 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc


if __name__ == "__main__":
    best_acc = 0
    start_epoch = 0
    net = VisionTransformer(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=dim_head, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout, emb_dropout=emb_dropout)
    print(device)
    # make parallel
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    scaler = torch.amp.GradScaler(enabled=False)
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    dir_stat = os.path.join(os.path.dirname(__file__), "results", timestamp, "statistics")
    dir_ckpt = os.path.join(os.path.dirname(__file__), "results", timestamp, "checkpoints")
    os.makedirs(dir_stat, exist_ok=True)
    os.makedirs(dir_ckpt, exist_ok=True)
    
    list_loss = []
    list_acc = []
    # net.cuda()
    for epoch in range(start_epoch, n_epochs):
        start = time.time()
        train_loss = train(epoch)
        val_loss, acc = test(epoch, dir_stat, dir_ckpt)
        
        # step cosine scheduling
        scheduler.step(epoch-1)
        
        list_loss.append(val_loss)
        list_acc.append(acc)

        # Write out csv.
        csv_file = os.path.join(dir_stat, f"log_vit_cifar10_patch{patch_size}.csv")
        with open(csv_file, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(list_loss)
            writer.writerow(list_acc)
        print(list_loss)
