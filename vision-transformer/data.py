import os

import torch
import torchvision
import torchvision.transforms as transforms
from conf import batch_size, image_size, mean, std

dataset_class = torchvision.datasets.CIFAR10

transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

transform_val = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

root_path = os.path.join(os.path.dirname(__file__), "datasets")
print(root_path)

# Prepare dataset
train_set = dataset_class(root=root_path, train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=8
)

val_set = dataset_class(root=root_path, train=False, download=True, transform=transform_val)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=100, shuffle=False, num_workers=8)
