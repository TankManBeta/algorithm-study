import os

import torch
import torchvision
from conf import (
    auto_augment,
    batch_size,
    color_jitter,
    image_size,
    interpolation_train,
    num_workers,
    recount,
    remode,
    reprob,
)
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms

dataset_class = torchvision.datasets.ImageFolder

root_path = os.path.join(os.path.dirname(__file__), "datasets")
print(root_path)

# Training transform
resize_im = image_size > 32
transform_train = create_transform(
    input_size=image_size,
    is_training=True,
    color_jitter=color_jitter,
    auto_augment=auto_augment,
    interpolation=interpolation_train,
    re_prob=reprob,
    re_mode=remode,
    re_count=recount,
)
if not resize_im:
    transform_train.transforms[0] = transforms.RandomCrop(image_size, padding=4)

# Validation transform
t = []
if resize_im:
    size = int((256 / 224) * image_size)
    t.append(transforms.Resize(size, interpolation=3))
    t.append(transforms.CenterCrop(image_size))
t.append(transforms.ToTensor())
t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
transform_val = transforms.Compose(t)

# Dataset
dataset_train = dataset_class(root=os.path.join(root_path, "train"), transform=transform_train)
dataset_val = dataset_class(root=os.path.join(root_path, "val"), transform=transform_val)

# Sampler
sampler_train = torch.utils.data.RandomSampler(dataset_train)
sampler_val = torch.utils.data.SequentialSampler(dataset_val)

# DataLoader
dataloader_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=batch_size,
    sampler=sampler_train,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True,
)

dataloader_val = torch.utils.data.DataLoader(
    dataset_val,
    batch_size=batch_size,
    sampler=sampler_val,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False,
)
