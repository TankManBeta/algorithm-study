import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Optimizer params
lr = 5e-4
warmup_lr = 5e-7
min_lr = 5e-6
eps = 1e-8
betas = (0.9, 0.999)
weight_decay = 0.05
# Training params
n_epochs = 300
warmup_epochs = 20
num_workers = 1
batch_size = 64
# Model params
image_size = 224
patch_size = 4
hidden_dim = 96
layers = (2, 2, 6, 2)
heads = (3, 6, 12, 24)
head_dim = 32
window_size = 7
downscaling_factors = (4, 2, 2, 2)
in_channels = 3
num_classes = 2
# Data augmentation params
color_jitter = 0.4
auto_augment = "rand-m9-mstd0.5-inc1"
interpolation_train = "bicubic"
reprob = 0.25
remode = "pixel"
recount = 1
# Mixup params
mixup = 0.8
mixup_prob = 1.0
mixup_switch_prob = 0.5
mixup_mode = "batch"
# Cutmix params
cutmix = 1.0
cutmix_minmax = None
# Label smoothing params
label_smoothing = 0.1
