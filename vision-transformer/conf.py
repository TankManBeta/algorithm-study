import torch

lr = 1e-4
n_epochs = 200
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 512
image_size = 32
patch_size = 4
dim_head = 512
depth = 6
heads = 8
mlp_dim = 512
dropout = 0.1
emb_dropout = 0.1
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
num_classes = 10
