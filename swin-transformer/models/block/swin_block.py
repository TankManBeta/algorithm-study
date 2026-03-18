from models.layer.feed_forward import FeedForward
from models.layer.pre_norm import PreNorm
from models.layer.residual import Residual
from models.layer.window_attention import WindowAttention
from torch import nn


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual(
            PreNorm(
                dim,
                WindowAttention(
                    dim=dim,
                    heads=heads,
                    head_dim=head_dim,
                    shifted=shifted,
                    window_size=window_size,
                    relative_pos_embedding=relative_pos_embedding,
                ),
            )
        )
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x
