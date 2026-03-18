import torch
from einops import rearrange
from models.helper.helper import create_mask, get_relative_distances
from models.layer.cyclic_shift import CyclicShift
from torch import einsum, nn


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim**-0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(
                create_mask(
                    window_size=window_size,
                    displacement=displacement,
                    upper_lower=True,
                    left_right=False,
                ),
                requires_grad=False,
            )
            self.left_right_mask = nn.Parameter(
                create_mask(
                    window_size=window_size,
                    displacement=displacement,
                    upper_lower=False,
                    left_right=True,
                ),
                requires_grad=False,
            )

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size**2, window_size**2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)

        _, n_h, n_w, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(
                t,
                "b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d",
                h=h,
                w_h=self.window_size,
                w_w=self.window_size,
            ),
            qkv,
        )

        dots = einsum("b h w i d, b h w j d -> b h w i j", q, k) * self.scale

        if self.relative_pos_embedding:
            dots += self.pos_embedding[
                self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]
            ]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1 :: nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum("b h w i j, b h w j d -> b h w i d", attn, v)
        out = rearrange(
            out,
            "b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)",
            h=h,
            w_h=self.window_size,
            w_w=self.window_size,
            nw_h=nw_h,
            nw_w=nw_w,
        )
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out
