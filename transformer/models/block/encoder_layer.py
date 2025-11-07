from torch import nn

from models.layer.layer_norm import LayerNorm
from models.layer.multi_head_attention import MultiHeadAttention
from models.layer.position_wise_feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """Encoder layer with self-attention and feed forward network"""
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        """
        Initialize EncoderLayer module.
        :param d_model: dimension of model
        :param ffn_hidden: hidden dimension of feed forward network
        :param n_head: number of attention heads
        :param drop_prob: dropout probability
        """
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        """
        Forward pass for encoder layer.
        :param x: encoder input [batch_size, src_len, d_model]
        :param src_mask: source mask [batch_size, 1, 1, src_len]
        :return: output of encoder layer [batch_size, src_len, d_model]
        """
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
      
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x