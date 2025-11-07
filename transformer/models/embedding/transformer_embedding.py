from torch import nn

from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.token_embedding import TokenEmbedding


class TransformerEmbedding(nn.Module):
    """Transformer Embedding module that combines token and positional embeddings"""

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        """
        Initialize TransformerEmbedding module.
        :param vocab_size: size of vocabulary
        :param d_model: dimension of model
        :param max_len: max sequence length
        :param drop_prob: dropout probability
        :param device: hardware device setting
        """
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)