from torch import nn


class TokenEmbedding(nn.Embedding):
    """Token Embedding module"""

    def __init__(self, vocab_size, d_model):
        """
        Initialize TokenEmbedding module.
        :param vocab_size: size of vocabulary
        :param d_model: dimension of model
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)
