from torch import nn


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed Forward Network module"""

    def __init__(self, d_model, hidden, drop_prob=0.1):
        """
        Initialize PositionwiseFeedForward module.
        :param d_model: dimension of model
        :param hidden: hidden dimension of feed forward network
        :param drop_prob: dropout probability
        """
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
