import torch
from torch import nn

from models.block.decoder_layer import DecoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Decoder(nn.Module):
    """Transformer Decoder module"""
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        """
        Initialize Transformer Decoder module.
        :param dec_voc_size: vocabulary size of decoder
        :param max_len: maximum length of input sequences
        :param d_model: dimension of model
        :param ffn_hidden: hidden dimension of feed forward network
        :param n_head: number of attention heads
        :param n_layers: number of decoder layers
        :param drop_prob: dropout probability
        :param device: device to run the model on
        """
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        vocab_size=dec_voc_size,
                                        device=device)

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        """
        Forward pass for decoder.
        :param trg: target sequences [batch_size, trg_len]
        :param enc_src: encoded source sequences [batch_size, src_len, d_model]
        :param trg_mask: target mask [batch_size, 1, trg_len, trg_len]
        :param src_mask: source mask [batch_size, 1, 1, src_len]
        :return: output logits [batch_size, trg_len, dec_voc_size]
        """
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        # pass to LM head
        output = self.linear(trg)
        return output