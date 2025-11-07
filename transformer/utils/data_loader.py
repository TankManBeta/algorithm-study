import os
import logging
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torchtext.datasets import Multi30k
from torchtext.vocab import build_vocab_from_iterator
from types import SimpleNamespace


SPECIAL_TOKENS = {
    "unk": "<unk>",
    "pad": "<pad>",
    "sos": "<sos>",
    "eos": "<eos>",
}


class _VocabAdapter:
    def __init__(self, vocab):
        self._vocab = vocab
        self.stoi = vocab.get_stoi()
        self.itos = vocab.get_itos()

    def __len__(self):
        return len(self._vocab)


class DataLoader:
    source = None
    target = None

    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token, root: str | None = None):
        self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token
        if root is None:
            root = os.path.dirname(os.path.dirname(__file__))
        self.root = root
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Dataset initializing start")

    def _ext_to_lang_pair(self):
        mapping = {".en": "en", ".de": "de"}
        assert len(self.ext) == 2 and self.ext[0] in mapping and self.ext[1] in mapping
        return mapping[self.ext[0]], mapping[self.ext[1]]

    def make_dataset(self):
        src_lang, tgt_lang = self._ext_to_lang_pair()
        train_data = Multi30k(root=self.root, split="train", language_pair=(src_lang, tgt_lang))
        valid_data = Multi30k(root=self.root, split="valid", language_pair=(src_lang, tgt_lang))
        test_data = Multi30k(root=self.root, split="test", language_pair=(src_lang, tgt_lang))
        return train_data, valid_data, test_data

    def _iter_tokenized(self, dataset_iter, is_src):
        for src_text, tgt_text in dataset_iter:
            if is_src:
                tokens = self.tokenize_en(src_text.lower()) if self.ext[0] == ".en" else self.tokenize_de(src_text.lower())
            else:
                tokens = self.tokenize_en(tgt_text.lower()) if self.ext[1] == ".en" else self.tokenize_de(tgt_text.lower())
            # 与旧 Field 行为一致：在两侧都加入 <sos>/<eos>
            yield [self.init_token] + tokens + [self.eos_token]

    def build_vocab(self, train_data, min_freq):
        # 由于 Multi30k 的迭代器是一次性可迭代对象，需各自重新创建
        train_src_iter = Multi30k(root=self.root, split="train", language_pair=self._ext_to_lang_pair())
        train_tgt_iter = Multi30k(root=self.root, split="train", language_pair=self._ext_to_lang_pair())

        specials = [SPECIAL_TOKENS["unk"], SPECIAL_TOKENS["pad"], SPECIAL_TOKENS["sos"], SPECIAL_TOKENS["eos"]]

        src_vocab = build_vocab_from_iterator(self._iter_tokenized(train_src_iter, is_src=True),
                                              min_freq=min_freq,
                                              specials=specials)
        src_vocab.set_default_index(src_vocab[SPECIAL_TOKENS["unk"]])

        tgt_vocab = build_vocab_from_iterator(self._iter_tokenized(train_tgt_iter, is_src=False),
                                              min_freq=min_freq,
                                              specials=specials)
        tgt_vocab.set_default_index(tgt_vocab[SPECIAL_TOKENS["unk"]])

        # 适配旧接口：提供 .vocab.stoi / .vocab.itos
        self.source = SimpleNamespace(vocab=_VocabAdapter(src_vocab))
        self.target = SimpleNamespace(vocab=_VocabAdapter(tgt_vocab))

    def _numericalize(self, tokens, is_src):
        if is_src:
            vocab = self.source.vocab._vocab
        else:
            vocab = self.target.vocab._vocab
        return [vocab[token] for token in tokens]

    def _collate_fn(self, batch, device):
        src_batch_tokens = []
        tgt_batch_tokens = []

        for src_text, tgt_text in batch:
            src_tokens = self.tokenize_en(src_text.lower()) if self.ext[0] == ".en" else self.tokenize_de(src_text.lower())
            tgt_tokens = self.tokenize_en(tgt_text.lower()) if self.ext[1] == ".en" else self.tokenize_de(tgt_text.lower())

            src_tokens = [self.init_token] + src_tokens + [self.eos_token]
            tgt_tokens = [self.init_token] + tgt_tokens + [self.eos_token]

            src_batch_tokens.append(src_tokens)
            tgt_batch_tokens.append(tgt_tokens)

        pad_idx_src = self.source.vocab.stoi[SPECIAL_TOKENS["pad"]]
        pad_idx_tgt = self.target.vocab.stoi[SPECIAL_TOKENS["pad"]]

        src_numerical = [self._numericalize(toks, is_src=True) for toks in src_batch_tokens]
        tgt_numerical = [self._numericalize(toks, is_src=False) for toks in tgt_batch_tokens]

        src_max_len = max(len(x) for x in src_numerical)
        tgt_max_len = max(len(x) for x in tgt_numerical)

        src_padded = [x + [pad_idx_src] * (src_max_len - len(x)) for x in src_numerical]
        tgt_padded = [x + [pad_idx_tgt] * (tgt_max_len - len(x)) for x in tgt_numerical]

        src_tensor = torch.tensor(src_padded, dtype=torch.long, device=device)
        tgt_tensor = torch.tensor(tgt_padded, dtype=torch.long, device=device)

        return SimpleNamespace(src=src_tensor, trg=tgt_tensor)

    def make_iter(self, train, validate, test, batch_size, device):
        collate = lambda batch: self._collate_fn(batch, device=device)
        train_iterator = TorchDataLoader(list(train), batch_size=batch_size, shuffle=True, collate_fn=collate)
        valid_iterator = TorchDataLoader(list(validate), batch_size=batch_size, shuffle=False, collate_fn=collate)
        test_iterator = TorchDataLoader(list(test), batch_size=batch_size, shuffle=False, collate_fn=collate)
        self.logger.info("Dataset initializing done")
        return train_iterator, valid_iterator, test_iterator

    def _ids_to_sentence(self, ids, itos):
        tokens = [itos[i] for i in ids]
        start = tokens.index(SPECIAL_TOKENS["sos"]) + 1 if SPECIAL_TOKENS["sos"] in tokens else 0
        end = tokens.index(SPECIAL_TOKENS["eos"]) if SPECIAL_TOKENS["eos"] in tokens else len(tokens)
        return ' '.join(tokens[start:end])

    def preview_iterators(self, train_iterator, valid_iterator, num_examples=2):
        # train preview
        try:
            tb = next(iter(train_iterator))
            self.logger.info("[Preview][train] src shape: %s trg shape: %s", tuple(tb.src.shape), tuple(tb.trg.shape))
            for j in range(min(num_examples, tb.src.size(0))):
                src_sent = self._ids_to_sentence(tb.src[j].tolist(), self.source.vocab.itos)
                trg_sent = self._ids_to_sentence(tb.trg[j].tolist(), self.target.vocab.itos)
                self.logger.info("[train ex %s] SRC: %s", j, src_sent)
                self.logger.info("[train ex %s] TRG: %s", j, trg_sent)
        except Exception as e:
            self.logger.exception("[Preview][train] failed: %s", e)

        # valid preview
        try:
            vb = next(iter(valid_iterator))
            self.logger.info("[Preview][valid] src shape: %s trg shape: %s", tuple(vb.src.shape), tuple(vb.trg.shape))
            for j in range(min(num_examples, vb.src.size(0))):
                src_sent = self._ids_to_sentence(vb.src[j].tolist(), self.source.vocab.itos)
                trg_sent = self._ids_to_sentence(vb.trg[j].tolist(), self.target.vocab.itos)
                self.logger.info("[valid ex %s] SRC: %s", j, src_sent)
                self.logger.info("[valid ex %s] TRG: %s", j, trg_sent)
        except Exception as e:
            self.logger.exception("[Preview][valid] failed: %s", e)