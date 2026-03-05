import logging
import os
from collections import Counter
from types import SimpleNamespace

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader as TorchDataLoader

SPECIAL_TOKENS = {
    "unk": "<unk>",
    "pad": "<pad>",
    "sos": "<sos>",
    "eos": "<eos>",
}


class Vocab:
    def __init__(self, counter, min_freq=1, specials=None):
        if specials is None:
            specials = []
        self.itos = list(specials)
        self.stoi = {tok: i for i, tok in enumerate(specials)}
        for tok, freq in counter.items():
            if freq >= min_freq and tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)
        self.unk_index = self.stoi[specials[0]] if specials else None

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, token):
        return self.stoi.get(token, self.unk_index)

    def get_stoi(self):
        return self.stoi

    def get_itos(self):
        return self.itos[:]

    def set_default_index(self, index):
        self.unk_index = index


def build_vocab_from_iterator(token_iterator, min_freq, specials):
    counter = Counter()
    for tokens in token_iterator:
        counter.update(tokens)
    return Vocab(counter, min_freq=min_freq, specials=specials)


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

    def __init__(
        self, ext, tokenize_en, tokenize_de, init_token, eos_token, root: str | None = None
    ):
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
        hf_data = load_dataset("bentrevett/multi30k")
        train_data = hf_data["train"]
        valid_data = hf_data["validation"]
        test_data = hf_data["test"]
        return train_data, valid_data, test_data

    def _iter_tokenized(self, dataset, is_src):
        src_lang, tgt_lang = self._ext_to_lang_pair()
        lang = src_lang if is_src else tgt_lang
        tokenize = self.tokenize_en if lang == "en" else self.tokenize_de
        for example in dataset:
            text = example[lang].lower()
            tokens = tokenize(text)
            yield [self.init_token] + tokens + [self.eos_token]

    def build_vocab(self, train_data, min_freq):
        specials = [
            SPECIAL_TOKENS["unk"],
            SPECIAL_TOKENS["pad"],
            SPECIAL_TOKENS["sos"],
            SPECIAL_TOKENS["eos"],
        ]
        src_vocab = build_vocab_from_iterator(
            self._iter_tokenized(train_data, is_src=True), min_freq=min_freq, specials=specials
        )
        src_vocab.set_default_index(src_vocab[SPECIAL_TOKENS["unk"]])
        tgt_vocab = build_vocab_from_iterator(
            self._iter_tokenized(train_data, is_src=False), min_freq=min_freq, specials=specials
        )
        tgt_vocab.set_default_index(tgt_vocab[SPECIAL_TOKENS["unk"]])
        self.source = SimpleNamespace(vocab=_VocabAdapter(src_vocab))
        self.target = SimpleNamespace(vocab=_VocabAdapter(tgt_vocab))

    def _numericalize(self, tokens, is_src):
        if is_src:
            vocab = self.source.vocab._vocab
        else:
            vocab = self.target.vocab._vocab
        return [vocab[token] for token in tokens]

    def _collate_fn(self, batch, device):
        src_lang, tgt_lang = self._ext_to_lang_pair()
        src_batch_tokens = []
        tgt_batch_tokens = []
        for example in batch:
            src_text = example[src_lang].lower()
            tgt_text = example[tgt_lang].lower()
            src_tokens = (
                self.tokenize_en(src_text) if src_lang == "en" else self.tokenize_de(src_text)
            )
            tgt_tokens = (
                self.tokenize_en(tgt_text) if tgt_lang == "en" else self.tokenize_de(tgt_text)
            )
            src_batch_tokens.append([self.init_token] + src_tokens + [self.eos_token])
            tgt_batch_tokens.append([self.init_token] + tgt_tokens + [self.eos_token])

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
        def collate(batch):
            return self._collate_fn(batch, device=device)

        train_iterator = TorchDataLoader(
            train, batch_size=batch_size, shuffle=True, collate_fn=collate
        )
        valid_iterator = TorchDataLoader(
            validate, batch_size=batch_size, shuffle=False, collate_fn=collate
        )
        test_iterator = TorchDataLoader(
            test, batch_size=batch_size, shuffle=False, collate_fn=collate
        )
        self.logger.info("Dataset initializing done")
        return train_iterator, valid_iterator, test_iterator

    def _ids_to_sentence(self, ids, itos):
        tokens = [itos[i] for i in ids]
        start = tokens.index(SPECIAL_TOKENS["sos"]) + 1 if SPECIAL_TOKENS["sos"] in tokens else 0
        end = (
            tokens.index(SPECIAL_TOKENS["eos"]) if SPECIAL_TOKENS["eos"] in tokens else len(tokens)
        )
        return " ".join(tokens[start:end])

    def preview_iterators(self, train_iterator, valid_iterator, num_examples=2):
        try:
            tb = next(iter(train_iterator))
            self.logger.info(
                "[Preview][train] src shape: %s trg shape: %s",
                tuple(tb.src.shape),
                tuple(tb.trg.shape),
            )
            for j in range(min(num_examples, tb.src.size(0))):
                src_sent = self._ids_to_sentence(tb.src[j].tolist(), self.source.vocab.itos)
                trg_sent = self._ids_to_sentence(tb.trg[j].tolist(), self.target.vocab.itos)
                self.logger.info("[train ex %s] SRC: %s", j, src_sent)
                self.logger.info("[train ex %s] TRG: %s", j, trg_sent)
        except Exception as e:
            self.logger.exception("[Preview][train] failed: %s", e)

        try:
            vb = next(iter(valid_iterator))
            self.logger.info(
                "[Preview][valid] src shape: %s trg shape: %s",
                tuple(vb.src.shape),
                tuple(vb.trg.shape),
            )
            for j in range(min(num_examples, vb.src.size(0))):
                src_sent = self._ids_to_sentence(vb.src[j].tolist(), self.source.vocab.itos)
                trg_sent = self._ids_to_sentence(vb.trg[j].tolist(), self.target.vocab.itos)
                self.logger.info("[valid ex %s] SRC: %s", j, src_sent)
                self.logger.info("[valid ex %s] TRG: %s", j, trg_sent)
        except Exception as e:
            self.logger.exception("[Preview][valid] failed: %s", e)
