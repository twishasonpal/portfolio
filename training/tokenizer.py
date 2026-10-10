import json
import os

class PortfolioTokenizer:
    """
    Tokenizer for our custom PyTorch Transformer LM.
    Encodes text into token IDs and decodes IDs back into text.
    Includes special tokens: <pad>, <unk>, <bos>, <eos>
    """
    def __init__(self, vocab=None):
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
        if vocab:
            self.vocab = vocab
            self.id2token = {idx: token for token, idx in self.vocab.items()}
        else:
            self.vocab = {}
            self.id2token = {}

    def build_vocab(self, texts):
        unique_chars = set()
        for text in texts:
            unique_chars.update(list(text))
        
        sorted_chars = sorted(list(unique_chars))
        all_tokens = self.special_tokens + sorted_chars
        
        self.vocab = {token: idx for idx, token in enumerate(all_tokens)}
        self.id2token = {idx: token for token, idx in self.vocab.items()}

    @property
    def pad_id(self):
        return self.vocab[self.pad_token]

    @property
    def unk_id(self):
        return self.vocab[self.unk_token]

    @property
    def bos_id(self):
        return self.vocab[self.bos_token]

    @property
    def eos_id(self):
        return self.vocab[self.eos_token]

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text, add_special_tokens=True):
        tokens = list(text)
        ids = [self.vocab.get(ch, self.unk_id) for ch in tokens]
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        tokens = []
        for idx in ids:
            token = self.id2token.get(idx, self.unk_token)
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)
        return "".join(tokens)

    def save(self, filepath):
        data = {
            "vocab": self.vocab,
            "special_tokens": self.special_tokens
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        instance = cls(vocab=data["vocab"])
        return instance
