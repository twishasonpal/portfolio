"""
model/tokenizer.py - Custom Vocabulary and Tokenizer built from scratch.
Implements tokenization, special token handling, vocabulary construction,
saving, loading, and encoding/decoding for the Portfolio LLM.
"""

import re
import json
from typing import List, Dict, Optional


class PortfolioTokenizer:
    """
    Subword & WordPiece Tokenizer for Portfolio LLM.
    Splits text on punctuation and whitespace, tracks subword pieces,
    and maps them to discrete integer token IDs.
    """
    
    # Core special tokens
    PAD_TOKEN = "<|pad|>"  # Padding for batching (ID: 0)
    UNK_TOKEN = "<|unk|>"  # Unknown token (ID: 1)
    BOS_TOKEN = "<|bos|>"  # Beginning of sequence (ID: 2)
    EOS_TOKEN = "<|eos|>"  # End of sequence (ID: 3)

    def __init__(self):
        self.special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN
        ]
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._init_special_tokens()

    def _init_special_tokens(self):
        """Register the core special tokens with fixed IDs 0, 1, 2, 3."""
        self.token_to_id.clear()
        self.id_to_token.clear()
        for idx, token in enumerate(self.special_tokens):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.PAD_TOKEN]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id[self.UNK_TOKEN]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id[self.BOS_TOKEN]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id[self.EOS_TOKEN]

    @property
    def vocab_size(self) -> int:
        """Returns total number of unique tokens in vocabulary."""
        return len(self.token_to_id)

    def _pre_tokenize(self, text: str) -> List[str]:
        """
        Splits text into words, punctuation, and symbols.
        Example: "Twisha's Python (v3)!" -> ['Twisha', "'", 's', 'Python', '(', 'v3', ')', '!']
        """
        # Regex matches alphanumeric sequences or individual punctuation/symbol characters
        pattern = r"\w+|[^\w\s]"
        return re.findall(pattern, text)

    def build_vocab(self, texts: List[str], min_freq: int = 1):
        """
        Scans a list of text documents, counts word frequencies,
        and constructs the integer-to-token mappings.
        """
        self._init_special_tokens()
        freqs: Dict[str, int] = {}

        for text in texts:
            tokens = self._pre_tokenize(text)
            for tok in tokens:
                freqs[tok] = freqs.get(tok, 0) + 1

        # Sort by frequency descending for clean vocabulary IDs
        sorted_tokens = sorted(freqs.items(), key=lambda x: x[1], reverse=True)

        current_id = len(self.token_to_id)
        for tok, count in sorted_tokens:
            if count >= min_freq and tok not in self.token_to_id:
                self.token_to_id[tok] = current_id
                self.id_to_token[current_id] = tok
                current_id += 1

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Converts human text into a list of integer token IDs.
        Input:  "Who is Twisha?"
        Output: [2, 15, 23, 42, 8, 3]  (with BOS and EOS)
        """
        raw_tokens = self._pre_tokenize(text)
        token_ids: List[int] = []

        if add_special_tokens:
            token_ids.append(self.bos_token_id)

        for tok in raw_tokens:
            if tok in self.token_to_id:
                token_ids.append(self.token_to_id[tok])
            else:
                token_ids.append(self.unk_token_id)

        if add_special_tokens:
            token_ids.append(self.eos_token_id)

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Converts integer token IDs back into human readable text.
        Input:  [2, 15, 23, 42, 8, 3]
        Output: "Who is Twisha ?"
        """
        tokens: List[str] = []
        for tid in token_ids:
            if tid not in self.id_to_token:
                tok = self.UNK_TOKEN
            else:
                tok = self.id_to_token[tid]

            if skip_special_tokens and tok in self.special_tokens:
                continue

            tokens.append(tok)

        # Reconstruct clean text spacing
        text = " ".join(tokens)
        # Detach contractions and standard punctuation spacing
        text = re.sub(r"\s+([.,!?;:\'])", r"\1", text)
        text = re.sub(r"([(\[{])\s+", r"\1", text)
        text = re.sub(r"\s+([)\]}])", r"\1", text)
        return text.strip()

    def save(self, filepath: str):
        """Saves the vocabulary to a JSON file."""
        data = {
            "special_tokens": self.special_tokens,
            "token_to_id": self.token_to_id,
            "vocab_size": self.vocab_size
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> "PortfolioTokenizer":
        """Loads a saved vocabulary JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls()
        tokenizer.special_tokens = data.get("special_tokens", tokenizer.special_tokens)
        tokenizer.token_to_id = data["token_to_id"]
        tokenizer.id_to_token = {int(v): k for k, v in tokenizer.token_to_id.items()}
        return tokenizer
