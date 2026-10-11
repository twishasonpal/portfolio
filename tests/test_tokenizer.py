import unittest
import os
import sys
sys.path.append('.')
from model.tokenizer import PortfolioTokenizer

class TestPortfolioTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = PortfolioTokenizer()
        self.corpus = [
            "Twisha Sonpal is a Python Developer and AI Engineer.",
            "She builds REST APIs, PostgreSQL databases, and AWS applications."
        ]
        self.tokenizer.build_vocab(self.corpus)

    def test_special_tokens(self):
        self.assertEqual(self.tokenizer.pad_token_id, 0)
        self.assertEqual(self.tokenizer.unk_token_id, 1)
        self.assertEqual(self.tokenizer.bos_token_id, 2)
        self.assertEqual(self.tokenizer.eos_token_id, 3)

    def test_encode_decode_roundtrip(self):
        text = "Twisha builds Python APIs."
        encoded = self.tokenizer.encode(text)
        # First must be BOS (2), last must be EOS (3)
        self.assertEqual(encoded[0], self.tokenizer.bos_token_id)
        self.assertEqual(encoded[-1], self.tokenizer.eos_token_id)
        
        decoded = self.tokenizer.decode(encoded)
        self.assertIn("Twisha", decoded)
        self.assertIn("Python", decoded)

    def test_unknown_token_handling(self):
        # A completely unknown word
        encoded = self.tokenizer.encode("Xylophone", add_special_tokens=False)
        self.assertEqual(encoded, [self.tokenizer.unk_token_id])

    def test_save_and_load(self):
        filepath = "data/processed/test_tokenizer.json"
        self.tokenizer.save(filepath)
        loaded = PortfolioTokenizer.load(filepath)
        self.assertEqual(loaded.vocab_size, self.tokenizer.vocab_size)
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    unittest.main()
