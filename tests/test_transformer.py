import unittest
import torch
import sys
sys.path.append('.')
from model.transformer import (
    CausalSelfAttention,
    FeedForward,
    TransformerBlock,
    PortfolioLanguageModel
)

class TestTransformerArchitecture(unittest.TestCase):
    def setUp(self):
        self.d_model = 64
        self.n_heads = 4
        self.max_seq_len = 32
        self.vocab_size = 50

    def test_causal_self_attention_shape(self):
        attn = CausalSelfAttention(d_model=self.d_model, n_heads=self.n_heads, max_seq_len=self.max_seq_len)
        x = torch.randn(2, 8, self.d_model) # (batch=2, seq=8, d_model=64)
        out = attn(x)
        self.assertEqual(out.shape, x.shape)

    def test_causal_masking_prevents_future_peeking(self):
        """Verify that position 0 output is strictly identical regardless of future tokens."""
        attn = CausalSelfAttention(d_model=self.d_model, n_heads=self.n_heads, max_seq_len=self.max_seq_len, dropout=0.0)
        attn.eval()
        
        # Sequence A of length 3
        seq_a = torch.randn(1, 3, self.d_model)
        # Sequence B has the exact same first token, but different future tokens
        seq_b = seq_a.clone()
        seq_b[:, 1:, :] = torch.randn(1, 2, self.d_model)
        
        out_a = attn(seq_a)
        out_b = attn(seq_b)
        
        # Token 0 output must be identical because it cannot attend to future tokens
        diff = (out_a[:, 0, :] - out_b[:, 0, :]).abs().max().item()
        self.assertAlmostEqual(diff, 0.0, places=5)

    def test_feed_forward_shape(self):
        ffn = FeedForward(d_model=self.d_model, d_ff=128)
        x = torch.randn(2, 8, self.d_model)
        out = ffn(x)
        self.assertEqual(out.shape, x.shape)

    def test_full_model_forward_and_loss(self):
        model = PortfolioLanguageModel(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=2,
            max_seq_len=self.max_seq_len
        )
        tokens = torch.randint(0, self.vocab_size, (2, 10))
        targets = torch.randint(0, self.vocab_size, (2, 10))
        
        logits, loss = model(tokens, targets=targets)
        self.assertEqual(logits.shape, (2, 10, self.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(loss.item() > 0)

    def test_autoregressive_generation(self):
        model = PortfolioLanguageModel(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=2,
            max_seq_len=self.max_seq_len
        )
        prompt = torch.tensor([[1, 2, 3]]) # Prompt of length 3
        generated = model.generate(prompt, max_new_tokens=5)
        # Prompt length 3 + 5 generated tokens = 8 tokens
        self.assertEqual(generated.shape, (1, 8))

if __name__ == '__main__':
    unittest.main()
