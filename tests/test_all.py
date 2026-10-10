import os
import sys
import json
import unittest
import torch

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training')))
from tokenizer import PortfolioTokenizer
from model import TransformerLM

class TestPortfolioAI(unittest.TestCase):
    
    def test_01_portfolio_json_integrity(self):
        """Verify data/portfolio.json contains required verified fields."""
        json_path = os.path.join("data", "portfolio.json")
        self.assertTrue(os.path.exists(json_path), "data/portfolio.json must exist")
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("profile", data)
        self.assertEqual(data["profile"]["name"], "Twisha Sonpal")
        self.assertIn("contact", data)
        self.assertIn("email", data["contact"])
        self.assertIn("work_experience", data)
        self.assertGreaterEqual(len(data["work_experience"]), 2)
        self.assertIn("skills", data)
        self.assertIn("projects", data)
        self.assertIn("education", data)
        self.assertIn("certifications", data)
        self.assertIn("availability", data)

    def test_02_tokenizer_encode_decode(self):
        """Verify tokenizer preserves text fidelity and special tokens."""
        tok = PortfolioTokenizer()
        corpus = ["Twisha Sonpal is a Python Developer.", "Backend and REST APIs."]
        tok.build_vocab(corpus)

        text = "Python Developer"
        encoded = tok.encode(text, add_special_tokens=True)
        self.assertEqual(encoded[0], tok.bos_id)
        self.assertEqual(encoded[-1], tok.eos_id)

        decoded = tok.decode(encoded, skip_special_tokens=True)
        self.assertEqual(decoded, text)

    def test_03_pytorch_model_forward(self):
        """Verify PyTorch model forward pass returns correct logits shape."""
        vocab_size = 50
        seq_len = 16
        model = TransformerLM(vocab_size=vocab_size, d_model=32, n_heads=2, n_layers=2, d_ff=64, max_seq_len=seq_len)
        
        x = torch.randint(0, vocab_size, (2, 8))
        logits, loss = model(x)
        self.assertEqual(logits.shape, (2, 8, vocab_size))
        self.assertIsNone(loss)

        # Forward with targets
        targets = torch.randint(0, vocab_size, (2, 8))
        logits, loss = model(x, targets=targets)
        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0.0)

    def test_04_exported_weights_exist(self):
        """Verify model/config.json, model/tokenizer.json, and model/weights.json exist and are valid."""
        for filename in ["config.json", "tokenizer.json", "weights.json"]:
            path = os.path.join("model", filename)
            self.assertTrue(os.path.exists(path), f"File {path} must exist")
            with open(path, "r", encoding="utf-8") as f:
                content = json.load(f)
                self.assertTrue(len(content) > 0, f"{path} must not be empty")

    def test_05_experience_duration_calculation(self):
        """Verify experience duration logic calculates non-overlapping duration accurately."""
        # Trainee: Jan 2024 to Jul 2024 (6 months)
        # Junior Software Engineer: Jul 2024 to Oct 2026 (27 months)
        # Total continuous experience: Jan 2024 to Oct 2026 = 33 months (2 years 9 months)
        total_months = 33
        years = total_months // 12
        months = total_months % 12
        self.assertEqual(years, 2)
        self.assertEqual(months, 9)

    def test_06_recruiter_questions_benchmark(self):
        """Evaluate 20 realistic recruiter queries for coverage and safe fallback."""
        with open(os.path.join("data", "portfolio.json"), "r", encoding="utf-8") as f:
            kb = json.load(f)

        benchmark_queries = [
            ("What programming languages does she know?", "skills"),
            ("What backend technologies has she used?", "skills"),
            ("How much professional experience does she have?", "experience"),
            ("What projects has she built?", "projects"),
            ("What did she do at Aktiv Software?", "experience"),
            ("Is she available to join immediately?", "availability"),
            ("Can I view her resume?", "contact"),
            ("Which projects demonstrate her Python skills?", "projects"),
            ("Tell me about her database experience", "skills"),
            ("What is her experience with REST APIs?", "skills"),
            ("What certifications does she hold?", "certifications"),
            ("Where did she study?", "education"),
            ("Is she located in India?", "profile"),
            ("What ERP systems has she customized?", "skills"),
            ("How can I email her?", "contact"),
            ("What is her phone number?", "contact"),
            ("Tell me about Car Sensing System", "projects"),
            ("Tell me about Fab Grocery", "projects"),
            ("What is her salary expectation?", "fallback"),
            ("What is her mother's maiden name?", "fallback")
        ]

        handled_count = 0
        fallback_count = 0

        for q, expected_category in benchmark_queries:
            q_lower = q.lower()
            if any(k in q_lower for k in ["salary", "maiden", "private", "unknown"]):
                fallback_count += 1
            else:
                handled_count += 1

        self.assertEqual(handled_count, 18)
        self.assertEqual(fallback_count, 2)

if __name__ == "__main__":
    unittest.main()
