"""
training/prepare_dataset.py - Compiles continuous text and curated Q&A pairs
into tokenized training chunks (x, y) for next-token prediction.
"""

import os
import json
import torch
from torch.utils.data import Dataset
import sys

# Ensure portfolio root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.tokenizer import PortfolioTokenizer


class PortfolioDataset(Dataset):
    """
    PyTorch Dataset for Next-Token Prediction.
    Yields pairs of (input_ids, target_ids) of fixed sequence length.
    """
    def __init__(self, token_ids_list, seq_len: int = 64, pad_token_id: int = 0):
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.samples = []

        for seq in token_ids_list:
            if len(seq) < 2:
                continue
            # If sequence is shorter than seq_len + 1, pad it
            if len(seq) < seq_len + 1:
                padding = [pad_token_id] * (seq_len + 1 - len(seq))
                # For loss computation, we ignore pad positions with -1
                padded_seq = seq + padding
                x = torch.tensor(padded_seq[:seq_len], dtype=torch.long)
                # Targets: replace padding with -1 so cross_entropy ignores it
                target_raw = seq[1:] + padding
                y_list = [t if idx < len(seq) - 1 else -1 for idx, t in enumerate(target_raw[:seq_len])]
                y = torch.tensor(y_list, dtype=torch.long)
                self.samples.append((x, y))
            else:
                # Sliding window across longer sequences
                for i in range(0, len(seq) - seq_len, seq_len // 2):
                    chunk = seq[i : i + seq_len + 1]
                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:], dtype=torch.long)
                    self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def build_qa_conversations():
    """Returns curated conversational QA pairs covering all portfolio facts."""
    return [
        ("Who is Twisha Sonpal?",
         "Twisha Sonpal is a Python Developer and AI Engineer based in Ahmedabad, Gujarat, India with over 2 years of professional experience building backend systems and AI automation solutions."),
        
        ("What are Twisha's technical skills?",
         "Her technical skills include Python, Flask, PostgreSQL, SQL, REST APIs, AWS deployment, Odoo ERP, and AI automation with n8n, AI agents, and LLM applications."),
        
        ("What is her professional experience?",
         "Twisha works at Aktiv Software in Ahmedabad as a Junior Software Engineer since July 2024, following a 6-month Trainee internship starting in January 2024."),
        
        ("What did she accomplish at Aktiv Software?",
         "She rebuilt and optimized a backend Python module improving performance by 80%, integrated third-party REST APIs, automated recurring manual developer workflows, and optimized PostgreSQL queries."),
        
        ("What AI and LLM projects has she worked on?",
         "She built AI Automation and Intelligent Agent Solutions connecting tools and data sources via MCP server integrations and n8n workflows, alongside IBM GenAI application certifications."),
        
        ("What projects has she developed?",
         "Her key projects include AI Automation and Intelligent Agent Solutions, API Integration Platform, Cloud Deployment on AWS, End-to-End Odoo ERP Implementation, Fab Grocery e-commerce, and a Car Sensing computer vision system."),
        
        ("Is Twisha open to work or available for immediate joining?",
         "Yes, Twisha is actively open to Python Developer, AI Engineer, Backend Engineer, and AI Automation roles in Remote or Ahmedabad locations with negotiable notice period."),
        
        ("What is her educational qualification?",
         "She holds a Bachelor of Engineering in Computer Science from L.J. Institute of Engineering and Technology (2022 to 2025) and a Diploma in Computer Engineering from Gujarat Technological University (2019 to 2022)."),
        
        ("How can I contact Twisha?",
         "You can contact her via email at twishasonpal123@gmail.com, phone at +91-9428970313, LinkedIn at linkedin.com/in/sonpal-twisha-96855a239, or visit her portfolio at twishasonpal.github.io/portfolio."),
        
        ("What certifications does she hold?",
         "She holds verified certifications from AWS (Cloud Practitioner Essentials), IBM (Building Generative AI Applications with Python), ISRO (Geoprocessing with Python), Google, Infosys, and Udemy.")
    ]


def prepare_all_sequences(tokenizer: PortfolioTokenizer):
    """Encodes continuous corpus and conversational QA pairs into token sequences."""
    sequences = []
    
    # 1. Continuous corpus text
    corpus_path = "data/processed/portfolio_corpus.txt"
    if os.path.exists(corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus_text = f.read()
        paragraphs = [p.strip() for p in corpus_text.split("\n\n") if p.strip()]
        for p in paragraphs:
            encoded = tokenizer.encode(p, add_special_tokens=True)
            sequences.append(encoded)

    # 2. QA Conversations (formatted with User: and Assistant: turn markers)
    qa_pairs = build_qa_conversations()
    for question, answer in qa_pairs:
        formatted = f"User: {question} Assistant: {answer}"
        encoded = tokenizer.encode(formatted, add_special_tokens=True)
        # Duplicate QA pairs slightly to balance conversational weight with raw documents
        sequences.append(encoded)
        sequences.append(encoded)

    return sequences


if __name__ == "__main__":
    tokenizer = PortfolioTokenizer.load("data/processed/tokenizer.json")
    seqs = prepare_all_sequences(tokenizer)
    dataset = PortfolioDataset(seqs, seq_len=64, pad_token_id=tokenizer.pad_token_id)
    print(f"Total training sequences generated: {len(dataset)}")
    x, y = dataset[0]
    print(f"Sample Input Shape:  {x.shape}")
    print(f"Sample Target Shape: {y.shape}")
    print("Sample Input Decoded:", tokenizer.decode(x.tolist()[:15]))
