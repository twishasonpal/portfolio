import os
import sys
import json
import time
import torch
from torch.utils.data import DataLoader, random_split

# Force UTF-8 on Windows stdout if possible
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from tokenizer import PortfolioTokenizer
from model import TransformerLM
from dataset import load_corpus_from_json, PortfolioDataset

def train():
    print("=" * 60)
    print("[PyTorch Transformer LM Training from Scratch]")
    print("=" * 60)

    # 1. Paths
    json_path = os.path.join("data", "portfolio.json")
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)

    # 2. Load Corpus
    print(f"📂 Loading corpus from {json_path}...")
    corpus_sentences = load_corpus_from_json(json_path)
    print(f"✓ Loaded {len(corpus_sentences)} sentences / QA pairs.")

    # 3. Tokenizer
    tokenizer = PortfolioTokenizer()
    tokenizer.build_vocab(corpus_sentences)
    tokenizer_save_path = os.path.join(model_dir, "tokenizer.json")
    tokenizer.save(tokenizer_save_path)
    print(f"✓ Tokenizer built and saved to {tokenizer_save_path} (Vocab size: {tokenizer.vocab_size})")

    # 4. Dataset & DataLoader
    seq_len = 64
    dataset = PortfolioDataset(corpus_sentences, tokenizer, seq_len=seq_len)
    
    # Train/Val split (90/10)
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    batch_size = 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"📊 Dataset split: {train_size} training samples, {val_size} validation samples.")

    # 5. Model Configuration
    config = {
        "vocab_size": tokenizer.vocab_size,
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 3,
        "d_ff": 128,
        "max_seq_len": seq_len
    }

    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"✓ Model config saved to {config_path}")

    # 6. Instantiate Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Device set to: {device}")

    model = TransformerLM(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        max_seq_len=config["max_seq_len"]
    ).to(device)

    # Estimate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Total Model Parameters: {total_params:,} (Lightweight & Browser-Optimized)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    # 7. Training Loop
    epochs = 40
    start_time = time.time()

    print("\n[Training Loop Started...]")
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            _, loss = model(x_batch, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                _, val_loss = model(x_batch, y_batch)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Elapsed: {elapsed:.2f}s")

    # 8. Save PyTorch Model
    pt_model_path = os.path.join(model_dir, "pytorch_model.pt")
    torch.save(model.state_dict(), pt_model_path)
    print(f"\n[Saved] PyTorch model checkpoint saved to {pt_model_path}")

    # 9. Test Text Generation
    print("\n[Testing Text Generation with Prompt: 'Question: Who is Twisha?']")
    prompt_ids = torch.tensor([tokenizer.encode("Question: Who is Twisha?", add_special_tokens=False)], device=device)
    gen_ids = model.generate(prompt_ids, max_new_tokens=40, temperature=0.7, top_k=5)
    gen_text = tokenizer.decode(gen_ids[0].tolist(), skip_special_tokens=True)
    print(f"Generated output:\n{gen_text}\n")

if __name__ == "__main__":
    train()
