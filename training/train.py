"""
training/train.py - Training script for the Portfolio Language Model.
Uses:
- PyTorch CPU operations
- AdamW optimizer with cosine learning rate schedule
- Cross-Entropy Loss ignoring padding (-1)
- Validation loss evaluation & checkpoint saving
"""

import os
import sys
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.tokenizer import PortfolioTokenizer
from model.transformer import PortfolioLanguageModel
from training.prepare_dataset import PortfolioDataset, prepare_all_sequences


def train(
    epochs: int = 40,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    seq_len: int = 64,
    eval_interval: int = 5,
    checkpoint_dir: str = "checkpoints"
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cpu")
    print(f"=== Starting Training on Device: {device} ===")

    # 1. Load Tokenizer & Vocabulary
    tokenizer_path = "data/processed/tokenizer.json"
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Missing {tokenizer_path}. Run prepare_data.py first.")
    tokenizer = PortfolioTokenizer.load(tokenizer_path)
    vocab_size = tokenizer.vocab_size
    print(f"Loaded Tokenizer: Vocabulary Size = {vocab_size} tokens")

    # 2. Build Dataset
    sequences = prepare_all_sequences(tokenizer)
    dataset = PortfolioDataset(sequences, seq_len=seq_len, pad_token_id=tokenizer.pad_token_id)
    
    # Train / Val Split (90% train, 10% validation)
    val_size = max(2, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Dataset: {len(train_dataset)} training chunks, {len(val_dataset)} validation chunks")

    # 3. Initialize Model
    model = PortfolioLanguageModel(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=seq_len,
        dropout=0.1
    ).to(device)

    print(f"Model Parameters: {model.count_parameters():,}")

    # 4. Optimizer: AdamW
    # Separate parameters that should have weight decay from LayerNorm/biases
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95))

    # 5. Training Loop
    best_val_loss = float("inf")
    start_time = time.time()

    print("\n--- Training Progress ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        num_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            # Forward pass (computes cross entropy loss automatically)
            logits, loss = model(x, targets=y)
            
            # Backward pass (computes gradients dLoss/dW)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step (updates weights: W = W - lr * grad)
            optimizer.step()

            total_train_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_train_loss / max(1, num_batches)

        # Periodic Validation Evaluation
        if epoch % eval_interval == 0 or epoch == epochs:
            model.eval()
            total_val_loss = 0.0
            num_val_batches = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    _, vloss = model(vx, targets=vy)
                    total_val_loss += vloss.item()
                    num_val_batches += 1
            avg_val_loss = total_val_loss / max(1, num_val_batches)

            # Checkpoint the best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_path = os.path.join(checkpoint_dir, "portfolio_model.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "vocab_size": vocab_size,
                    "d_model": 128,
                    "n_heads": 4,
                    "n_layers": 4,
                    "d_ff": 512,
                    "max_seq_len": seq_len
                }, checkpoint_path)
                star = " [BEST CHECKPOINT]"
            else:
                star = ""

            elapsed = time.time() - start_time
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {elapsed:.1f}s{star}")

    print(f"\nTraining Complete in {time.time() - start_time:.1f}s!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    return checkpoint_path


if __name__ == "__main__":
    train(epochs=35, batch_size=4, learning_rate=4e-4)
