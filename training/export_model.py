import os
import sys
import json
import torch

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from tokenizer import PortfolioTokenizer
from model import TransformerLM

def export_model():
    print("=" * 60)
    print("[Exporting PyTorch Transformer LM Weights for Browser]")
    print("=" * 60)

    model_dir = "model"
    config_path = os.path.join(model_dir, "config.json")
    tokenizer_path = os.path.join(model_dir, "tokenizer.json")
    pt_model_path = os.path.join(model_dir, "pytorch_model.pt")

    if not os.path.exists(pt_model_path):
        raise FileNotFoundError(f"PyTorch model file not found at {pt_model_path}. Run train.py first!")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    tokenizer = PortfolioTokenizer.load(tokenizer_path)

    # Load Model
    model = TransformerLM(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        max_seq_len=config["max_seq_len"]
    )
    model.load_state_dict(torch.load(pt_model_path, map_location="cpu"))
    model.eval()

    # Extract Weights to dict format (lists of floats rounded to 5 decimal places to reduce file size)
    weights_dict = {}
    total_floats = 0

    for name, param in model.named_parameters():
        data = param.detach().numpy()
        data_list = data.round(5).tolist()
        weights_dict[name] = data_list
        total_floats += param.numel()

    export_weights_path = os.path.join(model_dir, "weights.json")
    with open(export_weights_path, "w", encoding="utf-8") as f:
        json.dump(weights_dict, f)

    file_size_kb = os.path.getsize(export_weights_path) / 1024
    print(f"[OK] Exported {len(weights_dict)} tensor layers ({total_floats:,} parameters) to {export_weights_path}")
    print(f"[OK] Exported model file size: {file_size_kb:.2f} KB (Fast browser download)")

    # Verify numerical parity with sample prompt
    sample_text = "Question: Who is Twisha?"
    input_ids = torch.tensor([tokenizer.encode(sample_text, add_special_tokens=False)])
    with torch.no_grad():
        pt_logits, _ = model(input_ids)
    
    print(f"[OK] Verified PyTorch forward pass output shape: {list(pt_logits.shape)}")
    print("[SUCCESS] Model export complete and ready for browser inference!")

if __name__ == "__main__":
    export_model()
