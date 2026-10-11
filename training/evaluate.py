"""
training/evaluate.py - Stage 5: Evaluate the trained Portfolio Language Model.

This script:
  1. Loads the tokenizer from data/processed/tokenizer.json
  2. Loads the best checkpoint from checkpoints/portfolio_model.pt
  3. Reconstructs PortfolioLanguageModel with saved hyperparameters
  4. Runs autoregressive text generation on portfolio questions
  5. Tests two temperature settings:
       temperature=0.3  (focused / greedy-ish)
       temperature=0.8  (creative / more diverse)
  6. Prints ACTUAL model output - results are NOT fabricated or improved.

Honest warning:
  The model was trained on only 35 samples for 35 epochs.
  It can memorize fragments from the training corpus but will
  NOT generalize like a large pretrained model. Outputs will be
  partial sentences, sometimes repetitive or incomplete.
  This is expected and is a direct consequence of the tiny dataset.
"""

import os
import sys
import torch

# Ensure project root is in sys.path so 'model' package is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.tokenizer import PortfolioTokenizer
from model.transformer import PortfolioLanguageModel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "tokenizer.json")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "portfolio_model.pt")

# Questions to evaluate
PROMPTS = [
    "User: Who is Twisha Sonpal? Assistant:",
    "User: What are her technical skills? Assistant:",
    "User: What projects has she worked on? Assistant:",
    "User: Is she available to join immediately? Assistant:",
    "User: How can I contact Twisha? Assistant:",
    "User: What certifications does she hold? Assistant:",
]

MAX_NEW_TOKENS = 40   # tokens to generate after the prompt
TOP_K = 10            # top-k sampling filter

TEMPERATURES = [
    (0.3, "focused  (temperature=0.3)"),
    (0.8, "creative (temperature=0.8)"),
]


# ---------------------------------------------------------------------------
# Helper: Load model from checkpoint
# ---------------------------------------------------------------------------

def load_model_and_tokenizer():
    # ---- Tokenizer ----
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(
            f"Tokenizer not found at: {TOKENIZER_PATH}\n"
            "Run training/prepare_data.py first."
        )
    tokenizer = PortfolioTokenizer.load(TOKENIZER_PATH)
    print(f"[OK] Tokenizer loaded  | vocab_size = {tokenizer.vocab_size}")

    # ---- Checkpoint ----
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found at: {CHECKPOINT_PATH}\n"
            "Run training/train.py first."
        )
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    print(f"[OK] Checkpoint loaded | saved at epoch {checkpoint['epoch']}")
    print(f"     Train loss : {checkpoint['train_loss']:.4f}")
    print(f"     Val   loss : {checkpoint['val_loss']:.4f}")

    # ---- Reconstruct model with saved hyperparameters ----
    vocab_size  = checkpoint["vocab_size"]
    d_model     = checkpoint["d_model"]
    n_heads     = checkpoint["n_heads"]
    n_layers    = checkpoint["n_layers"]
    d_ff        = checkpoint["d_ff"]
    max_seq_len = checkpoint["max_seq_len"]

    model = PortfolioLanguageModel(
        vocab_size  = vocab_size,
        d_model     = d_model,
        n_heads     = n_heads,
        n_layers    = n_layers,
        d_ff        = d_ff,
        max_seq_len = max_seq_len,
        dropout     = 0.0   # disable dropout at inference time
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"     Architecture : d_model={d_model}, n_heads={n_heads}, "
          f"n_layers={n_layers}, d_ff={d_ff}, max_seq_len={max_seq_len}")
    print(f"     Parameters   : {model.count_parameters():,}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Helper: Generate text for one prompt
# ---------------------------------------------------------------------------

def generate_answer(model, tokenizer, prompt_text, temperature, max_new_tokens):
    """
    Encode the prompt, run model.generate(), decode and return only the
    newly generated tokens (the part after the prompt).
    """
    # Encode prompt to token IDs
    input_ids_list = tokenizer.encode(prompt_text)
    prompt_len = len(input_ids_list)

    # If prompt itself is too long for the model context, truncate from left
    max_seq = model.max_seq_len
    if prompt_len >= max_seq:
        input_ids_list = input_ids_list[-(max_seq - 1):]
        prompt_len = len(input_ids_list)

    input_ids = torch.tensor([input_ids_list], dtype=torch.long)  # (1, prompt_len)

    # Autoregressive generation
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=TOP_K
    )

    # Decode only the newly generated portion (after the prompt)
    generated_ids = output_ids[0, prompt_len:].tolist()
    generated_text = tokenizer.decode(generated_ids)
    return generated_text.strip()


# ---------------------------------------------------------------------------
# Helper: Print a separator line
# ---------------------------------------------------------------------------

def sep(char="-", width=70):
    print(char * width)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate():
    print()
    sep("=")
    print("  Stage 5: Portfolio Language Model - Evaluation")
    sep("=")
    print()
    print("Loading model and tokenizer ...")
    print()

    model, tokenizer = load_model_and_tokenizer()

    print()
    sep("=")
    print("  HONEST EXPECTATION")
    sep("=")
    print(
        "  This model was trained from scratch on only 35 samples for 35 epochs.\n"
        "  It has learned some vocabulary patterns from the portfolio corpus but\n"
        "  does NOT have the knowledge of a large pretrained model.\n"
        "  Outputs may be:\n"
        "    - Partial sentences that start correctly but trail off\n"
        "    - Repetitions of words seen in training\n"
        "    - Occasionally coherent short phrases about Twisha\n"
        "  This is expected. Stage 6 will improve answer quality via retrieval."
    )
    print()

    # Iterate over every question x every temperature
    for q_idx, prompt in enumerate(PROMPTS, start=1):
        sep("=")
        print(f"  Q{q_idx}: {prompt}")
        sep("=")

        for temperature, temp_label in TEMPERATURES:
            print(f"\n  [{temp_label}]")
            sep()
            answer = generate_answer(model, tokenizer, prompt, temperature, MAX_NEW_TOKENS)
            # Show the full response including prompt so it is easy to read
            print(f"  PROMPT  : {prompt}")
            print(f"  ANSWER  : {answer if answer else '[empty - all tokens were unknown or padding]'}")
            sep()

        print()

    # ---------------------------------------------------------------------------
    # Summary: what the model does and does not do well
    # ---------------------------------------------------------------------------
    sep("=")
    print("  EVALUATION SUMMARY")
    sep("=")
    print(
        "  What the model has learned (from the 35-sample training corpus):\n"
        "    - Basic word co-occurrences in portfolio-style text\n"
        "    - Short phrases like 'Python Developer', 'Twisha Sonpal',\n"
        "      'REST API', 'available', 'contact', etc.\n"
        "    - Very rough question -> fragment associations\n"
        "\n"
        "  Where the model struggles:\n"
        "    - Cannot construct full grammatically correct sentences reliably\n"
        "    - May repeat tokens or produce unrelated fragments\n"
        "    - Low generalization to paraphrased or unseen questions\n"
        "    - No understanding of meaning - pure statistical next-token prediction\n"
        "\n"
        "  Root cause:\n"
        "    - Dataset: only 35 training samples (the corpus is ~4,000 characters)\n"
        "    - A well-functioning language model typically needs millions of tokens\n"
        "    - With only 364 vocabulary tokens the model can only recombine\n"
        "      words that appeared in the training text\n"
        "\n"
        "  Next step (Stage 6 - requires your approval):\n"
        "    - FastAPI backend that wraps this model\n"
        "    - Retrieval-Augmented Generation (RAG): fetch the most relevant\n"
        "      portfolio facts and prepend them to the model's context,\n"
        "      so answers stay grounded and coherent\n"
        "    - Connect to the existing chatbot.js frontend\n"
    )
    sep("=")
    print("  Stage 5 complete. Waiting for your approval before Stage 6.")
    sep("=")
    print()


if __name__ == "__main__":
    evaluate()
