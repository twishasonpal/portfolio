"""
model/transformer.py - Causal Transformer Language Model built from scratch in PyTorch.
Implements:
- Positional & Token Embeddings
- Scaled Dot-Product Self-Attention with Causal Masking
- Multi-Head Attention (MHA)
- Position-wise Feed-Forward Network (GELU)
- Transformer Decoder Block with LayerNorm & Residual Connections
- Causal Language Model with Language Modeling Output Head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention.
    Computes Q, K, V projections and applies masked scaled dot-product attention
    to ensure tokens can only attend to past and current positions.
    """
    def __init__(self, d_model: int = 128, n_heads: int = 4, max_seq_len: int = 128, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head (e.g. 128 // 4 = 32)
        
        # Linear projections for Query, Key, Value
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection back to d_model
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask: lower-triangular matrix of 1s (shape: 1, 1, max_seq_len, max_seq_len)
        # register_buffer ensures this tensor is saved with the model state but not treated as a trainable weight
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape:  (batch_size, seq_len, d_model)
        Output shape: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # 1. Project inputs to Q, K, V
        # Shape: (batch_size, seq_len, d_model)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 2. Reshape into multi-head format: (batch_size, n_heads, seq_len, d_k)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 3. Scaled Dot-Product Attention: (Q * K^T) / sqrt(d_k)
        # Scores shape: (batch_size, n_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 4. Apply Causal Mask (mask out future tokens by setting them to -inf)
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        
        # 5. Softmax to get attention probabilities
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 6. Multiply attention weights by Values: (attn_weights * V)
        # Context shape: (batch_size, n_heads, seq_len, d_k)
        context = torch.matmul(attn_weights, v)
        
        # 7. Concatenate heads back to d_model: (batch_size, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # 8. Final linear projection
        return self.out_proj(context)


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    Expands d_model to 4 * d_model with GELU activation, then projects back.
    FFN(x) = GELU(x * W1 + b1) * W2 + b2
    """
    def __init__(self, d_model: int = 128, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape:  (batch_size, seq_len, d_model)
        Output shape: (batch_size, seq_len, d_model)
        """
        return self.linear2(self.dropout(self.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    Single Decoder Transformer Block.
    Contains:
    - Pre-LayerNorm -> Causal Self-Attention -> Residual Connection
    - Pre-LayerNorm -> FeedForward -> Residual Connection
    """
    def __init__(self, d_model: int = 128, n_heads: int = 4, d_ff: int = 512, max_seq_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads, max_seq_len=max_seq_len, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Residual structure: x = x + Attention(LayerNorm(x))
                            x = x + FeedForward(LayerNorm(x))
        """
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class PortfolioLanguageModel(nn.Module):
    """
    Full Causal Transformer Language Model.
    Architecture:
    1. Token Embeddings + Learned Positional Embeddings
    2. N stacked Transformer Blocks
    3. Final LayerNorm
    4. Language Modeling Output Head (projects back to vocab_size)
    """
    def __init__(
        self,
        vocab_size: int = 364,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_seq_len: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token and Positional Embedding tables
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Stacked Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                max_seq_len=max_seq_len,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        # Final LayerNorm
        self.ln_final = nn.LayerNorm(d_model)
        
        # LM Head: Maps d_model to vocab_size (logits for each vocabulary token)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying (tying token embeddings and output head weights)
        # This reduces parameter count and improves representation learning
        self.lm_head.weight = self.token_embeddings.weight
        
        # Initialize weights with standard normal distribution (std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def count_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for training or inference.
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_len) with integer token IDs
            targets: Optional tensor of shape (batch_size, seq_len) with ground-truth next tokens
            
        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
            loss: Optional scalar CrossEntropyLoss if targets are provided
        """
        batch_size, seq_len = input_ids.shape
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
        
        # Create position indices: [0, 1, 2, ..., seq_len - 1]
        device = input_ids.device
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        
        # 1. Sum Token and Positional Embeddings
        tok_emb = self.token_embeddings(input_ids)      # (batch, seq_len, d_model)
        pos_emb = self.position_embeddings(positions)   # (1, seq_len, d_model)
        x = self.dropout(tok_emb + pos_emb)
        
        # 2. Pass through Transformer Blocks
        for block in self.blocks:
            x = block(x)
            
        # 3. Final LayerNorm
        x = self.ln_final(x)
        
        # 4. Project to Vocabulary Logits
        # Shape: (batch_size, seq_len, vocab_size)
        logits = self.lm_head(x)
        
        # 5. Compute Cross-Entropy Loss if targets are provided (Training mode)
        loss = None
        if targets is not None:
            # Flatten batch and sequence dimensions for cross-entropy:
            # logits: (batch * seq_len, vocab_size), targets: (batch * seq_len)
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1
            )
            
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 30,
        temperature: float = 0.7,
        top_k: int = 10,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Autoregressive next-token text generation loop.
        Repeatedly predicts the next token and appends it to input_ids.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to context window if needed
            idx_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
            
            # Forward pass to get logits for the last token position
            logits, _ = self(idx_cond)
            # Focus on the very last time step: (batch_size, vocab_size)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            
            # Optional Top-K sampling: keep only top k highest scoring tokens
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample next token from probability distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append next_token to the running sequence
            input_ids = torch.cat((input_ids, next_token), dim=1)
            
            # Stop early if End-of-Sequence token was generated
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
                
        return input_ids
