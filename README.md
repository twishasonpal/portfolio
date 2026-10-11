> ⚠️ **Copyright © 2024 Twisha Sonpal. All Rights Reserved.**
> This repository is **not open source**. Viewing is permitted for reference only. Copying, modifying, distributing, or reusing any part of this code or assets is strictly prohibited without explicit written permission. See [LICENSE](LICENSE) for details.

---

# Twisha Sonpal - Personal AI Portfolio & Neural Language Model

An interactive portfolio featuring an **in-browser neural language model assistant** and **verified portfolio knowledge base**. Built to run 100% client-side on **GitHub Pages** with zero external API dependencies, zero server hosting costs, and complete privacy.

---

## 🌟 Architecture Overview

```
                      ┌──────────────────────────────────────────────┐
                      │             Recruiter Question               │
                      └──────────────────────┬───────────────────────┘
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │       Twisha AI Chatbot       │
                             └───────────────┬───────────────┘
                                             │
                    ┌────────────────────────┴────────────────────────┐
                    ▼                                                 ▼
     ┌─────────────────────────────┐                  ┌──────────────────────────────┐
     │   Verified Portfolio KB     │                  │  Client-Side Transformer LM  │
     │     (data/portfolio.json)   │                  │     (js/nn_inference.js)     │
     ├─────────────────────────────┤                  ├──────────────────────────────┤
     │ • Skills & Tech Stack       │                  │ • 105,728 Parameters         │
     │ • Non-overlapping Experience│                  │ • Causal Self-Attention      │
     │ • Verified Project Details  │                  │ • Autoregressive Generation  │
     │ • Availability & Contact    │                  │ • Zero Server Calls          │
     └──────────────┬──────────────┘                  └──────────────┬───────────────┘
                    │                                                │
                    └────────────────────────┬───────────────────────┘
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │    Verified Answer Display    │
                             │   (Text + Web Speech Audio)   │
                             └───────────────────────────────┘
```

---

## 📁 Repository Structure

```
portfolio/
├── data/
│   └── portfolio.json         # Verified knowledge base (skills, experience, projects, contact)
├── model/
│   ├── config.json            # Model hyperparameters (d_model=64, n_heads=4, n_layers=3)
│   ├── tokenizer.json         # Vocabulary and token mappings
│   ├── weights.json           # Exported weights for browser inference (JSON format)
│   └── pytorch_model.pt       # PyTorch checkpoint trained locally
├── training/
│   ├── tokenizer.py           # Custom tokenizer with special tokens (<bos>, <eos>, <pad>, <unk>)
│   ├── model.py               # PyTorch Transformer LM implementation from scratch
│   ├── dataset.py             # Portfolio text sequence loader & QA dataset
│   ├── train.py               # PyTorch local training script with loss logging
│   ├── train_numpy.py         # Standalone NumPy training script (no PyTorch required)
│   └── export_model.py        # Exporter converting PyTorch weights into browser format
├── js/
│   ├── portfolio_kb.js        # Deterministic fact retriever & dynamic duration calculator
│   └── nn_inference.js        # Pure JavaScript Transformer forward pass & token generator
├── tests/
│   └── test_all.py            # Comprehensive test suite (tokenizer, model, KB, benchmark)
├── chatbot.js                 # Floating chatbot interface & dual-engine controller
├── chatbot.css                # Chatbot styling matching the dark zinc theme
├── index.html                 # Portfolio homepage
└── README.md
```

---

## 🚀 Quickstart & Windows Commands

### 1. Run the Local Test Suite
```powershell
python -m unittest tests/test_all.py
```
*Runs 6 comprehensive tests checking tokenizer encoding/decoding, PyTorch forward pass, weights integrity, non-overlapping experience calculation, and the 20-question recruiter benchmark.*

---

### 2. Retrain the Neural Model Locally (Optional)
If you update `data/portfolio.json` with new projects or certifications:

**Using PyTorch:**
```powershell
python training/train.py
python training/export_model.py
```

**Using Standalone NumPy (Instant on any CPU):**
```powershell
python training/train_numpy.py
```

---

### 3. Test Locally in Browser
You can preview the website and chatbot locally using Python's built-in HTTP server:
```powershell
python -m http.server 8000
```
Open your browser to: **[http://localhost:8000](http://localhost:8000)**

---

### 4. Deploy to GitHub Pages
Because the entire neural network inference and factual retrieval run directly in the browser via JavaScript, deployment requires only pushing your commits to GitHub:

```powershell
git add .
git commit -m "feat: in-browser neural language model and verified portfolio chatbot"
git push origin main
```

Your live site at `https://twishasonpal.github.io/portfolio` will immediately host:
- The updated portfolio pages
- The floating AI Avatar Assistant
- Client-side neural model weights
- Deterministic fact retrieval engine