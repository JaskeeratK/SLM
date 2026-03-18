# SLM — Small Language Model from Scratch

A 162M parameter GPT-style decoder-only transformer built entirely from scratch in PyTorch. Pretrained on WikiText-2 using the causal language modeling objective on a free-tier Colab T4 GPU.

---

## Results

| Metric | Value |
|--------|-------|
| Validation Loss | 2.11 |
| Validation Perplexity | 8.25 |
| Parameters | 162,320,640 |
| Training Time | ~1h 15min (3 epochs, T4 GPU) |

> **Note on perplexity:** 8.25 is lower than expected for a model this size and reflects partial overfitting to WikiText-2 (~2MB, 36K examples) — a small dataset relative to 162M parameters (~80 params per training token). GPT-2 (117M params) achieves ~29 perplexity trained on orders of magnitude more data. The number reflects memorization of a small corpus, not superior generalization.

---

## Architecture

```
Input token IDs  (batch=16, seq_len=127)
        ↓
Token Embedding  (50257 × 768)
        +
Positional Embedding  (127 × 768)
        ↓
× 12 Transformer Blocks
  ├── LayerNorm
  ├── Multi-Head Attention (12 heads × 64 dims)
  │     ├── Causal mask (upper triangular -inf)
  │     └── Scaled dot-product attention + W_O projection
  ├── Residual connection
  ├── LayerNorm
  ├── Feed Forward Network (768 → 3072 → 768, GELU)
  └── Residual connection
        ↓
LayerNorm
        ↓
LM Head  (768 → 50257)
        ↓
Logits  (batch=16, seq_len=127, vocab=50257)
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embedding dimension | 768 |
| Number of layers | 12 |
| Attention heads | 12 |
| Head dimension | 64 |
| FFN hidden dimension | 3072 (4× expand) |
| Max sequence length | 127 |
| Vocabulary size | 50,257 |
| Dropout | 0.1 |
| Activation | GELU |
| Normalization | Pre-norm (LayerNorm before each sublayer) |

---

## Training

| Setting | Value |
|---------|-------|
| Dataset | WikiText-2 |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 0.01 |
| Batch size | 16 |
| Epochs | 3 |
| Gradient clipping | 1.0 |
| Loss function | CrossEntropyLoss |
| Hardware | NVIDIA T4 (Google Colab free tier) |

**Loss curve (per epoch avg):**
```
Epoch 1 → 2.33
Epoch 2 → 1.99
Epoch 3 → 1.78
```

Started at batch 0 loss of **11.45** — consistent with random initialization where expected loss = log(50257) ≈ 10.82.

---

## Implementation Details

**Tokenizer** — GPT-2's pretrained BPE tokenizer (vocab size 50,257). Reused rather than trained from scratch to focus effort on the model architecture. BPE splits rare words into subword units ("Messi" → "Mess" + "i") while common words get dedicated tokens.

**Causal masking** — upper triangular boolean mask applied before softmax in every attention head. Sets future token scores to -inf so they become 0 after softmax, enforcing autoregressive generation — each token can only attend to itself and preceding tokens.

**Residual connections** — wrap both the attention and FFN sublayers. Provide direct gradient highways during backpropagation, solving the vanishing gradient problem in deep networks. Borrowed from ResNets (He et al., 2015).

**Pre-norm vs post-norm** — LayerNorm applied before each sublayer rather than after. More stable training, especially in the early stages.

**LM Head** — linear projection from embed_dim (768) to vocab_size (50,257). Accounts for ~38M of the 162M total parameters. GPT-2 ties this layer's weights with the token embedding matrix to reduce parameter count — not implemented here.

**Parameter count breakdown (approximate):**
```
Token embedding:      50257 × 768  =  38.6M
Positional embedding: 127   × 768  =   0.1M
× 12 Transformer blocks:
  Attention (W_Q,K,V,O): 4 × 768²  =   2.4M  × 12 = 28.3M
  FFN (W1, W2):          2 × 768×3072 = 4.7M × 12 = 56.6M
  LayerNorms:            negligible
LM Head:              768 × 50257  =  38.6M
─────────────────────────────────────────────
Total:                               ~162M
```

---

## Text Generation

Two decoding strategies implemented:

**Greedy decoding** — argmax at every step. Deterministic, fast, prone to repetitive loops.

**Top-k sampling with temperature** — logits divided by temperature τ before softmax (controls distribution sharpness), then sample from top-k most probable tokens via multinomial sampling. Produces more diverse and natural output.

```python
# example generation
output = generate_sampled(model, tokenizer, 
                          prompt="The football match ended",
                          temperature=0.8, 
                          top_k=40)
```

Sample output after training:
```
The football match ended in the first time since the second quarter 
of the quarter of the season, having scored the first time in the 
second quarter...
```

---
