# Architecture notes

## Data flow

```
Text input (per sequence in batch)
        │
        ▼
Tokenizer.tokenize()  -> Vec<usize>  (word -> vocabulary id, "<PAD>" reserved as id 0)
        │
        ▼
Padding               -> all sequences in a batch padded to the same seq_len
        │
        ▼
Embedding.embed()     -> Vec<&Tensor>  (one [d_model] vector per token id, looked up from a
                          fixed, randomly-initialized table)
        │
        ▼
Tensor::stack_batch() -> Tensor [batch, seq_len, d_model]
        │
        ▼
For each of num_layers TransformerLayer instances:
    │
    ├─ attention(x)
    │     Q = x · W_Q        [batch, seq_len, d_k]
    │     K = x · W_K        [batch, seq_len, d_k]
    │     V = x · W_V        [batch, seq_len, d_v]
    │     scores = (Q · Kᵀ) / sqrt(d_k)   [batch, seq_len, seq_len]
    │     weights = softmax_rows(scores)
    │     weights = dropout(weights, p)
    │     attn_out = weights · V          [batch, seq_len, d_v]
    │
    ├─ x1 = layer_norm(x + attn_out)      -- residual connection #1
    │
    ├─ mlp(x1)
    │     hidden = gelu(x1 · W1 + b1)     [batch, seq_len, d_ff]
    │     hidden = dropout(hidden, p)
    │     mlp_out = hidden · W2 + b2      [batch, seq_len, d_model]
    │
    └─ x2 = layer_norm(x1 + mlp_out)      -- residual connection #2
    │
    (x2 becomes the input to the next layer)
        │
        ▼
Final output tensor [batch, seq_len, d_model], printed via to_nested_string()
```

## Key design decisions

- **d_v is forced equal to d_model.** Since attention output must be added back to the
  original input (residual connection), and true multi-head concatenation isn't implemented,
  d_v = d_model keeps shapes compatible. d_k is still derived from d_model / num_heads and
  used only for score scaling.
- **Weights are per-layer, seeded independently.** Each TransformerLayer is constructed with
  `MANUAL_SEED + layer_index`, so stacked layers don't end up with identical random weights.
- **Flat buffer + shape, not nested Vec<Vec<...>>.** Tensor stores data as one contiguous
  `Vec<f32>` plus a `shape: Vec<usize>` describing how to interpret it. This mirrors how real
  tensor libraries (PyTorch, ndarray) lay out memory, and avoids the poor cache locality and
  awkward indexing of nested Vecs.
- **Two matmul variants.** `matmul_weight` handles batched input × a shared 2D weight matrix
  (used for all Q/K/V/MLP projections, since weights don't vary per batch element).
  `matmul_batched` handles batched input × batched input (used for Q·Kᵀ and weights·V, since
  both operands vary per sequence in the batch).
- **Dropout defaults to a no-op at p = 0.** Since there's no training loop yet, dropout exists
  as scaffolding for future training work but has no effect unless explicitly enabled.

## Known simplifications vs. a production transformer

| Aspect               | This project                   | Real transformer (e.g. nn.Transformer)           |
| -------------------- | ------------------------------ | ------------------------------------------------ |
| Tokenization         | Whitespace split               | Subword (BPE/WordPiece)                          |
| Multi-head attention | num_heads affects scaling only | Full split → parallel attention → concat         |
| Padding              | Sequences padded with `<PAD>`  | Padded + masked out of attention                 |
| Positional info      | None                           | Sinusoidal or learned positional encoding        |
| Weights              | Fixed random, never updated    | Learned via backpropagation over training data   |
| Precision            | f32 throughout                 | Often mixed precision (f16/bf16) for performance |
