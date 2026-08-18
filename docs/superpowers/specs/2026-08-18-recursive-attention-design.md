# Recursive Attention Layer Design

## Goal

Complete the `## Recursive Attention` entry in `List of Layers.md` with a focused explanation of Transformer multi-head self-attention and causal masking. Add Catppuccin Mocha diagrams that make the data flow and masking behavior easy to follow.

The scope deliberately excludes residual connections and LayerNorm, which can be documented separately.

## Existing conventions

`List of Layers.md` is a layer reference. Entries generally contain:

- an embedded `Layer - ...png` image;
- a **What it does** section;
- a **Where it is usually placed** section;
- an educational hand-written Python implementation with `__call__` and `parameters()`.

The new entry should preserve this structure and match the surrounding explanatory style. Existing uncommitted edits in the Transformer note and workspace must remain untouched.

## Content design

The new section will contain:

1. `Layer - MultiHeadAttention.png`, a left-to-right pipeline showing:
   - input `[B, T, C]`;
   - Q/K/V projections;
   - splitting into `h` heads with shape `[B, h, T, d]`;
   - scaled dot-product attention;
   - concatenating heads;
   - output projection.
2. A **What it does** explanation covering:
   - self-attention, where Q, K, and V come from the same input;
   - scaled dot-product attention:
     `Attention(Q,K,V) = softmax(QK^T / sqrt(d_k) + M)V`;
   - the role of multiple heads in learning different token relationships in parallel.
3. `Layer - CausalMask.png`, showing:
   - a score matrix with token labels;
   - future positions replaced by `-∞` before softmax;
   - the resulting lower-triangular attention pattern;
   - a legend distinguishing current/past tokens from future tokens.
4. A masking explanation stating that causal masking prevents a token from attending to future tokens, and distinguishing decoder causal self-attention from generally unmasked encoder self-attention.
5. A **Where it is usually placed** explanation locating the operation inside Transformer encoder/decoder blocks.
6. An educational hand-written `MultiHeadAttention` class using explicit tensor reshaping, scaling, a registered lower-triangular mask, softmax, head merging, and output projection. It will validate that `C % n_heads == 0` and expose `parameters()`.
7. A production-style PyTorch example using `nn.MultiheadAttention`, `batch_first=True`, and a boolean causal mask, with a note about `is_causal=True` where supported.

All tensor examples use `[B, T, C]`, where `B` is batch size, `T` is sequence length, and `C` is embedding width. The educational implementation will make the intermediate shape `[B, n_heads, T, head_dim]` explicit.

## Visual design

Both PNGs will use a dark Catppuccin Mocha palette:

- background `#1e1e2e`;
- primary text `#cdd6f4`;
- blue, mauve, and green for Q/K/V and projections;
- yellow for attention weights;
- red for masked future positions;
- teal for data flow;
- lavender/subtext colors for dimensions and annotations.

The images should be high-resolution PNGs readable at the scale used by the existing layer images and still legible when zoomed in Obsidian. The two concepts remain separate rather than combining all labels into one crowded diagram.

## Validation

Before completion:

- run Python syntax/compile checks on any generated example/helper code;
- if PyTorch is available, run a smoke test for both implementations;
- verify future-token attention weights are zero in the educational implementation;
- verify image files exist and Markdown embeds point to the correct filenames;
- check code fences and headings in `List of Layers.md`;
- inspect the final Git diff and confirm existing user modifications are preserved.

## Out of scope

- residual connections;
- LayerNorm;
- cross-attention;
- positional encoding;
- modifying `Transformer - Attention is all you need.md`;
- changing existing layer entries or their images.
