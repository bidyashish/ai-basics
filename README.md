# AI Basics: From Neurons to Language Models

A complete, beginner-to-pro guide to building modern language models from scratch. Written in plain English with PyTorch code you can actually run.

## Who this is for

- **Beginners** who know a little Python and want to understand how ChatGPT-style models work, end to end.
- **Intermediate folks** who can use `nn.Linear` but want to know why a transformer block looks the way it does.
- **Pros** who want refreshers on RoPE math, GQA cache savings, MoE routing, or quantization formats.

You should not feel stuck. If a section feels too dense, skip to the code, run it, and come back. Reading and running is faster than reading alone.

## The learning path

The chapters build on each other. If you only have time for a fast tour, read the **TL;DR** at the top of each file.

| # | Topic | What you walk away with |
|---|-------|--------------------------|
| 00 | [Essential Math](./00-essential-math.md) | Matmul, gradients, chain rule, softmax, cross-entropy — the math you'll see every chapter. |
| 01 | [Neural Networks](./01-neural-networks.md) | A neural net from scratch in NumPy + PyTorch. Backprop intuition. |
| 02 | [PyTorch](./02-pytorch.md) | Tensors, autograd, modules, training loops, mixed precision. |
| 03 | [GPU Computing](./03-gpu-computing.md) | Why GPUs are fast, memory hierarchy, kernel fusion, profiling. |
| 04 | [Data](./04-data.md) | FineWeb-Edu, DCLM, MinHash dedup, classifier filtering, sharding. |
| 05 | [Model Scale](./05-model-scale.md) | Scaling laws, Chinchilla, over-training, test-time compute. |
| 06 | [Tokenization & Embeddings](./06-tokenization-embeddings.md) | BPE, vocab choices, embedding tables, tied heads. |
| 07 | [Positional Encodings](./07-positional-encodings.md) | Sinusoidal, ALiBi, RoPE (full math + code), YaRN. |
| 08 | [Attention Mechanisms](./08-attention-mechanisms.md) | Q/K/V, multi-head, causal masking, Flash Attention 3, FlexAttention. |
| 09 | [KV Cache, MQA, GQA](./09-kv-cache-mqa-gqa.md) | The single biggest inference win, plus MLA, PagedAttention, speculative decoding. |
| 10 | [Building Blocks](./10-building-blocks.md) | RMSNorm, SwiGLU, residuals, pre-norm, QK-norm. |
| 11 | [Building Qwen from Scratch](./11-building-qwen-from-scratch.md) | Glue everything into a real, working LLM with weights you can load. |
| 12 | [Quantization](./12-quantization.md) | INT8/INT4/FP8/NVFP4, GPTQ, AWQ, GGUF, BitNet, QLoRA. |
| 13 | [Mixture of Experts](./13-mixture-of-experts.md) | Sparse models, routing, DeepSeek-V3-style fine-grained MoE. |
| 14 | [Training Small Language Models](./14-training-small-language-models.md) | Full pretraining + SFT + DPO/GRPO pipeline with Muon and FSDP2. |
| 15 | [Reading Training Logs](./15-reading-training-logs.md) | W&B charts, what each metric means, debugging loss curves. |

## How to use this guide

1. **Start with [00-essential-math.md](./00-essential-math.md)** if any of "matrix multiplication / chain rule / softmax / cross-entropy" sound fuzzy. It takes 30 minutes and the rest of the book reads twice as fast afterward.
2. **Read 01-15 in order** the first time. 00-03 are foundations. 04-13 build the model. 14-15 ship it and read its logs.
3. **Type the code yourself.** Reading code is not the same as writing it. Open a Jupyter notebook and recreate the snippets.
4. **Run on real hardware when possible.** A free Colab T4 GPU is enough for most exercises. For training, even a small CPU run teaches you the loop.
5. **Re-read the math once you've coded it.** The equations make more sense after you've seen the tensor shapes flow through them.

## Prerequisites

- Python 3.10+ comfort (functions, classes, list comprehensions).
- High-school algebra. We explain matrix multiplication when it first appears.
- A working PyTorch install: `pip install torch numpy`.
- Optional but useful: `pip install transformers datasets matplotlib`.

## Conventions

- **Simple English first**, equations and code second.
- All code is **PyTorch 2.x**. We mark CUDA-only sections clearly.
- **Tensor shapes** are spelled out in comments: `# x: (B, T, D)` means batch, time/sequence, dimension.
- **B** = batch, **T** = sequence length, **D** = model dimension, **H** = heads, **V** = vocab size.

## A note on style

The goal is understanding, not impressive jargon. If you find a sentence here that sounds smart but doesn't help, that's a bug — open the file and rewrite it for the next reader.

Now go to **[00-essential-math.md](./00-essential-math.md)** if you need the math refresher, or jump straight to **[01-neural-networks.md](./01-neural-networks.md)**.
