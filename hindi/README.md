# AI Basics: न्यूरॉन से लेकर लैंग्वेज मॉडल तक

मॉडर्न लैंग्वेज मॉडल को शुरुआत से बनाने की एक पूरी गाइड — beginner से लेकर pro तक। आसान हिंदी में, और PyTorch कोड के साथ जो आप सच में चला सकते हैं।

> **नोट:** सारा कोड English में है (variables, function names, comments) ताकि आप उसे सीधे चला सकें। Explanation और intuition हिंदी में है ताकि समझना आसान हो।

## ये किसके लिए है

- **Beginners** जिनको थोड़ा Python आता है और वो समझना चाहते हैं कि ChatGPT जैसे मॉडल कैसे काम करते हैं, शुरू से अंत तक।
- **Intermediate लोग** जो `nn.Linear` use कर सकते हैं लेकिन समझना चाहते हैं कि transformer block ऐसा क्यों दिखता है।
- **Pros** जिनको RoPE math, GQA cache savings, MoE routing, या quantization formats पर refresher चाहिए।

अगर कोई section मुश्किल लगे, तो कोड पर सीधे जाओ, उसे चलाओ, और वापस आओ। पढ़ना और चलाना दोनों मिलकर पढ़ने से ज़्यादा तेज़ काम करते हैं।

## Learning Path

Chapters एक के बाद एक build होते हैं। अगर आपके पास सिर्फ़ fast tour के लिए time है, तो हर file के top पर **TL;DR** पढ़ लो।

| # | Topic | इससे क्या सीखोगे |
|---|-------|------------------|
| 00 | [Essential Math](./00-essential-math.md) | Matmul, gradients, chain rule, softmax, cross-entropy — हर chapter में दिखेगा। |
| 01 | [Neural Networks](./01-neural-networks.md) | NumPy + PyTorch में neural net शुरू से। Backprop intuition. |
| 02 | [PyTorch](./02-pytorch.md) | Tensors, autograd, modules, training loops, mixed precision। |
| 03 | [GPU Computing](./03-gpu-computing.md) | GPU इतने तेज़ क्यों हैं, memory hierarchy, kernel fusion, profiling। |
| 04 | [Data](./04-data.md) | FineWeb-Edu, DCLM, MinHash dedup, classifier filtering, sharding। |
| 05 | [Model Scale](./05-model-scale.md) | Scaling laws, Chinchilla, over-training, test-time compute। |
| 06 | [Tokenization & Embeddings](./06-tokenization-embeddings.md) | BPE, vocab choices, embedding tables, tied heads। |
| 07 | [Positional Encodings](./07-positional-encodings.md) | Sinusoidal, ALiBi, RoPE (पूरा math + code), YaRN। |
| 08 | [Attention Mechanisms](./08-attention-mechanisms.md) | Q/K/V, multi-head, causal masking, Flash Attention 3, FlexAttention। |
| 09 | [KV Cache, MQA, GQA](./09-kv-cache-mqa-gqa.md) | Inference में सबसे बड़ा win, plus MLA, PagedAttention, speculative decoding। |
| 10 | [Building Blocks](./10-building-blocks.md) | RMSNorm, SwiGLU, residuals, pre-norm, QK-norm। |
| 11 | [Building Qwen from Scratch](./11-building-qwen-from-scratch.md) | सब कुछ glue करके एक real, working LLM बनाओ। |
| 12 | [Quantization](./12-quantization.md) | INT8/INT4/FP8/NVFP4, GPTQ, AWQ, GGUF, BitNet, QLoRA। |
| 13 | [Mixture of Experts](./13-mixture-of-experts.md) | Sparse models, routing, DeepSeek-V3-style fine-grained MoE। |
| 14 | [Training Small Language Models](./14-training-small-language-models.md) | पूरा pretraining + SFT + DPO/GRPO pipeline Muon और FSDP2 के साथ। |
| 15 | [Reading Training Logs](./15-reading-training-logs.md) | W&B charts, हर metric का मतलब, loss curves debug करना। |
| 16 | [Frontier Models in 2026](./16-frontier-models-2026.md) | Gemma 4 (E2B/E4B/26B-A4B/31B Dense) और Qwen 3.6 (27B Dense / 35B-A3B MoE)। |
| 17 | [Production Inference](./17-production-inference.md) | 10,000 requests serve करना: vLLM/SGLang, FP8, prefix cache, HBM bandwidth, token caching, per-archetype pricing। |
| 18 | [AI Apps & Agents](./18-ai-apps-and-agents.md) | Model के ऊपर का harness: 12 usage patterns, MCP, real apps (Cursor, Claude Code, Cline), provider pricing, 300-line agent। |

## इस guide को कैसे use करें

1. **पहली बार 00-17 क्रम में पढ़ो।** अगर "matrix multiplication / chain rule / softmax / cross-entropy" थोड़े fuzzy हैं, तो [00-essential-math.md](./00-essential-math.md) से शुरू करो — 30 मिनट लगते हैं और बाकी book दोगुनी आसानी से पढ़ी जाएगी।
2. **00-03 foundations हैं। 04-13 model build करते हैं। 14-15 train और debug करते हैं। 16-17 frontier models और real users तक deployment। 18 इसे एक real product में wrap करता है।**
3. **कोड खुद type करो।** कोड पढ़ना और कोड लिखना अलग-अलग बातें हैं। Jupyter notebook खोलो और snippets recreate करो।
4. **जब भी हो सके, real hardware पर चलाओ।** Free Colab T4 GPU ज़्यादातर exercises के लिए काफ़ी है। Training के लिए, छोटा CPU run भी loop सिखा देता है।
5. **Math को कोड के बाद दोबारा पढ़ो।** Equations तब ज़्यादा clear होते हैं जब tensor shapes flow करते हुए देख लेते हो।

## Prerequisites

- Python 3.10+ का comfort (functions, classes, list comprehensions)।
- High-school algebra. हम matrix multiplication को जब पहली बार आए तब समझाते हैं।
- एक working PyTorch install: `pip install torch numpy`।
- Optional लेकिन useful: `pip install transformers datasets matplotlib`।

## Conventions

- **पहले simple हिंदी**, फिर equations और कोड।
- सारा कोड **PyTorch 2.x** है। CUDA-only sections clearly mark किए गए हैं।
- **Tensor shapes** हर जगह comments में लिखे हैं: `# x: (B, T, D)` का मतलब batch, time/sequence, dimension।
- **B** = batch, **T** = sequence length, **D** = model dimension, **H** = heads, **V** = vocab size।

## Style पर एक नोट

Goal समझना है, smart-sounding jargon नहीं। अगर कोई sentence smart लगे लेकिन help न करे, वो bug है — file खोलो और अगले reader के लिए rewrite करो।

अब चलो **[00-essential-math.md](./00-essential-math.md)** से शुरू करते हैं अगर math refresher चाहिए, या सीधे **[01-neural-networks.md](./01-neural-networks.md)** पर jump करो।
