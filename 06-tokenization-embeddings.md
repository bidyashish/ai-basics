# 06 · Tokenization & Embeddings

> **TL;DR** A model can't read text directly. We chop text into **tokens** (sub-word pieces) and replace each token with an integer ID. The first model layer is an **embedding table**: a matrix of shape `(V, D)` where row `i` is the learned vector for token `i`. The last layer (the **LM head**) is `(D, V)` — often the same matrix transposed, called **tied embeddings**. In 2026, **byte-pair encoding (BPE) on UTF-8 bytes** with a vocab of 32k-256k is the universal default.

## 1. Why tokenize at all?

Three options:

| Unit | Pros | Cons |
|------|------|------|
| Characters / bytes | Tiny vocab, no OOV | Sequences are long → expensive |
| Words | Short sequences | Huge vocab, breaks on new words |
| **Subwords (BPE)** | Balanced: ~4 chars/token, no OOV, ~50k vocab | Slightly opaque |

Subword tokenization wins. The vocabulary contains every byte (256 entries) plus learned merges of common byte sequences. So **any** text — code, emoji, Chinese, gibberish — encodes losslessly.

A useful rule: **English text averages ~4 chars per token**. Code averages ~3.

---

## 2. Byte-pair encoding (BPE) in 60 seconds

The training algorithm:

1. Start with vocab = every individual byte.
2. Count all adjacent pairs in the corpus.
3. Merge the most frequent pair into a new symbol; add to vocab.
4. Repeat until vocab reaches target size (e.g., 50,000).

So the model learns merges like `t + h → th`, then `th + e → the`, then `▁ + the → ▁the`. By the end, common words are single tokens, rare words are several tokens, and unknown bytes can always be encoded byte-by-byte.

```python
# toy BPE training
from collections import Counter

def bpe_train(corpus, num_merges):
    # represent each word as a tuple of bytes
    words = [tuple(w.encode('utf-8')) for w in corpus.split()]
    vocab = set(b for w in words for b in w)
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for w in words:
            for i in range(len(w) - 1):
                pairs[(w[i], w[i+1])] += 1
        if not pairs: break
        pair, _ = pairs.most_common(1)[0]
        merges.append(pair)
        new_token = pair[0] + (pair[1] if isinstance(pair[1], bytes) else bytes([pair[1]]))
        vocab.add(new_token)
        # replace in all words
        new_words = []
        for w in words:
            i, out = 0, []
            while i < len(w):
                if i < len(w) - 1 and (w[i], w[i+1]) == pair:
                    out.append(new_token); i += 2
                else:
                    out.append(w[i]); i += 1
            new_words.append(tuple(out))
        words = new_words
    return merges, vocab
```

Real BPE uses byte-level encoding (`ä` → 2 bytes), pre-tokenization with regex (so it doesn't merge across spaces or numerals), and special tokens (`<bos>`, `<eos>`, `<pad>`, chat template tokens like `<|im_start|>`). **`tiktoken`** (OpenAI) and **`tokenizers`** (HuggingFace) are the standard fast implementations.

---

## 3. Vocab-size trade-offs

| Vocab | Sequence length | Embedding params (`V × D`) | When to use |
|-------|-----------------|---------------------------|-------------|
| 8 k | very long | small | toy / character-level |
| 32 k | longer | small | pre-2024 default; English-only |
| 50 k | balanced | medium | GPT-2 era |
| 100 k | shorter | bigger | code-heavy or multilingual |
| 128 k | shorter | bigger | **2026 default** (Llama 3, Qwen3) |
| 256 k+ | shortest | huge | DeepSeek-V3 (129 k), some multimodal |

Bigger vocab = fewer tokens per text = fewer FLOPs per token at inference. But the embedding table grows linearly. Llama-3's 128k vocab adds ~500M params at `D=4096`, but cuts non-English token counts in half. Worth it.

There's a sweet spot per language mix and target FLOPs. Empirically, for English-only models around 1B params, **32k-50k** is fine. For multilingual or bigger models, **100k-150k**.

---

## 4. Tokenizer libraries you'll use

### `tiktoken` (OpenAI)

```python
import tiktoken
enc = tiktoken.get_encoding('cl100k_base')      # GPT-3.5 / 4 tokenizer
ids = enc.encode("Hello, world!")               # [9906, 11, 1917, 0]
text = enc.decode(ids)
```

Fast (Rust). Used by OpenAI, but also general-purpose.

### `transformers.AutoTokenizer`

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B')
ids = tok("Hello world", return_tensors='pt').input_ids
text = tok.decode(ids[0])
```

The standard for using existing model checkpoints. Each model has its own tokenizer and you must use the matching one.

### Train your own with `tokenizers`

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

tok = Tokenizer(BPE(unk_token=None))
tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
trainer = BpeTrainer(vocab_size=50_000, special_tokens=['<|bos|>','<|eos|>','<|pad|>'])
tok.train(['big_corpus.txt'], trainer)
tok.save('my_tokenizer.json')
```

This is what teams running their own pretraining do.

---

## 5. Special tokens and chat templates

Modern LLMs aren't just trained on raw text — they're fine-tuned with **chat templates** that demarcate roles. Qwen / Llama / ChatML use tokens like:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi!<|im_end|>
```

The tokens `<|im_start|>`, `<|im_end|>`, role names — these all need to exist in your vocab. The Hugging Face tokenizer has `apply_chat_template`:

```python
msgs = [{'role': 'user', 'content': 'Hello'}]
prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
```

Reasoning models add more: `<think>`, `</think>`, sometimes tool-call markers `<|tool_call|>`. When you fine-tune, you must mirror the format the base model expects, or chat will break.

---

## 6. From token IDs to vectors: the embedding table

Once tokens are integers in `[0, V)`, the model needs vectors. The first layer is:

```python
class TokenEmbedding(nn.Module):
    def __init__(self, V, D):
        super().__init__()
        self.embedding = nn.Embedding(V, D)
    def forward(self, ids):           # ids: (B, T)
        return self.embedding(ids)    # → (B, T, D)
```

**`nn.Embedding(V, D)` is just a `(V, D)` weight matrix with a fast `gather` lookup.** It's mathematically equivalent to one-hot encoding the IDs and matrix-multiplying by `W`, but the lookup is `O(B × T)` instead of `O(B × T × V)`.

The embedding values are learned. After training, you can do all kinds of fun things — measure cosine similarity to find synonyms, project to 2D with t-SNE, etc.

---

## 7. The LM head and tied embeddings

At the **end** of the network, you predict the next token. Output shape: `(B, T, V)` — a logit per vocab entry per position.

```python
class LMHead(nn.Module):
    def __init__(self, D, V):
        super().__init__()
        self.proj = nn.Linear(D, V, bias=False)
    def forward(self, h):            # h: (B, T, D)
        return self.proj(h)          # → (B, T, V)
```

That `(D, V)` weight matrix has a transpose-relationship to the embedding `(V, D)`. **Tied embeddings** share the weights:

```python
class TiedLMModel(nn.Module):
    def __init__(self, V, D):
        super().__init__()
        self.embed = nn.Embedding(V, D)
        # ... transformer blocks ...

    def forward(self, ids):
        x = self.embed(ids)          # (B, T, D)
        # ... transformer ...
        logits = x @ self.embed.weight.T   # (B, T, V) — tied!
        return logits
```

Tying saves `V × D` parameters (significant for large vocabs) and often improves perplexity slightly. Llama 1, GPT-2, Qwen2.5-0.5B, SmolLM all tie. Llama 3-8B+ does **not** tie because the LM head can specialize separately at scale.

---

## 8. Embeddings as semantic vectors

After training, the embedding rows are far from random. Similar tokens cluster:

```python
# distance between 'king' and 'queen'
import torch.nn.functional as F
e = model.embed.weight                              # (V, D)
king, queen = tok.encode(' king'), tok.encode(' queen')
sim = F.cosine_similarity(e[king[0]], e[queen[0]], dim=0)
print(sim.item())                                   # surprisingly high
```

This is the famous "king - man + woman ≈ queen" trick from word2vec. It still works in LLM token embeddings to a degree — though token splits (BPE) add noise.

For document embeddings (semantic search, RAG), don't use `nn.Embedding` directly — use **sentence-transformers** or a dedicated retrieval model. The token embeddings of a generative LLM are tuned for prediction, not similarity.

---

## 9. Practical tokenization gotchas

- **Whitespace matters.** `"hello"` and `" hello"` (with leading space) tokenize differently. Most tokenizers attach the leading space to the token.
- **Numbers.** Some tokenizers split digits one-by-one (Llama 3 does), some merge multi-digit tokens. Single-digit tokenization is better for arithmetic — measurably so on benchmarks.
- **Code.** Code-heavy datasets need merges that respect whitespace (indentation matters!). Train your tokenizer on your actual data, including code, or use a model whose tokenizer was.
- **Chinese / Japanese / Korean.** Single characters can be high-frequency; you want lots of merges.
- **Special tokens during inference.** If the user sends a string containing `<|im_end|>`, naive tokenization treats it as text. Frameworks like vLLM offer `add_special_tokens=False` to keep user input safe.
- **Token count != character count.** Counting cost in characters is wrong; bill / measure in tokens.

---

## 10. Implementing a minimal BPE encoder/decoder

```python
class MiniBPE:
    def __init__(self, merges, special_tokens=None):
        # merges: list of (a, b) tuples, applied in order
        self.merges = {pair: i for i, pair in enumerate(merges)}
        self.vocab = {}
        for i in range(256):
            self.vocab[bytes([i])] = i
        for i, pair in enumerate(merges):
            a, b = self._dec(pair[0]), self._dec(pair[1])
            self.vocab[a + b] = 256 + i
        self.specials = special_tokens or {}
        self.inv = {v: k for k, v in self.vocab.items()}

    def _dec(self, t):
        return t if isinstance(t, bytes) else bytes([t])

    def encode(self, text):
        toks = [bytes([b]) for b in text.encode('utf-8')]
        while True:
            best = None; best_pri = float('inf')
            for i in range(len(toks) - 1):
                p = (toks[i], toks[i+1])
                if p in self.merges and self.merges[p] < best_pri:
                    best, best_pri = i, self.merges[p]
            if best is None: break
            toks[best:best+2] = [toks[best] + toks[best+1]]
        return [self.vocab[t] for t in toks]

    def decode(self, ids):
        return b''.join(self.inv[i] for i in ids).decode('utf-8', errors='replace')
```

Run it. Watch a real model encode `"Hello world"` and trace the merges. The "magic" disappears.

---

## 11. Embeddings + position = ready for the transformer

The transformer block expects shape `(B, T, D)`. The embedding gives you that. Many older models also added a **positional embedding** at this stage:

```python
class GPTEmbedding(nn.Module):
    def __init__(self, V, T_max, D):
        super().__init__()
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(T_max, D)

    def forward(self, ids):
        T = ids.size(1)
        pos = torch.arange(T, device=ids.device)
        return self.tok(ids) + self.pos(pos)        # (B, T, D)
```

In 2026, almost everyone has switched to **RoPE** (chapter 7), which is applied *inside* attention and not at the embedding layer at all. So a modern model's embedding is just `tok = self.embed(ids)`.

---

## 12. A 2026 cheat sheet

- Use **byte-level BPE** with vocab **32k-256k**.
- Choose vocab to match languages and code share. **Bigger vocab = shorter sequences = cheaper inference.**
- Use **`tokenizers` / `tiktoken`** to train, **`AutoTokenizer`** to consume.
- **Tied embeddings** for small models, untied for ≥7B.
- Single-digit tokenization helps math.
- Always print and inspect a few tokenized examples — bugs hide there.
- Reuse the model's tokenizer; never mix-and-match.
- Special tokens deserve respect. They define the chat protocol.

---

## Going deeper

- Karpathy's **`minbpe`** repo — BPE from scratch in 200 lines.
- Sennrich et al. 2016 — original BPE paper.
- Kudo & Richardson 2018 — SentencePiece (the Unigram alternative still used by some Asian-language tokenizers).
- The Llama 3 / Qwen 2.5 / DeepSeek tokenizer JSON files on HuggingFace — read them, they're surprisingly clear.

Next: **[07-positional-encodings.md](./07-positional-encodings.md)** — telling the model where each token is.
