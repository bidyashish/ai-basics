# 06 · Tokenization & Embeddings

> **TL;DR** Model directly text नहीं पढ़ सकता। हम text को **tokens** (sub-word pieces) में chop करते हैं और हर token को एक integer ID से replace करते हैं। पहली model layer एक **embedding table** है: shape `(V, D)` का matrix जहां row `i` token `i` के लिए learned vector है। Last layer (the **LM head**) `(D, V)` है — often same matrix transposed, called **tied embeddings**। 2026 में, **byte-pair encoding (BPE) on UTF-8 bytes** with vocab of 32k-256k universal default है।

## 1. Tokenize ही क्यों?

तीन options:

| Unit | Pros | Cons |
|------|------|------|
| Characters / bytes | Tiny vocab, no OOV | Sequences long → expensive |
| Words | Short sequences | Huge vocab, new words पर breaks |
| **Subwords (BPE)** | Balanced: ~4 chars/token, no OOV, ~50k vocab | Slightly opaque |

Subword tokenization जीतता है। Vocabulary हर individual byte (256 entries) plus common byte sequences के learned merges contain करती है। तो **कोई भी** text — code, emoji, Chinese, gibberish — losslessly encode होता है।

एक useful rule: **English text averages ~4 chars per token**। Code averages ~3।

---

## 2. Byte-pair Encoding (BPE) 60 seconds में

Training algorithm:

1. Vocab = हर individual byte से start करो।
2. Corpus में सारे adjacent pairs count करो।
3. सबसे frequent pair को एक नए symbol में merge करो; vocab में add करो।
4. Repeat करो जब तक vocab target size (e.g., 50,000) पर न पहुंचे।

तो model `t + h → th`, फिर `th + e → the`, फिर `▁ + the → ▁the` जैसे merges सीखता है। End तक, common words single tokens हैं, rare words कई tokens हैं, और unknown bytes हमेशा byte-by-byte encode हो सकते हैं।

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

Real BPE byte-level encoding (`ä` → 2 bytes), regex के साथ pre-tokenization (तो ये spaces या numerals के across merge नहीं करता), और special tokens (`<bos>`, `<eos>`, `<pad>`, chat template tokens जैसे `<|im_start|>`) use करता है। **`tiktoken`** (OpenAI) और **`tokenizers`** (HuggingFace) standard fast implementations हैं।

---

## 3. Vocab-size Trade-offs

| Vocab | Sequence length | Embedding params (`V × D`) | कब use करें |
|-------|-----------------|---------------------------|-------------|
| 8 k | very long | small | toy / character-level |
| 32 k | longer | small | pre-2024 default; English-only |
| 50 k | balanced | medium | GPT-2 era |
| 100 k | shorter | bigger | code-heavy या multilingual |
| 128 k | shorter | bigger | **2026 default** (Llama 3, Qwen3) |
| 256 k+ | shortest | huge | DeepSeek-V3 (129 k), some multimodal |

Bigger vocab = fewer tokens per text = inference पर fewer FLOPs per token। लेकिन embedding table linearly grow करती है। Llama-3 का 128k vocab `D=4096` पर ~500M params add करता है, लेकिन non-English token counts को आधा cut करता है। Worth it।

हर language mix और target FLOPs के लिए एक sweet spot है। Empirically, around 1B params के English-only models के लिए, **32k-50k** fine है। Multilingual या bigger models के लिए, **100k-150k**।

---

## 4. Tokenizer Libraries जो आप Use करोगे

### `tiktoken` (OpenAI)

```python
import tiktoken
enc = tiktoken.get_encoding('cl100k_base')      # GPT-3.5 / 4 tokenizer
ids = enc.encode("Hello, world!")               # [9906, 11, 1917, 0]
text = enc.decode(ids)
```

Fast (Rust)। OpenAI द्वारा used, लेकिन general-purpose भी।

### `transformers.AutoTokenizer`

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B')
ids = tok("Hello world", return_tensors='pt').input_ids
text = tok.decode(ids[0])
```

Existing model checkpoints use करने का standard। हर model का अपना tokenizer है और आपको matching वाला use करना चाहिए।

### `tokenizers` से अपना खुद का train करो

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

ये वो है जो अपनी khud की pretraining run करने वाली teams करती हैं।

---

## 5. Special Tokens और Chat Templates

Modern LLMs सिर्फ़ raw text पर trained नहीं हैं — वो **chat templates** के साथ fine-tuned हैं जो roles को demarcate करते हैं। Qwen / Llama / ChatML इस तरह के tokens use करते हैं:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi!<|im_end|>
```

Tokens `<|im_start|>`, `<|im_end|>`, role names — इन सब का आपकी vocab में exist करना ज़रूरी है। HuggingFace tokenizer में `apply_chat_template` है:

```python
msgs = [{'role': 'user', 'content': 'Hello'}]
prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
```

Reasoning models और जोड़ते हैं: `<think>`, `</think>`, sometimes tool-call markers `<|tool_call|>`। जब आप fine-tune करते हो, आपको base model जो format expect करता है उसको mirror करना होगा, या chat break हो जाएगा।

---

## 6. Token IDs से Vectors तक: Embedding Table

एक बार tokens `[0, V)` में integers हो जाते हैं, model को vectors चाहिए। पहली layer है:

```python
class TokenEmbedding(nn.Module):
    def __init__(self, V, D):
        super().__init__()
        self.embedding = nn.Embedding(V, D)
    def forward(self, ids):           # ids: (B, T)
        return self.embedding(ids)    # → (B, T, D)
```

**`nn.Embedding(V, D)` बस fast `gather` lookup के साथ एक `(V, D)` weight matrix है।** ये mathematically IDs को one-hot encode करने और `W` से matrix-multiply करने के equivalent है, लेकिन lookup `O(B × T × V)` के बजाय `O(B × T)` है।

Embedding values learned हैं। Training के बाद, आप कई fun चीज़ें कर सकते हो — synonyms find करने के लिए cosine similarity measure करो, t-SNE से 2D में project करो, etc।

---

## 7. LM Head और Tied Embeddings

Network के **end** पर, हम next token predict करते हैं। Output shape: `(B, T, V)` — per position per vocab entry एक logit।

```python
class LMHead(nn.Module):
    def __init__(self, D, V):
        super().__init__()
        self.proj = nn.Linear(D, V, bias=False)
    def forward(self, h):            # h: (B, T, D)
        return self.proj(h)          # → (B, T, V)
```

उस `(D, V)` weight matrix का embedding `(V, D)` के साथ transpose-relationship है। **Tied embeddings** weights share करते हैं:

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

Tying `V × D` parameters बचाता है (large vocabs के लिए significant) और often perplexity slightly improve करता है। Llama 1, GPT-2, Qwen2.5-0.5B, SmolLM सब tie करते हैं। Llama 3-8B+ **नहीं** tie करता क्योंकि LM head scale पर separately specialize कर सकता है।

---

## 8. Embeddings as Semantic Vectors

Training के बाद, embedding rows random से far हैं। Similar tokens cluster करते हैं:

```python
# 'king' और 'queen' के बीच distance
import torch.nn.functional as F
e = model.embed.weight                              # (V, D)
king, queen = tok.encode(' king'), tok.encode(' queen')
sim = F.cosine_similarity(e[king[0]], e[queen[0]], dim=0)
print(sim.item())                                   # surprisingly high
```

ये word2vec से famous "king - man + woman ≈ queen" trick है। ये LLM token embeddings में अभी भी काफी हद तक काम करता है — हालांकि token splits (BPE) noise add करते हैं।

Document embeddings (semantic search, RAG) के लिए, सीधे `nn.Embedding` use मत करो — **sentence-transformers** या एक dedicated retrieval model use करो। Generative LLM के token embeddings prediction के लिए tuned हैं, similarity के लिए नहीं।

---

## 9. Practical Tokenization Gotchas

- **Whitespace matter करता है।** `"hello"` और `" hello"` (leading space के साथ) अलग tokenize होते हैं। Most tokenizers leading space को token से attach करते हैं।
- **Numbers।** कुछ tokenizers digits को one-by-one split करते हैं (Llama 3 करता है), कुछ multi-digit tokens merge करते हैं। Single-digit tokenization arithmetic के लिए better है — benchmarks पर measurably।
- **Code।** Code-heavy datasets को merges चाहिए जो whitespace respect करें (indentation matters!)। अपने actual data, including code, पर अपना tokenizer train करो, या ऐसा model use करो जिसका tokenizer इस पर trained था।
- **Chinese / Japanese / Korean।** Single characters high-frequency हो सकते हैं; आप lots of merges चाहते हो।
- **Inference के दौरान special tokens।** अगर user `<|im_end|>` containing string send करता है, naive tokenization उसे text की तरह treat करता है। vLLM जैसे frameworks user input safe रखने के लिए `add_special_tokens=False` offer करते हैं।
- **Token count != character count।** Characters में cost count करना ग़लत है; tokens में bill / measure करो।

---

## 10. एक Minimal BPE Encoder/Decoder Implement करना

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

इसे run करो। Real model को `"Hello world"` encode करते हुए देखो और merges trace करो। "Magic" गायब हो जाता है।

---

## 11. Embeddings + Position = Transformer के लिए Ready

Transformer block expect करता है shape `(B, T, D)`। Embedding आपको ये देता है। कई पुराने models भी इस stage पर एक **positional embedding** add करते थे:

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

2026 में, almost हर कोई **RoPE** (chapter 7) पर switch हो गया है, जो attention के *अंदर* apply होता है और embedding layer पर बिल्कुल नहीं। तो modern model का embedding बस `tok = self.embed(ids)` है।

---

## 12. एक 2026 Cheat Sheet

- **Byte-level BPE** use करो vocab **32k-256k** के साथ।
- Vocab choose करो languages और code share match करने के लिए। **Bigger vocab = shorter sequences = cheaper inference।**
- Train करने के लिए **`tokenizers` / `tiktoken`** use करो, consume करने के लिए **`AutoTokenizer`**।
- Small models के लिए **tied embeddings**, ≥7B के लिए untied।
- Single-digit tokenization math में help करती है।
- हमेशा कुछ tokenized examples print और inspect करो — bugs वहां छुपते हैं।
- Model के tokenizer को reuse करो; कभी mix-and-match मत करो।
- Special tokens respect deserve करते हैं। वो chat protocol define करते हैं।

---

## और गहराई से

- Karpathy का **`minbpe`** repo — 200 lines में scratch से BPE।
- Sennrich et al. 2016 — original BPE paper।
- Kudo & Richardson 2018 — SentencePiece (Unigram alternative अभी भी कुछ Asian-language tokenizers द्वारा used)।
- HuggingFace पर Llama 3 / Qwen 2.5 / DeepSeek tokenizer JSON files — उन्हें पढ़ो, वो surprisingly clear हैं।

Next: **[07-positional-encodings.md](./07-positional-encodings.md)** — model को बताना कि हर token कहां है।
