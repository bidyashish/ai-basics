# 21 · Embeddings & RAG — Generate करने से पहले Search करो

> **TL;DR** ज़्यादातर "हमारे docs पर model fine-tune करो" projects RAG projects होने चाहिए। अपने documents को vectors में **embed** करो, उन्हें vector DB में **store** करो, हर query के लिए relevant ones को **retrieve** करो, और LLM को **feed** करो। 2026 SOTA stack है: embedding के लिए **mxbai-embed-large / Voyage / Cohere / NVEmbed**, storage के लिए **Qdrant / pgvector / Turbopuffer**, retrieval के लिए **hybrid search (BM25 + vector)**, reranking के लिए **BGE-reranker-v2 या Cohere Rerank**, और hard questions के लिए **agentic RAG** (model को sub-queries issue करने दो)। ये chapter हर piece runnable code और production teams को bite करने वाले failure modes के साथ cover करता है।

---

## 1. RAG क्यों, Fine-tuning नहीं?

तीन reasons लोग fine-tuning के बजाय RAG के लिए reach करते हैं:

1. **Retraining के बिना knowledge updates।** आज add किया गया एक new doc आज searchable है। एक fine-tune को re-train करना है।
2. **Citations।** RAG user को दिखा सकता है कि answer *किस* source से आया। Fine-tuning नहीं कर सकता।
3. **Smaller, cheaper models work करते हैं।** A 4B model + good retrieval often factual questions पर no retrieval वाले 27B model को beat करता है।

Exception: जब knowledge **how to behave** (एक tone, एक workflow, एक domain-specific format) हो, fine-tune करो। जब knowledge **what is true** (facts, docs, code) हो, RAG use करो।

---

## 2. Embeddings — ये Actually क्या हैं

एक **embedding** text (या images, या audio) को एक fixed-length vector पर map करता है ऐसे कि *semantically similar inputs nearby vectors पर map होते हैं*। Vector space में distance ≈ semantic similarity।

```python
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

vec = m.encode("How do I reset my password?")
print(vec.shape)   # (1024,)
```

Similar meaning वाले दो queries को similar vectors मिलते हैं:

```python
import numpy as np

a = m.encode("How do I reset my password?")
b = m.encode("I forgot my login.")
c = m.encode("What's the weather today?")

cos = lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
print(cos(a, b))   # ~0.85 — close
print(cos(a, c))   # ~0.20 — far
```

**Cosine similarity** standard distance metric है। Higher = more similar। Most embedding models **L2-normalized** vectors output करते हैं तो cosine similarity dot product के equal होती है।

### Embedding Model के दो Flavors

- **Symmetric / sentence**: query और document same encoder use करते हैं। Clustering, deduplication, retrieval जहां inputs similar दिखते हैं (paragraph ↔ paragraph) के लिए good।
- **Asymmetric / retrieval**: query और document different prompts use करते हैं ("Represent this query for retrieval:" vs "Represent this document:")। Typical RAG के लिए better जहां queries short और documents long होते हैं। **हमेशा model card check करो** — most modern retrieval models asymmetric हैं और query vs doc के लिए different prefixes चाहते हैं।

```python
# E5 / BGE / mxbai style — asymmetric
docs = m.encode(documents, prompt_name="passage")    # या "Represent this passage:"
qvec = m.encode(query,     prompt_name="query")      # या "Represent this query:"
```

Prompt भूलना 5-15% retrieval quality cost करता है। **Common bug।**

---

## 3. 2026 Embedding Model Shortlist

Late 2026 में leaderboard (MTEB) इन वालों से dominated है:

| Model | Size | Dim | Score (avg MTEB) | Notes |
|-------|------|-----|------------------|-------|
| **NVEmbed-v2** | ~7B | 4096 | 72-74 | top open-source, big & slow |
| **bge-en-icl / bge-m3** | 568M | 1024 | 70-71 | multi-functional (dense + sparse + multi-vector) |
| **mxbai-embed-large-v1** | 335M | 1024 | ~67 | strong open default, Apache 2.0 |
| **gte-Qwen2-7B-instruct** | 7B | 3584 | 70-71 | instruct-tunable, big |
| **e5-mistral-7B-instruct** | 7B | 4096 | 67-69 | older but battle-tested |
| **nomic-embed-text-v1.5** | 137M | 64-768 (Matryoshka) | 62-65 | tiny + Matryoshka truncation |
| **OpenAI text-embedding-3-large** | API | 3072 | 64-66 | popular default; 256 तक truncatable |
| **Voyage-3 / Voyage-large-2** | API | 1024-2048 | 70-72 | top-tier paid; Anthropic recommends |
| **Cohere Embed-v4** | API | 1024 | 70-72 | top-tier paid; 100+ languages |
| **gemini-embedding-001** | API | 768/1536/3072 | 71-73 | Google's, late 2026 में MTEB top पर |

Self-host के लिए: **mxbai-embed-large-v1** या **bge-m3** sane default है। अगर memory tight हो तो `nomic-embed` पर drop करो; अगर एक GPU spare कर सकते हो तो **NVEmbed-v2** पर jump करो।

API के लिए: **Voyage**, **Cohere**, या **Gemini Embedding** roughly top पर tied हैं।

### Matryoshka Embeddings

Models जैसे `nomic-embed-v1.5`, `mxbai-embed-large-v1`, और `text-embedding-3-large` ऐसे trained हैं कि vector को **truncate** करना अभी भी काम करता है — आप एक fast index के लिए first 256 dims use कर सकते हो, फिर full 1024 के साथ candidates re-rank करो। Storage 4× cut करता है minimal quality loss के साथ। **2026 में default by Matryoshka use करो।**

---

## 4. Vector Databases — Storage Layer

एक बार आपके पास vectors हो जाएं तो आपको "K nearest" fast query करना है। वो vector DB है।

| DB | Type | Notable | कब use करें |
|----|------|---------|-------------|
| **Qdrant** | dedicated, Rust | fast, filters, hybrid, on-prem ready | self-hosted production default |
| **pgvector** (Postgres) | extension | existing Postgres use करो | mixed relational + vector data |
| **Turbopuffer** | managed, S3-backed | cheap, billions तक scales | जब storage cost dominate करे |
| **Pinecone** | managed | mature, hybrid, namespaces | quick start, no ops |
| **Weaviate** | dedicated, Go | hybrid, modules | Java/Go shops |
| **Milvus / Zilliz** | dedicated | huge scales | enterprise, billions of vectors |
| **LanceDB** | embedded | parquet-based, in-process | small datasets, easy |
| **Chroma** | embedded | dev-friendly | prototypes |
| **Vespa** | search engine | best-in-class hybrid | जब SEO/ranking matter करे |

A 2026 production startup के लिए: cost के लिए **Qdrant self-hosted या Turbopuffer managed**, अगर आप already Postgres use करते हो तो **pgvector**, fastest time-to-launch के लिए **Pinecone**।

### Indexing: HNSW vs IVF vs SQ

- **Flat (brute force)**: exact, per query O(N)। ~100k vectors तक fine।
- **HNSW**: hierarchical small-world graph; ~100k-100M scale के लिए default। flat vs <10× slowdown पर ~98% recall।
- **IVF (inverted file)**: vectors cluster करो, clusters search करो। Lower memory, slightly worse recall।
- **Product Quantization (PQ) / Scalar Quantization (SQ)**: vectors को 8 / 4 / 1 bit per dim पर compress करो। 4-8× memory reduction, small recall loss।
- **DiskANN**: HNSW variant on disk; billion-scale के लिए।

2026 में, **HNSW + scalar quantization (int8)** default है। **Binary quantization** (1 bit per dim) very large indices के लिए ground gain कर रहा है — ~95% recall पर 32× memory cut।

---

## 5. Chunking — Unglamorous Lever

एक doc retrieve करने से पहले, आपको उसे **chunk** करना है: pieces में split करो small enough to fit context but big enough to make sense।

### Strategies

- **Fixed-size**: per chunk 256-512 tokens, 10-20% overlap। Simple, prose के लिए fine।
- **Recursive character splitter** (LangChain default): paragraphs पर split, फिर sentences, फिर characters। Better।
- **Semantic chunking**: sentences embed करो, similarity से group करो। Slow लेकिन heterogenous docs पर excellent।
- **Structure-aware**: markdown headings, code blocks, table boundaries respect करो। Technical docs के लिए best।
- **Late chunking** (Jina 2024): *whole document* एक pass में embed करो, फिर बाद में chunk करो। Better contextual embeddings; long-context embedders चाहिए।

### Practical Rules

- Most use cases के लिए **target chunk size: 200-500 tokens**।
- **Overlap: 10-20%** ताकि chunks के across split हुए concepts lost न हों।
- हर chunk में **structural context add करो**: title, section heading, doc URL। Model को जानना चाहिए ये कहां से आया।
- **Code के लिए: lines से नहीं function / class से chunk करो**।
- **Tables के लिए**: अगर possible हो तो whole रखो; otherwise header context के साथ row-by-row embed करो।

A chunk-quality eval often fancier retrieval को beat करता है: **labeled query set पर recall@k measure करो** different chunkers के साथ। Differences huge हैं।

---

## 6. Retrieval Pipelines

A modern (2026) retrieval pipeline में 3-5 stages हैं।

```
       query
         │
         ▼
   ┌──────────────┐  query rewriting (optional)
   │ rewrite/     │  - acronyms expand
   │ decompose    │  - HyDE
   │              │  - multi-query
   └──────┬───────┘
          ▼
   ┌──────────────┐
   │ first-stage  │  parallel:
   │  retrieval   │  - dense (vector)
   │              │  - sparse (BM25)
   └──────┬───────┘
          ▼
   ┌──────────────┐
   │  fusion      │  RRF (reciprocal rank fusion)
   │  (k=100)     │  dense + sparse combine करता है
   └──────┬───────┘
          ▼
   ┌──────────────┐
   │   reranker   │  cross-encoder
   │  (k → 8)     │  (query, doc) jointly scores
   └──────┬───────┘
          ▼
   ┌──────────────┐
   │  generation  │  retrieved context के साथ LLM
   └──────────────┘
```

### Hybrid Search: BM25 + Vector

Pure vector search **exact-match** queries (product codes, names, error strings) miss करती है। Pure BM25 **paraphrases** miss करती है। उन्हें combine करो।

```python
# Qdrant के साथ pseudo
dense_hits  = qdrant.search(vector=qvec, limit=100)
sparse_hits = qdrant.search(sparse=bm25(query), limit=100)
# Reciprocal Rank Fusion
def rrf(hits_a, hits_b, k=60):
    scores = {}
    for i, h in enumerate(hits_a): scores[h.id] = scores.get(h.id, 0) + 1/(k+i)
    for i, h in enumerate(hits_b): scores[h.id] = scores.get(h.id, 0) + 1/(k+i)
    return sorted(scores, key=scores.get, reverse=True)[:50]
```

**Hybrid almost always real-world data पर either alone को beat करता है।** Most 2026 vector DBs (Qdrant, Weaviate, Pinecone, `paradedb` के साथ pgvector) hybrid built in के साथ ship करते हैं।

### Cross-encoder के साथ Reranking

Dense + sparse retrieval ~50-100 candidates देने के बाद, उन्हें एक **cross-encoder** से rerank करो: एक smaller model जो `(query, document)` jointly लेता है और उन्हें score करता है। Dot-product से far ज़्यादा accurate, लेकिन slower — इसलिए आप इसे सिर्फ़ candidates पर run करते हो।

```python
from sentence_transformers import CrossEncoder
ce = CrossEncoder("BAAI/bge-reranker-v2-m3")

scores = ce.predict([(query, doc) for doc in candidates])
top_k = [d for _, d in sorted(zip(scores, candidates), reverse=True)][:8]
```

Worth knowing 2026 rerankers: **bge-reranker-v2-m3**, **Cohere Rerank-3**, **mixedbread mxbai-rerank-large-v2**, **Voyage Rerank-2**, **gemini-rerank**। Reranking recall@8 को reliably 5-15 points से move करता है।

### HyDE — Hypothetical Document Embedding

Tricky queries के लिए, LLM से ask करो *imagine* करने के लिए कि answer कैसा हो सकता है, *उसे* embed करो, और imagined answer के close documents retrieve करो:

```python
hypothetical = llm("इसके लिए 3-sentence answer लिखो: " + query)
qvec = embed(hypothetical)
docs = vector_db.search(qvec, k=8)
```

Often query directly embedding से beat करता है। काम करता है क्योंकि answers और answers questions और answers से ज़्यादा tightly cluster करते हैं।

---

## 7. Agentic RAG — Model को Search करने दो

Straightforward "embed → retrieve → generate" simple questions के लिए काम करती है। Multi-hop questions ("इन दो reports से X और Y compare करो") के लिए, single retrieval miss करती है। **Agentic RAG** model को retrieval को एक *tool* के रूप में call करने और multiple sub-queries issue करने देता है।

```python
tools = [search_docs, fetch_url, query_db]
agent_loop("2025 में Apple और Samsung का revenue growth compare करो।", tools)
```

Agent जो steps ले सकता है:

1. `search_docs("Apple revenue 2025")` call करो।
2. Result पढ़ो; number extract करो।
3. `search_docs("Samsung revenue 2025")` call करो।
4. Result पढ़ो; number extract करो।
5. Compute, summarize, citations के साथ answer।

Frameworks: **LangGraph**, **LlamaIndex agentic query engines**, **CrewAI**, custom harnesses (chapter 18)। Complex retrieval tasks के लिए **agentic RAG answer quality double कर सकती है** 3-5× token cost पर।

---

## 8. इसे Together रखना — एक Complete RAG ~80 lines में

```python
"""
End-to-end RAG hybrid search + reranking के साथ।
Requires: sentence-transformers, qdrant-client, anthropic, rank-bm25
"""
import os, json
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi
import anthropic

# ─── Setup ──────────────────────────────────────────────────────────────────
embedder = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
qd = QdrantClient(":memory:")
qd.create_collection(
    "docs",
    vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
)
client = anthropic.Anthropic()

# ─── Indexing ───────────────────────────────────────────────────────────────
def index(documents):
    """documents: list of {id, text, metadata}"""
    embs = embedder.encode([d["text"] for d in documents],
                           prompt_name="passage",
                           normalize_embeddings=True, batch_size=32)
    qd.upsert(
        collection_name="docs",
        points=[models.PointStruct(id=d["id"], vector=v.tolist(),
                                    payload=d) for d, v in zip(documents, embs)],
    )
    # BM25 भी build करो
    global bm25, doc_store
    doc_store = {d["id"]: d for d in documents}
    bm25 = BM25Okapi([d["text"].split() for d in documents])
    bm25.ids = [d["id"] for d in documents]

# ─── Retrieval ──────────────────────────────────────────────────────────────
def retrieve(query, k=8):
    qvec = embedder.encode(query, prompt_name="query",
                           normalize_embeddings=True)
    dense = qd.search("docs", qvec.tolist(), limit=50)
    sparse_scores = bm25.get_scores(query.split())
    sparse = sorted(zip(bm25.ids, sparse_scores), key=lambda x: -x[1])[:50]

    # RRF fusion
    scores = {}
    for i, h in enumerate(dense):     scores[h.id] = scores.get(h.id, 0) + 1/(60+i)
    for i, (id, _) in enumerate(sparse): scores[id] = scores.get(id, 0) + 1/(60+i)
    fused_ids = sorted(scores, key=scores.get, reverse=True)[:50]
    candidates = [doc_store[i] for i in fused_ids]

    # Rerank
    pairs = [(query, c["text"]) for c in candidates]
    rscores = reranker.predict(pairs)
    top = [c for _, c in sorted(zip(rscores, candidates),
                                 key=lambda x: -x[0])][:k]
    return top

# ─── Generate ───────────────────────────────────────────────────────────────
def answer(query, k=8):
    docs = retrieve(query, k)
    ctx = "\n\n".join(f"[{d['id']}] {d['text']}" for d in docs)
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=("Answer using only the documents provided. "
                "Cite using bracketed IDs like [doc-12]."),
        messages=[{"role":"user",
                   "content": f"<docs>\n{ctx}\n</docs>\n\nQuestion: {query}"}],
    )
    return resp.content[0].text, [d["id"] for d in docs]

if __name__ == "__main__":
    docs = [json.loads(l) for l in open("docs.jsonl")]   # {id, text}
    index(docs)
    out, sources = answer("How do I reset my password?")
    print(out, "\n\nSources:", sources)
```

ये एक working production-grade pattern है। Real deployment के लिए streaming, caching, async, और observability add करो।

---

## 9. RAG Evaluating

दो layers:

### Retrieval Quality
A labeled set of `(query, relevant_doc_ids)` पर:
- **Recall@k**: top k में relevant docs की fraction।
- **MRR**: first relevant doc की reciprocal rank।
- **NDCG**: discounted cumulative gain (preferred जब relevance graded हो)।

### End-to-end Quality
- **Faithfulness**: क्या answer सिर्फ़ retrieved context use करता है? (LLM judge)।
- **Answer correctness**: reference answers के against judge।
- **Citation accuracy**: क्या cited sources actually quoted/used हैं?
- **Refusal correctness**: जब answer docs में नहीं है, क्या model कहता है?

Tools: **RAGAS**, **LangSmith RAG evals**, **TruLens**, **Phoenix RAG evals**। चारों इन metrics को box के out implement करते हैं।

A common production pattern: weekly recall@8 track करो; अगर drop हो, users notice करने से पहले investigate करो।

---

## 10. Common RAG Failure Modes

- **Language के लिए wrong embedder**: most public models English-heavy हैं। Multilingual के लिए, `bge-m3` या Cohere Embed-v4 या jina-embeddings-v3 use करो।
- **Symmetric model asymmetrically used** (query/passage prompts भूले) — silent 5-15% recall hit।
- **Chunks too long**: model focus नहीं कर सकता; cite-ability drops।
- **Chunks too short**: no context; meaning lost।
- **No reranker**: top-k retrieval noisy है; reranker most fix करता है।
- **No hybrid**: exact-match queries (codes, names) miss करता है।
- **Stale index**: docs change, embeddings नहीं।
- **Non-deterministic retrieval order**: same query run-to-run different docs देती है अगर आप sort नहीं करते।
- **Authoritative context का loss**: forums vs official docs से docs equally treated। Weights add करो।
- **Citation hallucination**: model एक doc cite करने का claim करता है जो वो नहीं कहता जो ये claim करता है। "Relevant sentence quote करो" prompts के साथ mitigate करो।
- **Lost-in-the-middle**: long context पर, models start और end पर ज़्यादा attend करते हैं middle से। Most relevant chunks को top *या* bottom पर रखो।

---

## 11. RAG का Cost

1M queries / month के लिए hybrid + reranking + Sonnet generation के साथ:

- Embedding cost (cache friendly): ~$10-50 / month।
- Vector DB: hosted के लिए ~$50-500 / month, self-hosted के लिए ~$0।
- Reranker: self-hosted के लिए ~$0, API के लिए ~$100-500।
- Generation: dominant। ~$300-2000 / month।

Fine-tune-only approach से compare करो: ~$50-200 fine-tune compute one-time, ~$300-2000 / month inference। RAG ops cost में **few % add** करता है लेकिन fine-tune cost **बचाता** है और instantly updates। Cost picture knowledge-heavy products के लिए लगभग हमेशा RAG को favor करती है।

---

## 12. Multi-modal Embeddings

Late 2025-2026: text के साथ paired **CLIP-style** image embeddings:

- **CLIP / SigLIP / OpenCLIP** — classic, image+text shared space।
- **Cohere Embed-v4** — एक model में image और text support करता है।
- **Voyage multimodal-3** — image + text embedding।
- **NVIDIA NV-CLIP / NV-DINOv3** — strong image encoders।
- **`Marqo`, `LanceDB`, `Qdrant`** — multimodal-aware vector DBs।

Used for: image search, document understanding (PDF + diagram + text), e-commerce ("इस जैसे products find करो")। Retrieval pipeline same है; बस दोनों modalities को shared space में embed करो।

---

## 13. 2026 Cheat Sheet

- Knowledge के लिए **RAG > fine-tune**।
- Default embedder: **mxbai-embed-large-v1** (open) या **Voyage / Cohere / Gemini Embedding** (API)।
- **Default by Matryoshka use करो**; fast first pass के लिए 256 तक truncate करो।
- **Hybrid search (BM25 + vector)**, RRF से fused। Almost हमेशा either alone को beat करता है।
- एक cross-encoder से **हमेशा rerank करो** (BGE-reranker-v2 या Cohere Rerank-3)।
- **200-500 tokens के chunks**, possible हो तो structure-aware।
- हर chunk के text में **doc title/heading add करो**।
- **Weekly recall@k eval**; ये user complaints predict करता है।
- Multi-hop के लिए **Agentic RAG**; 3-5× cost expect करो।
- Trust के लिए **Citations mandatory हैं**।
- **Lost-in-the-middle**: high-priority context start AND end पर रखो।

---

## और गहराई से

- **MTEB leaderboard** at `huggingface.co/spaces/mteb/leaderboard` — quarterly refresh करो।
- **`sentence-transformers`** docs।
- **Qdrant docs और reference architectures**।
- **Anthropic's "Contextual Retrieval"** post — important variant जहां हर chunk एक model-generated summary से augmented है।
- Eval के लिए **`RAGAS` और `Phoenix RAG`**।
- **Cohere's Rerank-3 / Voyage's Rerank-2** technical posts — modern reranking के best public writeups।
- **"Lost in the middle"** (Liu et al. 2023) — long-context RAG failure mode।
- **`flashrank`** — production use के लिए fast open reranker।

Next: **[22-multimodal.md](./22-multimodal.md)** — pictures, sound, और models उन्हें कैसे read करते हैं।
