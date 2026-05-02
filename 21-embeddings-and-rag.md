# 21 · Embeddings & RAG — Search Before You Generate

> **TL;DR** Most "fine-tune the model on our docs" projects should be RAG projects. **Embed** your documents into vectors, **store** them in a vector DB, **retrieve** the relevant ones for each query, and **feed** them to the LLM. The 2026 SOTA stack is: **mxbai-embed-large / Voyage / Cohere / NVEmbed** for embedding, **Qdrant / pgvector / Turbopuffer** for storage, **hybrid search (BM25 + vector)** for retrieval, **BGE-reranker-v2 or Cohere Rerank** for reranking, and **agentic RAG** (let the model issue sub-queries) for hard questions. This chapter covers each piece with runnable code and the failure modes that bite production teams.

---

## 1. Why RAG, not fine-tuning?

Three reasons people reach for RAG over fine-tuning:

1. **Knowledge updates without retraining.** A new doc added today is searchable today. A fine-tune has to re-train.
2. **Citations.** RAG can show the user *which* source the answer came from. Fine-tuning can't.
3. **Smaller, cheaper models work.** A 4B model + good retrieval often beats a 27B model with no retrieval on factual questions.

The exception: when the knowledge is **how to behave** (a tone, a workflow, a domain-specific format), fine-tune. When the knowledge is **what is true** (facts, docs, code), use RAG.

---

## 2. Embeddings — what they really are

An **embedding** maps text (or images, or audio) to a fixed-length vector such that *semantically similar inputs map to nearby vectors*. Distance in vector space ≈ semantic similarity.

```python
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

vec = m.encode("How do I reset my password?")
print(vec.shape)   # (1024,)
```

Two queries with similar meaning get similar vectors:

```python
import numpy as np

a = m.encode("How do I reset my password?")
b = m.encode("I forgot my login.")
c = m.encode("What's the weather today?")

cos = lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
print(cos(a, b))   # ~0.85 — close
print(cos(a, c))   # ~0.20 — far
```

**Cosine similarity** is the standard distance metric. Higher = more similar. Most embedding models output **L2-normalized** vectors so cosine similarity equals dot product.

### The two flavors of embedding model

- **Symmetric / sentence**: query and document use the same encoder. Good for clustering, deduplication, retrieval where the inputs look similar (paragraph ↔ paragraph).
- **Asymmetric / retrieval**: query and document use different prompts ("Represent this query for retrieval:" vs "Represent this document:"). Better for typical RAG where queries are short and documents are long. **Always check the model card** — most modern retrieval models are asymmetric and want different prefixes for query vs doc.

```python
# E5 / BGE / mxbai style — asymmetric
docs = m.encode(documents, prompt_name="passage")    # or "Represent this passage:"
qvec = m.encode(query,     prompt_name="query")      # or "Represent this query:"
```

Forgetting the prompt costs 5-15% retrieval quality. **Common bug.**

---

## 3. The 2026 embedding model shortlist

In late 2026 the leaderboard (MTEB) is dominated by these:

| Model | Size | Dim | Score (avg MTEB) | Notes |
|-------|------|-----|------------------|-------|
| **NVEmbed-v2** | ~7B | 4096 | 72-74 | top open-source, big & slow |
| **bge-en-icl / bge-m3** | 568M | 1024 | 70-71 | multi-functional (dense + sparse + multi-vector) |
| **mxbai-embed-large-v1** | 335M | 1024 | ~67 | strong open default, Apache 2.0 |
| **gte-Qwen2-7B-instruct** | 7B | 3584 | 70-71 | instruct-tunable, big |
| **e5-mistral-7B-instruct** | 7B | 4096 | 67-69 | older but battle-tested |
| **nomic-embed-text-v1.5** | 137M | 64-768 (Matryoshka) | 62-65 | tiny + Matryoshka truncation |
| **OpenAI text-embedding-3-large** | API | 3072 | 64-66 | popular default; truncatable to 256 |
| **Voyage-3 / Voyage-large-2** | API | 1024-2048 | 70-72 | top-tier paid; Anthropic recommends |
| **Cohere Embed-v4** | API | 1024 | 70-72 | top-tier paid; 100+ languages |
| **gemini-embedding-001** | API | 768/1536/3072 | 71-73 | Google's, top of MTEB late 2026 |

For self-host: **mxbai-embed-large-v1** or **bge-m3** is the sane default. Drop to `nomic-embed` if memory is tight; jump to **NVEmbed-v2** if you can spare a GPU.

For API: **Voyage**, **Cohere**, or **Gemini Embedding** are roughly tied at the top.

### Matryoshka embeddings

Models like `nomic-embed-v1.5`, `mxbai-embed-large-v1`, and `text-embedding-3-large` are trained so that **truncating** the vector still works — you can use the first 256 dims for a fast index, then re-rank candidates with the full 1024. Cuts storage 4× with minimal quality loss. **Use Matryoshka by default in 2026.**

---

## 4. Vector databases — the storage layer

Once you have vectors you need to query "give me the K nearest" fast. That's a vector DB.

| DB | Type | Notable | When to use |
|----|------|---------|-------------|
| **Qdrant** | dedicated, Rust | fast, filters, hybrid, on-prem ready | self-hosted production default |
| **pgvector** (Postgres) | extension | use existing Postgres | mixed relational + vector data |
| **Turbopuffer** | managed, S3-backed | cheap, scales to billions | when storage cost dominates |
| **Pinecone** | managed | mature, hybrid, namespaces | quick start, no ops |
| **Weaviate** | dedicated, Go | hybrid, modules | Java/Go shops |
| **Milvus / Zilliz** | dedicated | scales huge | enterprise, billions of vectors |
| **LanceDB** | embedded | parquet-based, in-process | small datasets, easy |
| **Chroma** | embedded | dev-friendly | prototypes |
| **Vespa** | search engine | best-in-class hybrid | when SEO/ranking matters |

For a 2026 production startup: **Qdrant self-hosted or Turbopuffer managed** for cost, **pgvector** if you already use Postgres, **Pinecone** for fastest time-to-launch.

### Indexing: HNSW vs IVF vs SQ

- **Flat (brute force)**: exact, O(N) per query. Fine up to ~100k vectors.
- **HNSW**: hierarchical small-world graph; the default for ~100k-100M scale. ~98% recall at <10× slowdown vs flat.
- **IVF (inverted file)**: cluster vectors, search clusters. Lower memory, slightly worse recall.
- **Product Quantization (PQ) / Scalar Quantization (SQ)**: compress vectors to 8 / 4 / 1 bit per dim. 4-8× memory reduction, small recall loss.
- **DiskANN**: HNSW variant on disk; for billion-scale.

In 2026, **HNSW + scalar quantization (int8)** is the default. **Binary quantization** (1 bit per dim) is gaining ground for very large indices — 32× memory cut at ~95% recall.

---

## 5. Chunking — the unglamorous lever

Before you can retrieve a doc, you have to **chunk** it: split into pieces small enough to fit context but big enough to make sense.

### Strategies

- **Fixed-size**: 256-512 tokens per chunk, 10-20% overlap. Simple, fine for prose.
- **Recursive character splitter** (LangChain default): split on paragraphs, then sentences, then characters. Better.
- **Semantic chunking**: embed sentences, group by similarity. Slow but excellent on heterogenous docs.
- **Structure-aware**: respect markdown headings, code blocks, table boundaries. Best for technical docs.
- **Late chunking** (Jina 2024): embed the *whole document* in one pass, then chunk afterwards. Better contextual embeddings; needs long-context embedders.

### Practical rules

- **Target chunk size: 200-500 tokens** for most use cases.
- **Overlap: 10-20%** so concepts split across chunks aren't lost.
- **Add structural context** to each chunk: title, section heading, doc URL. The model needs to know where this came from.
- **For code: chunk by function / class**, not by lines.
- **For tables**: keep them whole if possible; otherwise embed row-by-row with header context.

A chunk-quality eval often beats fancier retrieval: **measure recall@k on a labeled query set** with different chunkers. The differences are huge.

---

## 6. Retrieval pipelines

A modern (2026) retrieval pipeline has 3-5 stages.

```
       query
         │
         ▼
   ┌──────────────┐  query rewriting (optional)
   │ rewrite/     │  - expand acronyms
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
   │  (k=100)     │  combines dense + sparse
   └──────┬───────┘
          ▼
   ┌──────────────┐
   │   reranker   │  cross-encoder
   │  (k → 8)     │  scores (query, doc) jointly
   └──────┬───────┘
          ▼
   ┌──────────────┐
   │  generation  │  LLM with retrieved context
   └──────────────┘
```

### Hybrid search: BM25 + vector

Pure vector search misses **exact-match** queries (product codes, names, error strings). Pure BM25 misses **paraphrases**. Combine them.

```python
# pseudo with Qdrant
dense_hits  = qdrant.search(vector=qvec, limit=100)
sparse_hits = qdrant.search(sparse=bm25(query), limit=100)
# Reciprocal Rank Fusion
def rrf(hits_a, hits_b, k=60):
    scores = {}
    for i, h in enumerate(hits_a): scores[h.id] = scores.get(h.id, 0) + 1/(k+i)
    for i, h in enumerate(hits_b): scores[h.id] = scores.get(h.id, 0) + 1/(k+i)
    return sorted(scores, key=scores.get, reverse=True)[:50]
```

**Hybrid almost always beats either alone on real-world data.** Most 2026 vector DBs (Qdrant, Weaviate, Pinecone, pgvector with `paradedb`) ship with hybrid built in.

### Reranking with a cross-encoder

After dense + sparse retrieval gives you ~50-100 candidates, rerank them with a **cross-encoder**: a smaller model that takes `(query, document)` jointly and scores them. Far more accurate than dot-product, but slower — that's why you only run it on candidates.

```python
from sentence_transformers import CrossEncoder
ce = CrossEncoder("BAAI/bge-reranker-v2-m3")

scores = ce.predict([(query, doc) for doc in candidates])
top_k = [d for _, d in sorted(zip(scores, candidates), reverse=True)][:8]
```

2026 rerankers worth knowing: **bge-reranker-v2-m3**, **Cohere Rerank-3**, **mixedbread mxbai-rerank-large-v2**, **Voyage Rerank-2**, **gemini-rerank**. Reranking moves recall@8 by 5-15 points reliably.

### HyDE — Hypothetical Document Embedding

For tricky queries, ask the LLM to *imagine* what the answer might look like, embed *that*, and retrieve documents close to the imagined answer:

```python
hypothetical = llm("Write a 3-sentence answer to: " + query)
qvec = embed(hypothetical)
docs = vector_db.search(qvec, k=8)
```

Often beats embedding the query directly. Works because answers and answers cluster more tightly than questions and answers.

---

## 7. Agentic RAG — let the model search

The straightforward "embed → retrieve → generate" works for simple questions. For multi-hop questions ("compare X and Y from these two reports"), a single retrieval misses. **Agentic RAG** lets the model call retrieval as a *tool* and issue multiple sub-queries.

```python
tools = [search_docs, fetch_url, query_db]
agent_loop("Compare revenue growth of Apple and Samsung in 2025.", tools)
```

Steps the agent might take:

1. Call `search_docs("Apple revenue 2025")`.
2. Read result; extract number.
3. Call `search_docs("Samsung revenue 2025")`.
4. Read result; extract number.
5. Compute, summarize, answer with citations.

Frameworks: **LangGraph**, **LlamaIndex agentic query engines**, **CrewAI**, custom harnesses (chapter 18). For complex retrieval tasks **agentic RAG can double answer quality** at 3-5× the token cost.

---

## 8. Putting it together — a complete RAG in ~80 lines

```python
"""
End-to-end RAG with hybrid search + reranking.
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
    # also build BM25
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

That's a working production-grade pattern. Add streaming, caching, async, and observability for real deployment.

---

## 9. Evaluating RAG

Two layers:

### Retrieval quality
On a labeled set of `(query, relevant_doc_ids)`:
- **Recall@k**: fraction of relevant docs in top k.
- **MRR**: reciprocal rank of first relevant doc.
- **NDCG**: discounted cumulative gain (preferred when relevance is graded).

### End-to-end quality
- **Faithfulness**: does the answer use only the retrieved context? (LLM judge.)
- **Answer correctness**: judge against reference answers.
- **Citation accuracy**: are cited sources actually quoted/used?
- **Refusal correctness**: when the answer isn't in the docs, does the model say so?

Tools: **RAGAS**, **LangSmith RAG evals**, **TruLens**, **Phoenix RAG evals**. All four implement these metrics out of the box.

A common production pattern: track recall@8 weekly; if it drops, investigate before users notice.

---

## 10. Common RAG failure modes

- **Wrong embedder for the language**: most public models are English-heavy. For multilingual, use `bge-m3` or Cohere Embed-v4 or jina-embeddings-v3.
- **Symmetric model used asymmetrically** (forgot query/passage prompts) — silent 5-15% recall hit.
- **Chunks too long**: model can't focus; cite-ability drops.
- **Chunks too short**: no context; meaning lost.
- **No reranker**: top-k retrieval is noisy; reranker fixes most of it.
- **No hybrid**: misses exact-match queries (codes, names).
- **Stale index**: docs change, embeddings don't update.
- **Non-deterministic retrieval order**: same query gives different docs run-to-run if you don't sort.
- **Loss of authoritative context**: docs from forums vs official docs treated equally. Add weights.
- **Citation hallucination**: the model claims to cite a doc that doesn't say what it claims. Mitigate with explicit "quote the relevant sentence" prompts.
- **Lost-in-the-middle**: at long context, models attend to start and end more than middle. Put the most relevant chunks at top *or* bottom of the context block.

---

## 11. The cost of RAG

For 1M queries / month with hybrid + reranking + Sonnet generation:

- Embedding cost (cache friendly): ~$10-50 / month.
- Vector DB: ~$50-500 / month for hosted, ~$0 self-hosted.
- Reranker: ~$0 if self-hosted, ~$100-500 if API.
- Generation: dominant. ~$300-2000 / month.

Compare to fine-tune-only approach: ~$50-200 fine-tune compute one-time, ~$300-2000 / month inference. RAG **adds** a few % ops cost but **saves** the fine-tune cost and updates instantly. The cost picture favors RAG nearly always for knowledge-heavy products.

---

## 12. Multi-modal embeddings

Late 2025-2026: **CLIP-style** image embeddings paired with text:

- **CLIP / SigLIP / OpenCLIP** — classic, image+text shared space.
- **Cohere Embed-v4** — supports image and text in one model.
- **Voyage multimodal-3** — image + text embedding.
- **NVIDIA NV-CLIP / NV-DINOv3** — strong image encoders.
- **`Marqo`, `LanceDB`, `Qdrant`** — multimodal-aware vector DBs.

Used for: image search, document understanding (PDF + diagram + text), e-commerce ("find products like this"). The retrieval pipeline is the same; just embed both modalities into a shared space.

---

## 13. The 2026 cheat sheet

- **RAG > fine-tune** for knowledge.
- Default embedder: **mxbai-embed-large-v1** (open) or **Voyage / Cohere / Gemini Embedding** (API).
- **Use Matryoshka** by default; truncate to 256 for fast first pass.
- **Hybrid search (BM25 + vector)**, fused with RRF. Almost always beats either alone.
- **Always rerank** with a cross-encoder (BGE-reranker-v2 or Cohere Rerank-3).
- **Chunks of 200-500 tokens**, structure-aware where possible.
- **Add doc title/heading** to every chunk's text.
- **Eval recall@k weekly**; it predicts user complaints.
- **Agentic RAG** for multi-hop; expect 3-5× cost.
- **Citations are mandatory** for trust.
- **Lost-in-the-middle**: put high-priority context at start AND end.

---

## Going deeper

- **MTEB leaderboard** at `huggingface.co/spaces/mteb/leaderboard` — refresh quarterly.
- **`sentence-transformers`** docs.
- **Qdrant docs and reference architectures**.
- **Anthropic's "Contextual Retrieval"** post — important variant where each chunk is augmented with a model-generated summary.
- **`RAGAS` and `Phoenix RAG`** for eval.
- **Cohere's Rerank-3 / Voyage's Rerank-2** technical posts — best public writeups of modern reranking.
- **"Lost in the middle"** (Liu et al. 2023) — long-context RAG failure mode.
- **`flashrank`** — fast open reranker for production use.

Next: **[22-multimodal.md](./22-multimodal.md)** — pictures, sound, and how models read them.
