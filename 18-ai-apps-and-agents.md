# 18 · How Users Use AI — Apps, Harnesses, and Agents

> **TL;DR** A model is roughly **5%** of an AI product. The remaining 95% is the **harness**: tool registry, memory, prompt templates, retries, permissioning, observability, and the UX wrapped around them. In 2026 the high-leverage skill is *not* training — it's building this harness well. This chapter covers the twelve canonical usage patterns, the **MCP** standard for tool wiring, real-world harnesses (Cursor, Claude Code, Cline, Devin, Aider, LangGraph), and a **complete coding-agent harness in ~300 lines** of Python you can read and modify. Plus a full provider pricing table, a self-hosted vs API decision framework, and per-pattern cost math.

---

## 1. The layers above the model

If you only see "user types message → model replies," you're missing four layers in between.

```
                ┌──────────────────────────┐
                │  UI / IDE / Voice / Bot  │  what the user touches
                ├──────────────────────────┤
                │   Orchestrator / Agent   │  multi-step planning, graph
                ├──────────────────────────┤
                │      Harness             │  tools, memory, retries,
                │                          │  prompts, permissions, logs
                ├──────────────────────────┤
                │   Inference engine       │  vLLM, SGLang, API client
                ├──────────────────────────┤
                │   Model weights          │  Qwen, Llama, Claude, GPT
                └──────────────────────────┘
```

The four layers above the model are where *all* product differentiation happens. Two products using the same model can feel completely different because of harness choices.

---

## 2. The 12 canonical usage patterns

Every AI product is one of these — or a composition. Pick the right one before designing anything else.

| # | Pattern | One-line goal | Best fit |
|---|---------|---------------|---------|
| 1 | **Single-shot completion** | Take a prompt, return one answer | classification, simple Q&A |
| 2 | **Streaming chat** | Multi-turn conversation | ChatGPT-style assistants |
| 3 | **Tool-using agent** | Model picks tools to call | function-calling APIs |
| 4 | **ReAct loop** | Think → act → observe → think | autonomous reasoning |
| 5 | **Plan-then-execute** | Plan upfront, then run steps | long tasks with structure |
| 6 | **Multi-agent (orchestrator + workers)** | One model delegates to others | research, complex pipelines |
| 7 | **RAG** | Retrieve docs → feed to model | knowledge over private data |
| 8 | **Structured output** | Force JSON / schema-conforming reply | data extraction, APIs |
| 9 | **Code completion (ghost text)** | Tab-complete the next lines | IDE assistants |
| 10 | **Coding agent** | Autonomous file edits + tests | Claude Code, Cursor agent |
| 11 | **Voice assistant** | Speech in, speech out | mobile assistants |
| 12 | **Multimodal Q&A** | Image / video / audio in | analysis, accessibility |

Not on the list (deliberately): "general AI assistant that does everything." Generic is rarely a product. Pick a primary pattern and a secondary one. **Stack only what you need.**

---

## 3. The harness — what it actually is

Every serious AI app has a **harness**. It's the code between "user said X" and "model returns Y" that makes the system robust. A harness has these subsystems:

### 3.1 Tool registry

Tools are external functions the model can invoke: `read_file`, `run_shell`, `web_search`, `query_db`, `send_email`, `book_flight`. The registry holds:

- The function pointer.
- A JSON schema describing arguments.
- Permissions (who can call this, in what context).
- Side-effect class (read-only, mutating, network, irreversible).
- Cost tracking (LLM-callable tools that themselves call paid APIs).

```python
# minimal tool registry
@dataclass
class Tool:
    name: str
    fn: Callable
    schema: dict           # JSON Schema for arguments
    side_effects: str      # "readonly", "filesystem", "network", "destructive"

class ToolRegistry:
    def __init__(self): self._tools: dict[str, Tool] = {}
    def register(self, t: Tool): self._tools[t.name] = t
    def schemas(self):
        return [{"name": t.name, "description": t.fn.__doc__,
                 "parameters": t.schema} for t in self._tools.values()]
    def call(self, name: str, args: dict):
        return self._tools[name].fn(**args)
```

### 3.2 Memory

Short-term: the conversation buffer (recent N messages, possibly compacted). Long-term: vector DB or key-value store for facts the user told you in week 1 that matter today.

A real harness summarizes/compacts old turns when nearing the context window. **Don't blow context on conversations the model doesn't need to see verbatim.**

### 3.3 Prompt templates

Don't string-concatenate prompts inline. Centralize in versioned templates with named variables and a way to A/B test. Tools like `dotprompt`, `BAML`, `Promptfoo`, or your own Jinja-based system. Prompt drift is the silent killer of consistency.

### 3.4 Retry / fallback / circuit breaker

- Transient API failures → exponential backoff retry.
- Hard rate-limit → wait or fall back to a different provider.
- Bad output (failed JSON parse, refused, hallucinated) → retry with feedback or with a stricter model.
- Cascading failures → circuit-break the bad provider for 30 sec.

### 3.5 Permissions and sandboxing

Tool-using agents will execute code, hit URLs, modify files. **Always sandbox.** Run shell commands in a Docker container or a dedicated VM. Filesystem operations confined to a project directory. Network egress through a proxy with allowlists. Default-deny for destructive operations.

### 3.6 Observability

Per-request: which model, which tools, token counts, cost, cache hit rate, latency, errors. Per-session: total spend, tool-call distribution, error types. Per-version: prompt template versions, model versions, eval scores. **You will tune from this data daily for the first six months. Build it before launch.**

### 3.7 Output formatter

Most user-facing apps want clean output. Strip thinking tokens, format tool calls as collapsible blocks, render code-fenced output, handle partial JSON during streaming. This is more work than you'd think.

A useful slogan: **"the model is the brain, the harness is the body."** A great brain in a bad body still trips on stairs.

---

## 4. The Model Context Protocol (MCP)

In late 2024 Anthropic introduced **MCP** — the Model Context Protocol — a JSON-RPC standard for how models connect to tools and data sources. By 2026 it's the **lingua franca** for harness-tool wiring: Anthropic, OpenAI, Google, JetBrains, VS Code, Cursor, Claude Code, Cline, Continue, and all major IDE/IDE-like products either support it or are integrating it.

Why it matters: instead of every app re-implementing GitHub integration, Slack, your filesystem, Postgres, and a hundred other services, a single MCP server provides them once and any MCP-compatible host can use them.

### MCP architecture

```
   Host                    MCP server
 (Claude Code /          (filesystem,
  Cursor / your app)      github, postgres,
       │                  custom tools…)
       │  ──── tool list ────▶
       │  ◀─── schemas ──────
       │  ──── invoke ──────▶
       │  ◀─── result ──────
```

A typical setup runs **multiple MCP servers** in parallel — one for files, one for git, one for browser automation, one for your private API. The host enumerates tools across all of them.

### Spinning up an MCP server in 30 lines

```python
# pip install mcp
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-tools")

@mcp.tool()
def search_internal(query: str) -> str:
    """Search the internal knowledge base for a query."""
    return run_search(query)

@mcp.tool()
def file_read(path: str) -> str:
    """Read a file from the project directory."""
    return open(path).read()

if __name__ == "__main__":
    mcp.run()
```

Drop the path to that script into Claude Code, Cursor, or any MCP host's config and the tools appear immediately. **This is the cleanest abstraction the AI ecosystem has settled on.**

### Catalogues to plug in directly

- **Composio** — hundreds of pre-built tool integrations behind one MCP/REST surface.
- **Zapier MCP** — Zapier's 6,000+ integrations exposed as tools.
- **Anthropic Reference MCP** — filesystem, sqlite, github, browser, postgres reference implementations.
- **Smithery** — package registry for MCP servers.

For most teams: don't write MCP servers from scratch unless you have a private API. Compose existing ones.

---

## 5. The patterns in code

Each pattern below is shown with the smallest sensible PyTorch/Python implementation. Real production code adds error handling, observability, and timeouts on top.

### Pattern 1 — Single-shot completion

```python
import anthropic
client = anthropic.Anthropic()
resp = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=512,
    messages=[{"role": "user", "content": "Summarize: " + text}],
)
print(resp.content[0].text)
```

Use for: classification, extraction, transformation, single-turn Q&A. Stateless, easy to cache and parallelize.

### Pattern 2 — Streaming chat with memory

```python
history = []
def chat_turn(user_msg: str):
    history.append({"role": "user", "content": user_msg})
    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system="You are a helpful assistant.",
        messages=history,
    ) as stream:
        text = ""
        for chunk in stream.text_stream:
            print(chunk, end="", flush=True); text += chunk
    history.append({"role": "assistant", "content": text})
```

Trim or summarize `history` when it crosses a threshold (e.g., > 30 turns or > 50k tokens). The harness owns this.

### Pattern 3 — Tool-using agent (function calling)

```python
tools = [
    {"name": "get_weather",
     "description": "Get current weather in a city.",
     "input_schema": {"type": "object",
                      "properties": {"city": {"type": "string"}},
                      "required": ["city"]}},
]

resp = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "Weather in Tokyo?"}],
)
# resp.content might be a tool_use block — invoke it, append result, call again.
```

Function calling is the lowest level of tool use. The model returns a structured request to invoke a tool; you run it; you append the result and call again. Loop until the model says "done."

### Pattern 4 — ReAct (Reason + Act)

The model alternates between **Thought**, **Action**, and **Observation** tokens, then a final **Answer**. Pre-2024 papers used XML/text tags; post-MCP, ReAct is essentially "tool-calling with chain-of-thought." Prompts look like:

```
You are an agent. For each step:
- Think out loud what to do next.
- If you need a tool, call it.
- After the tool result, continue.
- When you have enough information, give the final answer.
```

Modern reasoning models (Claude Opus 4.7 with thinking, GPT-5 with reasoning, DeepSeek-R1) effectively bake ReAct into the model — you just give them tools and they reason internally before acting.

### Pattern 5 — Plan-then-execute

```python
plan = llm("Make a plan for: " + task, system=PLAN_PROMPT)  # bullet list
for step in parse_plan(plan):
    result = execute_step(step, llm, tools)
    if result.failed: result = replan(task, plan, step, result)
```

Useful when steps have **dependencies** (write a function, then test it, then refactor based on results). Less useful for free-form research where the next step depends on what you found.

### Pattern 6 — Multi-agent (orchestrator + workers)

```python
def orchestrator(task):
    subtasks = llm_decompose(task)               # split into N subtasks
    results = [worker_agent(s) for s in subtasks]   # parallel
    return llm_synthesize(task, results)
```

Frameworks: **LangGraph**, **CrewAI**, **AutoGen**, **OpenAI Agents SDK**, **Anthropic's parallel-tool-use API**. Big productivity boost when subtasks are independent (e.g., "summarize 50 documents"). Adds coordination overhead and cost when they aren't.

### Pattern 7 — RAG (Retrieval-Augmented Generation)

```python
def rag_answer(question):
    docs = vector_db.search(embed(question), k=8)   # cosine similarity
    context = "\n\n".join(doc.text for doc in docs)
    prompt = f"<docs>\n{context}\n</docs>\n\nQuestion: {question}"
    return llm(prompt)
```

Modern variants: **hybrid search** (semantic + BM25), **reranking** with a cross-encoder, **HyDE** (hypothetical doc embedding), **agentic RAG** where the model issues sub-queries. By 2026 most production RAG uses an explicit retrieval step + a reranker rather than naive top-k.

### Pattern 8 — Structured output

```python
schema = {
    "type": "object",
    "properties": {
        "title":    {"type": "string"},
        "tags":     {"type": "array", "items": {"type": "string"}},
        "score":    {"type": "number"},
    },
    "required": ["title", "tags", "score"],
}
resp = client.messages.create(
    model="claude-sonnet-4-6",
    response_format={"type": "json_schema", "schema": schema},
    messages=[...],
)
data = json.loads(resp.content[0].text)
```

OpenAI's "structured outputs" and Anthropic's `tool_use` both guarantee the output validates against a JSON schema (via constrained decoding). For self-hosted, **`outlines`** and **SGLang's `regex`/`json_schema` modes** do the same.

### Pattern 9 — Code completion (ghost text)

```python
# Triggered on every keystroke or pause
def complete(prefix: str, suffix: str) -> str:
    fim_prompt = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
    return llm(fim_prompt, max_tokens=64, stop=["\n\n", "<|fim"])
```

Models: **Qwen-Coder**, **Codestral**, **DeepSeek-Coder-V2**, **StarCoder2**. Fill-in-the-middle (FIM) tokens let the model see what's *after* the cursor, which dramatically improves completion quality. Harness has to debounce keystrokes (300-500ms) and cancel in-flight requests.

### Pattern 10 — Coding agent

A coding agent reads the codebase, edits files, runs tests, fixes errors, and reports back. Pattern: tool-using agent + filesystem + shell + diff/patch tools + a loop until tests pass or task verified.

```python
tools = [read_file, write_file, list_dir, run_shell, run_tests, search_code]
agent_loop(task, tools, max_iter=40, sandbox=docker_container)
```

The big design decision: **how much autonomy**. Read-only mode (suggests diffs to apply manually) vs supervised mode (asks before destructive ops) vs full autonomy (just goes). Most successful 2026 coding products are *supervised by default, fully autonomous on opt-in*.

### Pattern 11 — Voice assistant

```
mic → ASR (Whisper / Deepgram) → LLM → TTS (ElevenLabs / Cartesia) → speaker
```

Each stage adds latency. Target end-to-end latency for "feels conversational" is **< 800 ms** from end-of-user-speech to start-of-assistant-speech. Streaming both the ASR and TTS, plus speculative decoding on the LLM, is required to hit this. **OpenAI Realtime API**, **Anthropic's voice mode**, and **Google's Live API** integrate the stages so you don't have to.

### Pattern 12 — Multimodal Q&A

```python
client.messages.create(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                       "data": b64}},
        {"type": "text", "text": "What's in this image?"},
    ]}],
)
```

Capability: image, video, audio, PDF input across most flagship 2026 models. Limit: per-request image count and resolution; tokens-per-image varies (Anthropic ~1.5k tokens for 1024×1024, Gemini ~258 per "tile"). **Always log the image-token cost; it surprises people.**

---

## 6. Real-world harnesses, dissected

What the popular AI apps actually look like inside.

### Cursor / Windsurf (IDE assistant)

- Models: routing across multiple providers (Claude, GPT, custom Cursor models).
- Tool surface: filesystem (within workspace), git, terminal, web search, MCP.
- Patterns used: code completion (Tab), chat, agent (Composer/Cascade).
- Memory: project-scoped notes, indexed codebase via vector DB.
- Killer move: **codebase indexing** + **inline diff UI**. The model writes diffs; the user accepts/rejects per-hunk.
- Pricing model: subscription with "fast" and "slow" requests; "max mode" for power users.

### Claude Code (CLI / IDE coding agent)

- Model: Claude Sonnet/Opus.
- Tool surface: filesystem, shell, git, web fetch, MCP servers, `Task` for sub-agents, `TodoWrite` for tracking.
- Patterns: tool-using agent + ReAct (with thinking) + sub-agent delegation.
- Sandboxing: optional permission mode (default ask, optional bypass for trusted).
- Memory: per-session conversation; CLAUDE.md project files for persistent context; `/memory` slash command.
- Killer move: deep MCP integration, reads `CLAUDE.md`, fits the developer flow.

### Cline / Roo Cline (open source agent in VS Code)

- Models: any (BYO API key).
- Tool surface: filesystem, terminal, browser (via MCP), MCP servers.
- Patterns: ReAct agent loop, plan-and-execute toggle.
- Permission: every action requires user approval by default.
- Killer move: full transparency — you see every tool call before it executes.

### Aider (CLI coding agent)

- Models: any.
- Tool surface: deliberately minimal — git, file edit, shell.
- Patterns: chat → diff → apply → commit.
- Killer move: every change is a git commit, so undo is trivial. Works well in scripts and CI.

### Devin / Manus / Replit Agent (autonomous full-task)

- Multi-agent: planner + executor + reviewer.
- Sandboxed VM, browser, terminal.
- Patterns: plan-then-execute, multi-agent orchestration, persistent memory.
- Long-running: hours per task, with check-ins.
- Trade-off: more autonomy = more wasted time on wrong direction. Best for clearly-scoped tasks.

### LangGraph / OpenAI Agents SDK (framework approach)

- You define a **graph** of nodes (each is an LLM call or a tool); edges are conditional transitions.
- Useful for production multi-agent systems with explicit control flow.
- LangGraph is the dominant 2026 production framework; OpenAI Agents SDK is gaining ground for OpenAI-centric stacks.

### LlamaIndex (RAG-first)

- Indexes documents into vector store + structured indexes (knowledge graphs, summary trees).
- Query engines: simple, sub-question, router, multi-step.
- Best for: RAG-heavy products with complex document collections.

### Custom thin harness (200-line Python)

Don't reach for a framework if your problem is simple. **Two-thirds of "agent frameworks" overhead is unjustified for most apps.** A direct Python harness with three or four tools and a 30-line loop is often the right call. We build one in §9.

---

## 7. API providers — the 2026 landscape

The big six API providers, plus the major aggregators. **Prices are list rates; volume discounts and yearly commitments are usually 20-40% lower.** Numbers are illustrative as of mid-2026 and shift every few months.

### Frontier-tier APIs

| Provider | Flagship | Mid-tier | Small-tier | Notes |
|----------|----------|----------|------------|-------|
| **Anthropic** | Claude Opus (~$15 in / $75 out / $0.30 cached) | Claude Sonnet (~$3 / $15 / $0.30) | Claude Haiku (~$0.25 / $1.25 / $0.03) | Strongest agentic / coding; best caching discount; MCP-native |
| **OpenAI** | GPT-flagship (~$5 / $15) | GPT-mid (~$0.50 / $2) | GPT-mini (~$0.10 / $0.40) | Best ecosystem, Realtime API, Agents SDK, structured outputs |
| **Google** | Gemini Pro (~$1.25 / $5) | Gemini Flash (~$0.075 / $0.30) | Gemini Flash-Lite | Cheapest mid-tier, 1M+ context, strong multimodal |
| **xAI** | Grok flagship (~$3 / $15) | Grok mid (~$0.30 / $1.50) | — | Strong on real-time/social context |
| **DeepSeek** | DeepSeek-V3.x (~$0.27 / $1.10, $0.07 cached) | — | — | Cheapest open-frontier model; 24h cache |
| **Mistral** | Large (~$2 / $6) | Medium / Small (~$0.40 / $2) | — | EU-hosted, popular for European compliance |

### Speed-tier APIs (LPU / custom silicon)

| Provider | Hardware | Notable | Price posture |
|----------|----------|---------|---------------|
| **Groq** | LPU | 500-1500 tok/s on Llama / Qwen / DeepSeek | mid-priced but blazing |
| **Cerebras Inference** | wafer-scale | similar speed to Groq | mid-priced |
| **SambaNova** | RDU | high-speed Llama hosting | mid-priced |

When the user **feels** speed (voice, agents, IDE), Groq or Cerebras can be the cheapest *effective* option even at 2× per-token.

### Open-model hosting (you bring the model, they run it)

| Provider | Models | Pricing | Notes |
|----------|--------|---------|-------|
| **Together AI** | Llama, Qwen, Mixtral, DeepSeek, custom | $0.20-2/M | broad coverage, BYO fine-tunes |
| **Fireworks** | Llama, Qwen, custom | $0.20-3/M | LoRA serving, function calling |
| **Replicate** | open models | per-second GPU billing | best for cold-batch jobs |
| **OctoAI / RunPod / Vast** | self-managed | hourly GPU | for serious workloads |

### Aggregators

- **OpenRouter** — single API, 100+ models, automatic fallback, transparent pricing. The default for "I'll figure out the right model later." Adds ~5% margin over upstream.
- **LiteLLM** — open-source proxy that maps any provider to OpenAI-compatible API. Self-host as a unified endpoint.
- **Portkey** — observability + routing + retries, sits in front of multiple providers.

### A useful default routing strategy in 2026

```
Best quality, no cost concern  →  Claude Opus / GPT flagship
Default workhorse              →  Claude Sonnet / GPT mid
High volume, cheap             →  Gemini Flash / DeepSeek-V3
Speed-critical (voice, IDE)    →  Groq Llama / Cerebras
Frontier coding (agentic)      →  Claude Sonnet / GPT flagship / Qwen 3.6-27B (self-host)
Long context (>200k)           →  Gemini Pro / Claude Sonnet
Confidential / on-prem         →  Self-hosted Qwen 3.6 / Llama 4 / Gemma 4
```

Set up your harness so changing this routing is one config edit. Provider performance and pricing shifts every few months in 2026.

---

## 8. Self-hosted vs API — when each wins

The question isn't "which is cheaper per token" — it's "which is cheaper *for your usage pattern*."

### Self-hosted economics

A single 8×H100 node ≈ **$25/hour** on-demand, ≈ **$15/hour** reserved.

Running Qwen 3.6-27B fp8 with vLLM at high utilization:
- ~5,000 decode tokens/sec aggregate.
- ~18M output tokens/hour.
- **~$0.85 per million output tokens at 100% utilization**, ~$1.70 at 50%, ~$5 at 15%.

Reality: you'll average 20-40% utilization unless you have very steady traffic. Plan accordingly.

### API economics (mid-tier reference: $3 in / $15 out)

You pay per token used. No idle cost. No DevOps. No GPU shortage to navigate.

### The break-even chart (rough)

```
Monthly output tokens         Self-host pays off at...
──────────────────────────────────────────────────
< 100M / month               API. Always.
100M - 500M / month          API or aggregator. Self-host not worth ops.
500M - 2B / month            ~Tied. Depends on team bandwidth.
2B - 20B / month             Self-host wins, often 3-5×.
> 20B / month                Self-host or Together/Fireworks wins big.
```

Plus secondary considerations:

- **Compliance / data residency**: self-host or EU-hosted API (Mistral, Anthropic EU).
- **Latency-critical** (sub-200ms TTFT): self-host with fast NIC or Groq.
- **Bursty traffic**: API. Self-host can't autoscale fast enough.
- **Steady traffic 24/7**: self-host wins because utilization stays high.
- **Custom fine-tunes**: self-host (or Fireworks/Together which support LoRA).
- **Frontier quality matters more than cost**: API.

A common pattern: **API for 90% of traffic; self-host the high-volume, low-quality bar tier.** Best of both.

---

## 9. Build your own coding agent harness (~300 lines)

A real, runnable harness. Reads files, writes diffs, runs shell, asks user before destructive operations, logs everything. Good starting point — fork and adapt.

```python
"""
Tiny coding agent harness. ~300 lines.
Requirements: anthropic, mcp (optional), rich
"""
from __future__ import annotations
import json, os, subprocess, sys, pathlib, textwrap, time, hashlib
from dataclasses import dataclass, field
from typing import Callable, Any
import anthropic

# ─── 1. Tool registry ───────────────────────────────────────────────────────────

@dataclass
class Tool:
    name: str
    fn: Callable
    description: str
    input_schema: dict
    side_effect: str = "readonly"   # "readonly" | "filesystem" | "destructive"

class Registry:
    def __init__(self):
        self.tools: dict[str, Tool] = {}
    def add(self, tool: Tool):
        self.tools[tool.name] = tool
    def schemas(self):
        return [{"name": t.name, "description": t.description,
                 "input_schema": t.input_schema} for t in self.tools.values()]
    def call(self, name: str, args: dict) -> str:
        try:
            return str(self.tools[name].fn(**args))
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"

# ─── 2. Sandbox / permissions ───────────────────────────────────────────────────

WORKSPACE = pathlib.Path.cwd().resolve()

def safe_path(p: str) -> pathlib.Path:
    full = (WORKSPACE / p).resolve()
    if WORKSPACE not in full.parents and full != WORKSPACE:
        raise ValueError(f"path outside workspace: {p}")
    return full

def confirm(action: str) -> bool:
    print(f"\n[confirm] {action}\n  press Enter to allow, anything else to deny: ",
          end="", flush=True)
    return input().strip() == ""

# ─── 3. Tools ───────────────────────────────────────────────────────────────────

def tool_read_file(path: str) -> str:
    """Read a file from the project workspace."""
    return safe_path(path).read_text()

def tool_write_file(path: str, content: str) -> str:
    """Write text to a file (overwrites). Asks user for permission."""
    p = safe_path(path)
    if not confirm(f"WRITE {p}  ({len(content)} bytes)"):
        return "DENIED by user"
    p.write_text(content)
    return f"wrote {len(content)} bytes to {path}"

def tool_list_dir(path: str = ".") -> str:
    """List files in a directory."""
    p = safe_path(path)
    return "\n".join(sorted(c.name for c in p.iterdir()))

def tool_run_shell(cmd: str) -> str:
    """Run a shell command in the workspace. Asks user."""
    if not confirm(f"SHELL: {cmd}"):
        return "DENIED by user"
    out = subprocess.run(cmd, shell=True, cwd=WORKSPACE,
                         capture_output=True, text=True, timeout=120)
    return f"exit={out.returncode}\nstdout:\n{out.stdout[-4000:]}\nstderr:\n{out.stderr[-2000:]}"

def tool_grep(pattern: str, path: str = ".") -> str:
    """Search for a regex in files under a directory."""
    out = subprocess.run(["grep", "-rEn", pattern, str(safe_path(path))],
                         capture_output=True, text=True, timeout=60)
    return out.stdout[-4000:] or "(no matches)"

def tool_run_tests() -> str:
    """Run the project's test suite (assumes pytest)."""
    out = subprocess.run(["pytest", "-x", "-q"],
                         cwd=WORKSPACE, capture_output=True, text=True, timeout=600)
    return f"exit={out.returncode}\n{out.stdout[-5000:]}"

# ─── 4. Build registry ──────────────────────────────────────────────────────────

reg = Registry()
reg.add(Tool("read_file", tool_read_file,
             "Read a file from workspace.",
             {"type":"object","properties":{"path":{"type":"string"}},
              "required":["path"]}, "readonly"))
reg.add(Tool("write_file", tool_write_file,
             "Write text to a file (asks user). Use only after reading.",
             {"type":"object",
              "properties":{"path":{"type":"string"},"content":{"type":"string"}},
              "required":["path","content"]}, "filesystem"))
reg.add(Tool("list_dir", tool_list_dir,
             "List files in a directory.",
             {"type":"object","properties":{"path":{"type":"string"}}}, "readonly"))
reg.add(Tool("run_shell", tool_run_shell,
             "Run a shell command. Asks user.",
             {"type":"object","properties":{"cmd":{"type":"string"}},
              "required":["cmd"]}, "destructive"))
reg.add(Tool("grep", tool_grep,
             "Search regex across files.",
             {"type":"object",
              "properties":{"pattern":{"type":"string"},"path":{"type":"string"}},
              "required":["pattern"]}, "readonly"))
reg.add(Tool("run_tests", tool_run_tests,
             "Run pytest in the workspace.",
             {"type":"object","properties":{}}, "readonly"))

# ─── 5. Observability ───────────────────────────────────────────────────────────

@dataclass
class Stats:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    tool_calls: int = 0
    cost_usd: float = 0.0
    def add(self, usage, model_rates):
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        cached = getattr(usage, "cache_read_input_tokens", 0) or 0
        self.cached_input_tokens += cached
        self.cost_usd += (
            (usage.input_tokens - cached) * model_rates["input"] / 1e6
            + cached * model_rates["cached"] / 1e6
            + usage.output_tokens * model_rates["output"] / 1e6
        )

# Sonnet 4.6 rates (illustrative)
RATES = {"input": 3.0, "cached": 0.30, "output": 15.0}

# ─── 6. Agent loop ──────────────────────────────────────────────────────────────

SYSTEM = """\
You are a careful coding agent. You work in a sandboxed project directory.
Always read files before writing. Prefer minimal diffs. Run tests after
edits. If something fails, debug with grep and shell before changing more.
Stop when the task is done; do not loop on noise.
"""

def run(task: str, max_iter: int = 30, max_total_tokens: int = 200_000):
    client = anthropic.Anthropic()
    messages = [{"role":"user","content": task}]
    stats = Stats()
    for it in range(max_iter):
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=[{"type":"text","text": SYSTEM,
                     "cache_control":{"type":"ephemeral"}}],
            tools=reg.schemas(),
            messages=messages,
        )
        stats.add(resp.usage, RATES)
        if stats.input_tokens + stats.output_tokens > max_total_tokens:
            print("\n[budget exhausted]"); break

        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        text_blocks = [b.text for b in resp.content if b.type == "text"]
        for t in text_blocks:
            print(t)

        if not tool_uses:
            break  # final answer reached

        # Execute tool calls
        results = []
        for tu in tool_uses:
            stats.tool_calls += 1
            print(f"\n  → tool: {tu.name}({json.dumps(tu.input)[:120]})")
            result = reg.call(tu.name, tu.input)
            print(f"    result: {result[:200]}{'…' if len(result)>200 else ''}")
            results.append({"type":"tool_result","tool_use_id":tu.id,"content":result})

        messages.append({"role":"assistant","content": resp.content})
        messages.append({"role":"user","content": results})

    print(f"\n[done] iters={it+1}  tools={stats.tool_calls}  "
          f"in={stats.input_tokens} cached={stats.cached_input_tokens} "
          f"out={stats.output_tokens}  cost=${stats.cost_usd:.4f}")

if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) or input("task: ")
    run(task)
```

What's not there (deliberately): vector memory, long-running session, MCP integration, retry on rate limits, streaming, web tools, file diffing UX. Each of those is 30-100 lines you add when you need it. **Start small, add only what hurts.**

---

## 10. Cost per pattern, worked out

Take a single user using your product 8 hours/day for a month (22 working days). Assume Anthropic Sonnet rates (`$3 / $0.30 cached / $15`) and 80% cache-hit on input.

| Pattern | Tokens / day | Cost / day | Cost / month |
|---------|-------------|------------|--------------|
| 1 — Single-shot per query (occasional, 50/day, 1k in / 200 out) | 50k in / 10k out | $0.18 | **$4** |
| 2 — Streaming chat (power user, 4k in / 1k out × 30 reqs) | 480k in / 120k out | $1.92 | **$42** |
| 3 — Tool-using agent (10 tasks, 8 turns each, 5k in / 800 out per turn) | 1.6M in / 256k out | $4.80 | **$106** |
| 5 — Plan-then-execute (5 tasks, 12 steps, 6k in / 1.5k out per step) | 1.8M in / 450k out | $7.83 | **$172** |
| 7 — RAG (50 queries, 8k retrieved, 400 out) | 400k in / 20k out | $0.54 | **$12** |
| 9 — Code completion (Tab, 60/hr × 8 hrs, 3k in / 80 out) | 1.4M in / 38k out | $1.42 | **$31** |
| 10 — Coding agent (5 tasks/day, 50k in / 3k out per task) | 1.25M in / 75k out | $1.88 | **$41** |
| 11 — Voice (30 turns/hr × 1 hr, 800 in / 200 out) | 24k in / 6k out | $0.10 | **$2** |
| 12 — Multimodal (20 image queries, 1.5k img-tok + 1k text in / 600 out) | 50k in / 12k out | $0.21 | **$4.6** |

The 8× spread between "single-shot Q&A" ($4/mo) and "plan-then-execute agent" ($172/mo) on the *same user* is exactly why pricing per pattern matters. **Match plan tier to pattern.**

---

## 11. Decision framework: which pattern, which model, which deployment

Before you write any code, write down:

```
1. What is the primary user action? (chat / tab-complete / autonomous task / extract / ...)
2. How many tokens per primary action, in vs out?
3. How many primary actions per active user per day?
4. What's the latency target? (TTFT / TPOT / total)
5. What is the quality bar? (frontier / mid / acceptable)
6. What is the budget per active user per month?
7. What private data flows through? (compliance constraint?)
8. Volume (10/day / 1k/day / 1M/day)?
```

Then read off:

- Q1+Q2 → choose pattern from §2.
- Q3 × Q2 → tokens/user/day → cost calc.
- Q4 → model speed tier (Groq for sub-200ms; standard cloud otherwise).
- Q5 → model tier (flagship / mid / small).
- Q6 vs cost → if over budget, drop to smaller tier or self-host.
- Q7 → on-prem / EU-only / Anthropic enterprise / OpenAI ZDR.
- Q8 → API for low-volume, self-host above ~2B output tokens/month.

This grid solves 90% of "which AI do I use?" debates.

---

## 12. The 2026 cheat sheet

- **Harness > Model.** A great harness with a mid-tier model beats a frontier model with a bad harness.
- **MCP is the standard tool-wiring protocol.** Use it, even for in-house tools.
- **Pick a primary pattern from the 12.** Don't build "general AI."
- **API for < 500M tokens/month, hybrid for 500M-2B, self-host above 2B.**
- **Caching is mandatory.** 70-90% of API bill savings live here.
- **Sandbox every tool call** that touches the filesystem, shell, or network.
- **Build observability before launch.** You will tune from this data.
- **Match plan tier to user pattern.** Casual chat ≠ coding agent. 360× spread.
- **Default route**: Claude Sonnet for agents, Gemini Flash for high-volume Q&A, DeepSeek-V3 for cheapest, Groq for speed-critical, self-host Qwen 3.6 / Llama 4 for steady volume.
- **Reasoning models are slow.** Use them only when reasoning matters; route easy queries to non-thinking variants.
- **Two-thirds of "agent frameworks" are unjustified for most apps.** A 200-line harness beats LangGraph for 80% of products.

---

## Going deeper

- **The MCP spec** at `modelcontextprotocol.io` — read the spec; build a tool server.
- **Anthropic, OpenAI, Google API cookbooks** — every major pattern has an officially-blessed example.
- **`langgraph`, `autogen`, `crewai` repos** — read their orchestration primitives; copy what fits, leave the rest.
- **Aider, Cline, Continue, Claude Code source / docs** — open or partly-open coding agents; reading their prompts and tool sets is the fastest way to learn harness design.
- **`outlines`, **`SGLang`, **`xgrammar`** — the open-source structured-output toolchain.
- **OpenRouter playground** — quickly compare models on the same prompt across providers; great for routing decisions.
- **Sweep the pricing pages quarterly** — providers cut prices regularly; old assumptions go stale fast.

If you've made it through chapters 00 → 18: you understand a model end-to-end (chapters 00-13), how to train it (14-15), what a 2026 frontier model looks like (16), how to serve it at scale (17), and now how to wrap it in a real product. **The rest is taste and shipping. Go build.**
