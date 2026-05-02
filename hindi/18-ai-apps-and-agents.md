# 18 · Users AI को कैसे Use करते हैं — Apps, Harnesses, और Agents

> **TL;DR** एक model AI product का roughly **5%** है। बाकी 95% **harness** है: tool registry, memory, prompt templates, retries, permissioning, observability, और इन सबके around wrapped UX। 2026 में high-leverage skill *training नहीं है* — ये harness को अच्छे से build करना है। ये chapter cover करता है twelve canonical usage patterns, tool wiring के लिए **MCP** standard, real-world harnesses (Cursor, Claude Code, Cline, Devin, Aider, LangGraph), और एक **complete coding-agent harness ~300 lines** Python में जिसे आप पढ़ और modify कर सकते हो। Plus एक full provider pricing table, self-hosted vs API decision framework, और per-pattern cost math।

---

## 1. Model के ऊपर के Layers

अगर आप सिर्फ़ "user message type करता है → model reply करता है" देखते हो, तो आप बीच में चार layers miss कर रहे हो।

```
                ┌──────────────────────────┐
                │  UI / IDE / Voice / Bot  │  user जो touch करता है
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

Model के ऊपर के चार layers वो जगह हैं जहां *सारा* product differentiation होता है। Same model use करने वाले दो products harness choices की वजह से completely different feel हो सकते हैं।

---

## 2. 12 Canonical Usage Patterns

हर AI product इनमें से एक है — या इनका composition। कुछ भी design करने से पहले सही वाला pick करो।

| # | Pattern | One-line goal | Best fit |
|---|---------|---------------|---------|
| 1 | **Single-shot completion** | एक prompt लो, एक answer return करो | classification, simple Q&A |
| 2 | **Streaming chat** | Multi-turn conversation | ChatGPT-style assistants |
| 3 | **Tool-using agent** | Model tools choose करता है call करने को | function-calling APIs |
| 4 | **ReAct loop** | Think → act → observe → think | autonomous reasoning |
| 5 | **Plan-then-execute** | पहले plan करो, फिर steps run करो | structure वाले long tasks |
| 6 | **Multi-agent (orchestrator + workers)** | एक model दूसरों को delegate करता है | research, complex pipelines |
| 7 | **RAG** | Docs retrieve करो → model को feed करो | private data पर knowledge |
| 8 | **Structured output** | JSON / schema-conforming reply force करो | data extraction, APIs |
| 9 | **Code completion (ghost text)** | अगली lines tab-complete करो | IDE assistants |
| 10 | **Coding agent** | Autonomous file edits + tests | Claude Code, Cursor agent |
| 11 | **Voice assistant** | Speech in, speech out | mobile assistants |
| 12 | **Multimodal Q&A** | Image / video / audio in | analysis, accessibility |

List पर नहीं (deliberately): "general AI assistant जो सब कुछ करे।" Generic rarely एक product है। एक primary pattern और एक secondary pick करो। **सिर्फ़ वही stack करो जो आपको चाहिए।**

---

## 3. Harness — Actually क्या है

हर serious AI app में एक **harness** है। ये "user ने X कहा" और "model Y return करता है" के बीच का code है जो system को robust बनाता है। एक harness में ये subsystems होते हैं:

### 3.1 Tool Registry

Tools external functions हैं जो model invoke कर सकता है: `read_file`, `run_shell`, `web_search`, `query_db`, `send_email`, `book_flight`। Registry hold करती है:

- Function pointer।
- एक JSON schema arguments describe करती हुई।
- Permissions (कौन इसे call कर सकता है, किस context में)।
- Side-effect class (read-only, mutating, network, irreversible)।
- Cost tracking (LLM-callable tools जो खुद paid APIs call करते हैं)।

```python
# minimal tool registry
@dataclass
class Tool:
    name: str
    fn: Callable
    schema: dict           # arguments के लिए JSON Schema
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

Short-term: conversation buffer (recent N messages, possibly compacted)। Long-term: vector DB या key-value store उन facts के लिए जो user ने आपको week 1 में बताए और जो आज matter करते हैं।

एक real harness जब context window के पास आता है तो old turns को summarize/compact करता है। **उन conversations पर context blow मत करो जिन्हें model को verbatim देखने की need नहीं है।**

### 3.3 Prompt Templates

Inline prompts string-concatenate मत करो। Versioned templates में centralize करो named variables और A/B test करने का तरीका के साथ। `dotprompt`, `BAML`, `Promptfoo`, या अपना khud का Jinja-based system जैसे tools। Prompt drift consistency का silent killer है।

### 3.4 Retry / Fallback / Circuit Breaker

- Transient API failures → exponential backoff retry।
- Hard rate-limit → wait करो या किसी different provider पर fall back करो।
- Bad output (failed JSON parse, refused, hallucinated) → feedback के साथ retry करो या एक stricter model के साथ।
- Cascading failures → 30 sec के लिए bad provider को circuit-break करो।

### 3.5 Permissions और Sandboxing

Tool-using agents code execute करेंगे, URLs hit करेंगे, files modify करेंगे। **हमेशा sandbox करो।** Shell commands को Docker container या dedicated VM में run करो। Filesystem operations project directory तक confined। Network egress allowlists के साथ proxy के through। Destructive operations के लिए default-deny।

### 3.6 Observability

Per-request: कौन सा model, कौन से tools, token counts, cost, cache hit rate, latency, errors। Per-session: total spend, tool-call distribution, error types। Per-version: prompt template versions, model versions, eval scores। **पहले छह महीने आप daily इस data से tune करोगे। Launch से पहले इसे build करो।**

### 3.7 Output Formatter

Most user-facing apps clean output चाहते हैं। Thinking tokens strip करो, tool calls को collapsible blocks के रूप में format करो, code-fenced output render करो, streaming के दौरान partial JSON handle करो। ये जितना सोचते हो उससे ज़्यादा work है।

एक useful slogan: **"model brain है, harness body है।"** एक great brain bad body में अभी भी stairs पर trip करेगा।

---

## 4. Model Context Protocol (MCP)

Late 2024 में Anthropic ने **MCP** introduce किया — Model Context Protocol — एक JSON-RPC standard कि models tools और data sources से कैसे connect हों। 2026 तक ये harness-tool wiring के लिए **lingua franca** है: Anthropic, OpenAI, Google, JetBrains, VS Code, Cursor, Claude Code, Cline, Continue, और सारे major IDE/IDE-like products या तो support करते हैं या integrate कर रहे हैं।

ये क्यों matter करता है: हर app को GitHub integration, Slack, अपना filesystem, Postgres, और सौ अन्य services re-implement करने के बजाय, एक single MCP server उन्हें एक बार provide करता है और कोई भी MCP-compatible host उन्हें use कर सकता है।

### MCP Architecture

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

A typical setup parallel में **multiple MCP servers** चलाता है — एक files के लिए, एक git के लिए, एक browser automation के लिए, एक आपकी private API के लिए। Host सारे tools उनके across enumerate करता है।

### 30 Lines में एक MCP Server Spin Up करना

```python
# pip install mcp
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-tools")

@mcp.tool()
def search_internal(query: str) -> str:
    """एक query के लिए internal knowledge base search करो।"""
    return run_search(query)

@mcp.tool()
def file_read(path: str) -> str:
    """Project directory से एक file पढ़ो।"""
    return open(path).read()

if __name__ == "__main__":
    mcp.run()
```

उस script का path Claude Code, Cursor, या किसी भी MCP host के config में drop करो और tools तुरंत appear हो जाते हैं। **ये AI ecosystem का सबसे cleanest abstraction है जिस पर settle हुआ है।**

### Directly Plug In करने के लिए Catalogues

- **Composio** — एक MCP/REST surface के पीछे hundreds of pre-built tool integrations।
- **Zapier MCP** — Zapier के 6,000+ integrations tools के रूप में exposed।
- **Anthropic Reference MCP** — filesystem, sqlite, github, browser, postgres reference implementations।
- **Smithery** — MCP servers के लिए package registry।

Most teams के लिए: scratch से MCP servers मत लिखो unless आपके पास private API हो। Existing वालों को compose करो।

---

## 5. Code में Patterns

नीचे हर pattern smallest sensible Python implementation के साथ shown है। Real production code error handling, observability, और timeouts ऊपर add करता है।

### Pattern 1 — Single-shot Completion

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

इसके लिए use करो: classification, extraction, transformation, single-turn Q&A। Stateless, cache और parallelize करना easy।

### Pattern 2 — Memory के साथ Streaming Chat

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

जब `history` एक threshold cross करे तो उसे trim या summarize करो (e.g., > 30 turns या > 50k tokens)। Harness इसे own करता है।

### Pattern 3 — Tool-using Agent (Function Calling)

```python
tools = [
    {"name": "get_weather",
     "description": "एक city में current weather get करो।",
     "input_schema": {"type": "object",
                      "properties": {"city": {"type": "string"}},
                      "required": ["city"]}},
]

resp = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "Tokyo में weather?"}],
)
# resp.content एक tool_use block हो सकता है — invoke करो, result append करो, फिर call करो।
```

Function calling tool use का lowest level है। Model एक tool invoke करने के लिए structured request return करता है; आप उसे run करते हो; आप result append करते हो और call करते हो। Loop तब तक जब तक model "done" न कहे।

### Pattern 4 — ReAct (Reason + Act)

Model **Thought**, **Action**, और **Observation** tokens के बीच alternate करता है, फिर एक final **Answer**। Pre-2024 papers ने XML/text tags use किए; post-MCP, ReAct essentially "chain-of-thought के साथ tool-calling" है। Prompts ऐसे दिखते हैं:

```
You are an agent. हर step के लिए:
- अगला क्या करना है loud में think करो।
- अगर tool चाहिए, उसे call करो।
- Tool result के बाद, continue करो।
- जब आपके पास enough information हो, final answer दो।
```

Modern reasoning models (Claude Opus 4.7 thinking के साथ, GPT-5 reasoning के साथ, DeepSeek-R1) effectively ReAct को model में bake कर देते हैं — आप बस उन्हें tools देते हो और वो internally reason करके act करते हैं।

### Pattern 5 — Plan-then-execute

```python
plan = llm("इसके लिए plan बनाओ: " + task, system=PLAN_PROMPT)  # bullet list
for step in parse_plan(plan):
    result = execute_step(step, llm, tools)
    if result.failed: result = replan(task, plan, step, result)
```

तब useful जब steps में **dependencies** हों (एक function लिखो, फिर उसे test करो, फिर results के basis पर refactor करो)। Free-form research के लिए कम useful जहां next step उस पर depend करता है जो आपने पाया।

### Pattern 6 — Multi-agent (Orchestrator + Workers)

```python
def orchestrator(task):
    subtasks = llm_decompose(task)               # N subtasks में split
    results = [worker_agent(s) for s in subtasks]   # parallel
    return llm_synthesize(task, results)
```

Frameworks: **LangGraph**, **CrewAI**, **AutoGen**, **OpenAI Agents SDK**, **Anthropic का parallel-tool-use API**। तब big productivity boost जब subtasks independent हों (e.g., "50 documents summarize करो")। तब coordination overhead और cost add करता है जब वो न हों।

### Pattern 7 — RAG (Retrieval-Augmented Generation)

```python
def rag_answer(question):
    docs = vector_db.search(embed(question), k=8)   # cosine similarity
    context = "\n\n".join(doc.text for doc in docs)
    prompt = f"<docs>\n{context}\n</docs>\n\nQuestion: {question}"
    return llm(prompt)
```

Modern variants: **hybrid search** (semantic + BM25), cross-encoder के साथ **reranking**, **HyDE** (hypothetical doc embedding), **agentic RAG** जहां model sub-queries issue करता है। 2026 तक most production RAG naive top-k के बजाय explicit retrieval step + reranker use करती है।

### Pattern 8 — Structured Output

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

OpenAI के "structured outputs" और Anthropic के `tool_use` दोनों guarantee करते हैं कि output एक JSON schema के against validate करता है (constrained decoding के via)। Self-hosted के लिए, **`outlines`** और **SGLang के `regex`/`json_schema` modes** same करते हैं।

### Pattern 9 — Code Completion (Ghost Text)

```python
# हर keystroke या pause पर triggered
def complete(prefix: str, suffix: str) -> str:
    fim_prompt = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
    return llm(fim_prompt, max_tokens=64, stop=["\n\n", "<|fim"])
```

Models: **Qwen-Coder**, **Codestral**, **DeepSeek-Coder-V2**, **StarCoder2**। Fill-in-the-middle (FIM) tokens model को देखने देते हैं कि cursor के *बाद* क्या है, जो completion quality को dramatically improve करता है। Harness को keystrokes debounce करने (300-500ms) और in-flight requests cancel करने पड़ते हैं।

### Pattern 10 — Coding Agent

एक coding agent codebase पढ़ता है, files edit करता है, tests run करता है, errors fix करता है, और report करता है। Pattern: tool-using agent + filesystem + shell + diff/patch tools + एक loop जब तक tests pass न हों या task verified न हो।

```python
tools = [read_file, write_file, list_dir, run_shell, run_tests, search_code]
agent_loop(task, tools, max_iter=40, sandbox=docker_container)
```

Big design decision: **कितनी autonomy**। Read-only mode (manually apply करने के लिए diffs suggest करता है) vs supervised mode (destructive ops से पहले asks) vs full autonomy (बस goes)। Most successful 2026 coding products *default by supervised, opt-in पर fully autonomous* हैं।

### Pattern 11 — Voice Assistant

```
mic → ASR (Whisper / Deepgram) → LLM → TTS (ElevenLabs / Cartesia) → speaker
```

हर stage latency add करता है। "Conversational feel" के लिए target end-to-end latency end-of-user-speech से start-of-assistant-speech तक **< 800 ms** है। ASR और TTS दोनों को streaming, plus LLM पर speculative decoding, इसे hit करने के लिए required हैं। **OpenAI Realtime API**, **Anthropic का voice mode**, और **Google का Live API** stages को integrate करते हैं ताकि आपको न करना पड़े।

### Pattern 12 — Multimodal Q&A

```python
client.messages.create(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                       "data": b64}},
        {"type": "text", "text": "इस image में क्या है?"},
    ]}],
)
```

Capability: most flagship 2026 models में image, video, audio, PDF input। Limit: per-request image count और resolution; tokens-per-image vary होते हैं (Anthropic ~1.5k tokens 1024×1024 के लिए, Gemini ~258 per "tile")। **हमेशा image-token cost log करो; ये लोगों को surprise करता है।**

---

## 6. Real-world Harnesses, Dissected

Popular AI apps inside actually कैसे दिखते हैं।

### Cursor / Windsurf (IDE Assistant)

- Models: multiple providers के across routing (Claude, GPT, custom Cursor models)।
- Tool surface: filesystem (workspace के अंदर), git, terminal, web search, MCP।
- Patterns used: code completion (Tab), chat, agent (Composer/Cascade)।
- Memory: project-scoped notes, vector DB के via indexed codebase।
- Killer move: **codebase indexing** + **inline diff UI**। Model diffs लिखता है; user per-hunk accept/reject करता है।
- Pricing model: "fast" और "slow" requests वाला subscription; power users के लिए "max mode"।

### Claude Code (CLI / IDE Coding Agent)

- Model: Claude Sonnet/Opus।
- Tool surface: filesystem, shell, git, web fetch, MCP servers, sub-agents के लिए `Task`, tracking के लिए `TodoWrite`।
- Patterns: tool-using agent + ReAct (thinking के साथ) + sub-agent delegation।
- Sandboxing: optional permission mode (default ask, trusted के लिए optional bypass)।
- Memory: per-session conversation; persistent context के लिए CLAUDE.md project files; `/memory` slash command।
- Killer move: deep MCP integration, `CLAUDE.md` reads करता है, developer flow में fits है।

### Cline / Roo Cline (VS Code में Open Source Agent)

- Models: कोई भी (BYO API key)।
- Tool surface: filesystem, terminal, browser (MCP के via), MCP servers।
- Patterns: ReAct agent loop, plan-and-execute toggle।
- Permission: by default हर action user approval require करती है।
- Killer move: full transparency — आप execute होने से पहले हर tool call देखते हो।

### Aider (CLI Coding Agent)

- Models: कोई भी।
- Tool surface: deliberately minimal — git, file edit, shell।
- Patterns: chat → diff → apply → commit।
- Killer move: हर change एक git commit है, तो undo trivial है। Scripts और CI में अच्छा काम करता है।

### Devin / Manus / Replit Agent (Autonomous Full-task)

- Multi-agent: planner + executor + reviewer।
- Sandboxed VM, browser, terminal।
- Patterns: plan-then-execute, multi-agent orchestration, persistent memory।
- Long-running: per task hours, check-ins के साथ।
- Trade-off: more autonomy = wrong direction पर more wasted time। Clearly-scoped tasks के लिए best।

### LangGraph / OpenAI Agents SDK (Framework Approach)

- आप nodes का एक **graph** define करते हो (हर एक एक LLM call या एक tool है); edges conditional transitions हैं।
- Explicit control flow वाले production multi-agent systems के लिए useful।
- LangGraph dominant 2026 production framework है; OpenAI Agents SDK OpenAI-centric stacks के लिए ground gain कर रहा है।

### LlamaIndex (RAG-first)

- Documents को vector store + structured indexes (knowledge graphs, summary trees) में index करता है।
- Query engines: simple, sub-question, router, multi-step।
- इसके लिए best: complex document collections वाले RAG-heavy products।

### Custom Thin Harness (200-line Python)

अगर आपकी problem simple है तो framework के लिए reach मत करो। **"agent frameworks" का दो-तिहाई overhead most apps के लिए unjustified है।** तीन या चार tools और एक 30-line loop वाला direct Python harness often सही call है। हम §9 में एक build करते हैं।

---

## 7. API Providers — 2026 Landscape

Big six API providers, plus major aggregators। **Prices list rates हैं; volume discounts और yearly commitments usually 20-40% lower होते हैं।** Numbers mid-2026 के illustrative हैं और हर few months shift होते हैं।

### Frontier-tier APIs

| Provider | Flagship | Mid-tier | Small-tier | Notes |
|----------|----------|----------|------------|-------|
| **Anthropic** | Claude Opus (~$15 in / $75 out / $0.30 cached) | Claude Sonnet (~$3 / $15 / $0.30) | Claude Haiku (~$0.25 / $1.25 / $0.03) | Strongest agentic / coding; best caching discount; MCP-native |
| **OpenAI** | GPT-flagship (~$5 / $15) | GPT-mid (~$0.50 / $2) | GPT-mini (~$0.10 / $0.40) | Best ecosystem, Realtime API, Agents SDK, structured outputs |
| **Google** | Gemini Pro (~$1.25 / $5) | Gemini Flash (~$0.075 / $0.30) | Gemini Flash-Lite | Cheapest mid-tier, 1M+ context, strong multimodal |
| **xAI** | Grok flagship (~$3 / $15) | Grok mid (~$0.30 / $1.50) | — | Real-time/social context पर strong |
| **DeepSeek** | DeepSeek-V3.x (~$0.27 / $1.10, $0.07 cached) | — | — | Cheapest open-frontier model; 24h cache |
| **Mistral** | Large (~$2 / $6) | Medium / Small (~$0.40 / $2) | — | EU-hosted, European compliance के लिए popular |

### Speed-tier APIs (LPU / Custom Silicon)

| Provider | Hardware | Notable | Price posture |
|----------|----------|---------|---------------|
| **Groq** | LPU | Llama / Qwen / DeepSeek पर 500-1500 tok/s | mid-priced लेकिन blazing |
| **Cerebras Inference** | wafer-scale | Groq के similar speed | mid-priced |
| **SambaNova** | RDU | high-speed Llama hosting | mid-priced |

जब user speed **feel** करता है (voice, agents, IDE), Groq या Cerebras 2× per-token पर भी cheapest *effective* option हो सकता है।

### Open-model Hosting (आप model लाओ, वो run करते हैं)

| Provider | Models | Pricing | Notes |
|----------|--------|---------|-------|
| **Together AI** | Llama, Qwen, Mixtral, DeepSeek, custom | $0.20-2/M | broad coverage, BYO fine-tunes |
| **Fireworks** | Llama, Qwen, custom | $0.20-3/M | LoRA serving, function calling |
| **Replicate** | open models | per-second GPU billing | cold-batch jobs के लिए best |
| **OctoAI / RunPod / Vast** | self-managed | hourly GPU | serious workloads के लिए |

### Aggregators

- **OpenRouter** — single API, 100+ models, automatic fallback, transparent pricing। "मैं बाद में सही model figure out करूंगा" के लिए default। Upstream पर ~5% margin add करता है।
- **LiteLLM** — open-source proxy जो कोई भी provider को OpenAI-compatible API पर map करता है। Unified endpoint के रूप में self-host करो।
- **Portkey** — multiple providers के सामने observability + routing + retries।

### 2026 में एक Useful Default Routing Strategy

```
Best quality, no cost concern  →  Claude Opus / GPT flagship
Default workhorse              →  Claude Sonnet / GPT mid
High volume, cheap             →  Gemini Flash / DeepSeek-V3
Speed-critical (voice, IDE)    →  Groq Llama / Cerebras
Frontier coding (agentic)      →  Claude Sonnet / GPT flagship / Qwen 3.6-27B (self-host)
Long context (>200k)           →  Gemini Pro / Claude Sonnet
Confidential / on-prem         →  Self-hosted Qwen 3.6 / Llama 4 / Gemma 4
```

अपने harness को इस तरह set up करो कि इस routing को change करना एक config edit हो। Provider performance और pricing 2026 में हर few months shift होती है।

---

## 8. Self-hosted vs API — कब कौन जीतता है

Question "per token कौन cheaper है" नहीं है — "*आपके usage pattern के लिए* कौन cheaper है" है।

### Self-hosted Economics

एक single 8×H100 node ≈ on-demand **$25/hour**, ≈ reserved **$15/hour**।

vLLM के साथ Qwen 3.6-27B fp8 high utilization पर:
- ~5,000 decode tokens/sec aggregate।
- ~18M output tokens/hour।
- **100% utilization पर ~$0.85 per million output tokens**, 50% पर ~$1.70, 15% पर ~$5।

Reality: आप 20-40% utilization average करोगे जब तक आपका traffic बहुत steady न हो। उसके according plan करो।

### API Economics (Mid-tier Reference: $3 in / $15 out)

आप used per token pay करते हो। No idle cost। No DevOps। No GPU shortage navigate करना।

### Break-even Chart (Rough)

```
Monthly output tokens         Self-host pays off at...
──────────────────────────────────────────────────
< 100M / month               API. हमेशा।
100M - 500M / month          API या aggregator। Self-host ops worth नहीं।
500M - 2B / month            ~Tied। Team bandwidth पर depend करता है।
2B - 20B / month             Self-host wins, often 3-5×।
> 20B / month                Self-host या Together/Fireworks big wins।
```

Plus secondary considerations:

- **Compliance / data residency**: self-host या EU-hosted API (Mistral, Anthropic EU)।
- **Latency-critical** (sub-200ms TTFT): fast NIC के साथ self-host या Groq।
- **Bursty traffic**: API। Self-host fast enough autoscale नहीं कर सकता।
- **Steady traffic 24/7**: self-host wins क्योंकि utilization high रहती है।
- **Custom fine-tunes**: self-host (या Fireworks/Together जो LoRA support करते हैं)।
- **Frontier quality cost से ज़्यादा matter करती है**: API।

A common pattern: **90% traffic के लिए API; high-volume, low-quality bar tier self-host**। Both का best।

---

## 9. अपना Coding Agent Harness Build करो (~300 lines)

A real, runnable harness। Files पढ़ता है, diffs लिखता है, shell run करता है, destructive operations से पहले user से ask करता है, सब log करता है। Good starting point — fork करो और adapt करो।

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
    print(f"\n[confirm] {action}\n  allow करने के लिए Enter दबाओ, deny करने के लिए कुछ और: ",
          end="", flush=True)
    return input().strip() == ""

# ─── 3. Tools ───────────────────────────────────────────────────────────────────

def tool_read_file(path: str) -> str:
    """Project workspace से एक file पढ़ो।"""
    return safe_path(path).read_text()

def tool_write_file(path: str, content: str) -> str:
    """एक file में text write करो (overwrites)। User से permission लेता है।"""
    p = safe_path(path)
    if not confirm(f"WRITE {p}  ({len(content)} bytes)"):
        return "DENIED by user"
    p.write_text(content)
    return f"wrote {len(content)} bytes to {path}"

def tool_list_dir(path: str = ".") -> str:
    """एक directory में files list करो।"""
    p = safe_path(path)
    return "\n".join(sorted(c.name for c in p.iterdir()))

def tool_run_shell(cmd: str) -> str:
    """Workspace में एक shell command run करो। User से asks करता है।"""
    if not confirm(f"SHELL: {cmd}"):
        return "DENIED by user"
    out = subprocess.run(cmd, shell=True, cwd=WORKSPACE,
                         capture_output=True, text=True, timeout=120)
    return f"exit={out.returncode}\nstdout:\n{out.stdout[-4000:]}\nstderr:\n{out.stderr[-2000:]}"

def tool_grep(pattern: str, path: str = ".") -> str:
    """एक directory के नीचे files में regex search करो।"""
    out = subprocess.run(["grep", "-rEn", pattern, str(safe_path(path))],
                         capture_output=True, text=True, timeout=60)
    return out.stdout[-4000:] or "(no matches)"

def tool_run_tests() -> str:
    """Project का test suite run करो (assumes pytest)।"""
    out = subprocess.run(["pytest", "-x", "-q"],
                         cwd=WORKSPACE, capture_output=True, text=True, timeout=600)
    return f"exit={out.returncode}\n{out.stdout[-5000:]}"

# ─── 4. Build registry ──────────────────────────────────────────────────────────

reg = Registry()
reg.add(Tool("read_file", tool_read_file,
             "Workspace से एक file पढ़ो।",
             {"type":"object","properties":{"path":{"type":"string"}},
              "required":["path"]}, "readonly"))
reg.add(Tool("write_file", tool_write_file,
             "एक file में text write करो (asks user)। पढ़ने के बाद ही use करो।",
             {"type":"object",
              "properties":{"path":{"type":"string"},"content":{"type":"string"}},
              "required":["path","content"]}, "filesystem"))
reg.add(Tool("list_dir", tool_list_dir,
             "एक directory में files list करो।",
             {"type":"object","properties":{"path":{"type":"string"}}}, "readonly"))
reg.add(Tool("run_shell", tool_run_shell,
             "एक shell command run करो। User से asks।",
             {"type":"object","properties":{"cmd":{"type":"string"}},
              "required":["cmd"]}, "destructive"))
reg.add(Tool("grep", tool_grep,
             "Files के across regex search।",
             {"type":"object",
              "properties":{"pattern":{"type":"string"},"path":{"type":"string"}},
              "required":["pattern"]}, "readonly"))
reg.add(Tool("run_tests", tool_run_tests,
             "Workspace में pytest run करो।",
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
You are a careful coding agent. आप एक sandboxed project directory में work करते हो।
हमेशा writing से पहले files पढ़ो। Minimal diffs prefer करो। Edits के बाद tests run करो।
अगर कुछ fail हो, और changes करने से पहले grep और shell से debug करो।
जब task done हो तो stop करो; noise पर loop मत करो।
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

        # Tool calls execute करो
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

जो वहां नहीं है (deliberately): vector memory, long-running session, MCP integration, rate limits पर retry, streaming, web tools, file diffing UX। हर एक 30-100 lines है जो आप तब add करते हो जब आपको need हो। **Small start करो, सिर्फ़ वही add करो जो hurts।**

---

## 10. Per Pattern, Worked Out Cost

एक single user को लो जो आपका product महीने के लिए 8 hours/day (22 working days) use करता है। Anthropic Sonnet rates (`$3 / $0.30 cached / $15`) और input पर 80% cache-hit assume करो।

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

*Same user* पर "single-shot Q&A" ($4/mo) और "plan-then-execute agent" ($172/mo) के बीच 8× spread exactly वो है क्यों pricing per pattern matter करती है। **Plan tier को pattern से match करो।**

---

## 11. Decision Framework: कौन सा Pattern, कौन सा Model, कौन सा Deployment

कोई code लिखने से पहले, ये लिखो:

```
1. Primary user action क्या है? (chat / tab-complete / autonomous task / extract / ...)
2. Per primary action कितने tokens, in vs out?
3. Per active user per day कितने primary actions?
4. Latency target क्या है? (TTFT / TPOT / total)
5. Quality bar क्या है? (frontier / mid / acceptable)
6. Per active user per month budget क्या है?
7. कौन सा private data flow होता है? (compliance constraint?)
8. Volume (10/day / 1k/day / 1M/day)?
```

फिर read off करो:

- Q1+Q2 → §2 से pattern choose करो।
- Q3 × Q2 → tokens/user/day → cost calc।
- Q4 → model speed tier (sub-200ms के लिए Groq; otherwise standard cloud)।
- Q5 → model tier (flagship / mid / small)।
- Q6 vs cost → अगर budget over हो, smaller tier पर drop करो या self-host करो।
- Q7 → on-prem / EU-only / Anthropic enterprise / OpenAI ZDR।
- Q8 → low-volume के लिए API, ~2B output tokens/month से ऊपर self-host।

ये grid 90% "मैं कौन सी AI use करूं?" debates solve करती है।

---

## 12. 2026 Cheat Sheet

- **Harness > Model।** Mid-tier model वाला great harness frontier model वाले bad harness को beat करता है।
- **MCP standard tool-wiring protocol है।** Use करो, in-house tools के लिए भी।
- **12 में से एक primary pattern pick करो।** "General AI" मत build करो।
- **< 500M tokens/month के लिए API, 500M-2B के लिए hybrid, 2B से ऊपर self-host।**
- **Caching mandatory है।** API bill savings का 70-90% यहां रहता है।
- **हर tool call sandbox करो** जो filesystem, shell, या network को touch करे।
- **Launch से पहले observability build करो।** आप इस data से tune करोगे।
- **Plan tier को user pattern से match करो।** Casual chat ≠ coding agent। 360× spread।
- **Default route**: agents के लिए Claude Sonnet, high-volume Q&A के लिए Gemini Flash, cheapest के लिए DeepSeek-V3, speed-critical के लिए Groq, steady volume के लिए self-host Qwen 3.6 / Llama 4।
- **Reasoning models slow हैं।** सिर्फ़ तब use करो जब reasoning matter करे; easy queries non-thinking variants पर route करो।
- **"Agent frameworks" का दो-तिहाई most apps के लिए unjustified है।** एक 200-line harness 80% products के लिए LangGraph को beat करता है।

---

## और गहराई से

- `modelcontextprotocol.io` पर **MCP spec** — spec पढ़ो; एक tool server build करो।
- **Anthropic, OpenAI, Google API cookbooks** — हर major pattern का एक officially-blessed example है।
- **`langgraph`, `autogen`, `crewai` repos** — उनके orchestration primitives पढ़ो; जो fit हो copy करो, बाकी छोड़ो।
- **Aider, Cline, Continue, Claude Code source / docs** — open या partly-open coding agents; उनके prompts और tool sets पढ़ना harness design सीखने का fastest तरीका है।
- **`outlines`, `SGLang`, `xgrammar`** — open-source structured-output toolchain।
- **OpenRouter playground** — providers के across same prompt पर models quickly compare करो; routing decisions के लिए great।
- **हर तीन महीने pricing pages sweep करो** — providers regularly prices cut करते हैं; पुरानी assumptions fast stale हो जाती हैं।

अगर आपने chapters 00 → 18 के through कर लिया: आप एक model end-to-end समझते हो (chapters 00-13), उसे कैसे train करना है (14-15), 2026 frontier model कैसा दिखता है (16), उसे scale पर कैसे serve करना है (17), और अब उसे एक real product में कैसे wrap करना है। **बाकी taste और shipping है। Build करने जाओ।**
