# 22 · Multimodal Models — Vision, Audio, and Beyond

> **TL;DR** A multimodal LLM (MLLM / VLM) is a text LLM with **encoder modules** for images, audio, or video that produce **soft tokens** (continuous embeddings) injected into the same residual stream. Architecturally: **vision encoder (ViT or SigLIP) → projector → LLM**. The 2026 frontier (Gemma 4, Qwen 3.6-VL, Llama 4, GPT-5, Claude Opus 4.7, Gemini 2.5) all share this pattern with native multi-resolution and (for some) audio. This chapter shows how the architecture works, how to use VLMs in practice, how to fine-tune one with QLoRA, and what eval benchmarks actually mean. Code stays English; the math is small.

---

## 1. The shared mental model

A VLM is **not** a separate model from a text LLM. It's a text LLM with extra ports.

```
   text tokens ───────────────┐
                              ▼
                        LLM (decoder)  ──→ output text tokens
                              ▲
   image patches → ViT → projector ─────┤
   audio frames  → conformer → projector┤
   video frames  → ViT(per-frame) ──────┘
```

Each non-text modality has its **own encoder**, a small **projector** (1-2 linear layers) that maps the encoder output dim into the LLM's hidden size, and the result is dropped into the LLM's input as if they were normal tokens. The LLM then attends across all of them.

That's it. No magic. The art is in:
- **Which encoder** you pick (ViT, SigLIP, native multi-resolution).
- **How tokens are formed** (one per patch? pooled? variable per image?).
- **How to align them** with text during training (alignment + instruction tuning).
- **How many tokens an image is worth** at inference (the cost lever).

---

## 2. The vision encoder

### Vision Transformer (ViT)

Split an image into non-overlapping **patches** (e.g., 14×14 pixels), flatten each, project to a vector, add positional embeddings, run a transformer over them. The output is a vector per patch (plus a learned [CLS] token in some variants).

For a 224×224 image with 14×14 patches: `(224/14)² = 256 patches → 256 vision tokens`. Larger images = more tokens, more cost.

### CLIP / SigLIP — pretrained for text-image alignment

**CLIP** (OpenAI, 2021) and its better successor **SigLIP** (Google, 2023) are ViTs trained with a contrastive objective: pull text and image embeddings of matching pairs together, push non-matching apart. The result is an image encoder whose output already lives in a "compatible" space with text.

VLMs almost always start from a CLIP/SigLIP image encoder. SigLIP-2 (mid-2024) and **SigLIP-2 SoViT** are the 2026 default backbones — they outperform OpenAI CLIP at every scale.

### Native multi-resolution / dynamic patches

Older VLMs resized everything to 224×224 or 448×448. Modern ones (Qwen-VL, InternVL3, Gemma 4, Pixtral, Llama 4) handle **native aspect ratios** by:

- Splitting large images into **multiple tiles**, each processed by ViT.
- Using **2D RoPE** in the vision encoder so position information is preserved across tiles.
- **Variable token budgets**: the user picks how many vision tokens an image is worth (Gemma 4: 70 / 140 / 280 / 560 / 1120; the more, the better quality, the more cost).

This matters a lot in practice. A 4K screenshot at 70 tokens is unreadable; at 1120 tokens, the model can read individual error messages. You pay for what you choose.

---

## 3. The projector

A tiny linear layer (or 2-layer MLP) mapping vision encoder dim → LLM hidden dim:

```python
class Projector(nn.Module):
    def __init__(self, vision_dim, llm_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )
    def forward(self, x):  # x: (B, n_patches, vision_dim)
        return self.proj(x)
```

Variants:
- **Q-Former** (BLIP-2, Flamingo): learned query tokens that cross-attend over the vision features, then project. Reduces token count.
- **Resampler / Perceiver**: similar — fixed number of output tokens.
- **Pixel-shuffle** (InternVL): merge 2×2 patches into one before projection. Simpler, fast.

Modern open VLMs (Qwen-VL, Llama-3.2-Vision, Gemma 4) mostly use **simple MLP projectors** plus pixel-shuffle. The complexity has moved into the encoder, not the projector.

---

## 4. The 2026 VLM landscape

Headline models, their architectural choices, and rough capability:

| Model | Vision encoder | Projector | LLM | Tokens/image | Strength |
|-------|----------------|-----------|-----|--------------|----------|
| **Gemma 4 31B** | SigLIP-2 SoViT, native AR, multi-tile | MLP | Gemma 4 LLM | 70-1120 (configurable) | best open math/reasoning multimodal |
| **Gemma 4 E4B** | SigLIP-2 SoViT | MLP | Gemma 4 (4.5B effective) | 70-1120 | mobile / on-device multimodal |
| **Qwen 3.6-27B** | Qwen-VL-style 2D RoPE, dynamic tiles | MLP | Qwen 3.6 hybrid | up to ~16k | best open VLM for OCR + agentic |
| **Llama 4** | NVIDIA NV-CLIP variant | MLP | Llama 4 dense | dynamic | open frontier multimodal |
| **GPT-5** | proprietary, multi-tile | proprietary | GPT-5 | dynamic | best general VLM |
| **Claude Opus 4.7** | proprietary | proprietary | Claude | ~1.5k @ 1024² | strongest agentic vision |
| **Gemini 2.5 Pro** | proprietary, very efficient | proprietary | Gemini | ~258 / tile | best long-video, 1M+ context |
| **Pixtral 12B** | Mistral's, native AR | MLP | Mistral Nemo | dynamic | strong open mid-size |
| **InternVL3** | InternViT-6B | pixel-shuffle MLP | various LLMs | dynamic | top open document understanding |
| **Molmo** | OpenAI CLIP, Olmo | MLP | OLMo | dynamic | open, point-and-click predictions |

Audio capability (separate or integrated):

| Model | Audio encoder | Audio in | Audio out |
|-------|---------------|----------|-----------|
| **Gemma 4 E2B/E4B** | USM-style conformer | yes | no |
| **GPT-5 / Realtime** | proprietary | yes | yes (speech-to-speech) |
| **Gemini 2.5** | USM | yes | yes |
| **Whisper-v3** | conformer encoder | (standalone ASR) | no |
| **Qwen2.5-Omni** | Qwen-Audio | yes | yes |
| **Moshi** (Kyutai) | RQ-Transformer | yes | yes (low-latency speech-to-speech) |

For 2026 audio applications: **Whisper-v3 / -v3-turbo** for high-accuracy ASR, **Gemma 4 E** for on-device multimodal-with-audio, **OpenAI Realtime / Moshi / Qwen-Omni** for sub-second speech-to-speech.

---

## 5. Using a VLM (API)

All major vision APIs follow the same shape:

```python
import anthropic, base64
client = anthropic.Anthropic()
img_b64 = base64.b64encode(open("chart.png","rb").read()).decode()

resp = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role":"user","content":[
        {"type":"image","source":{"type":"base64","media_type":"image/png",
                                   "data":img_b64}},
        {"type":"text","text":"What does this chart show? List the trend lines."},
    ]}],
)
print(resp.content[0].text)
```

OpenAI, Gemini, and Mistral use the same `messages` shape with slightly different field names for image attachments. Self-hosted via `transformers`:

```python
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

m = AutoModelForVision2Seq.from_pretrained("google/gemma-4-31b-it",
        torch_dtype=torch.bfloat16, device_map="auto")
proc = AutoProcessor.from_pretrained("google/gemma-4-31b-it")

inputs = proc.apply_chat_template(
    [{"role":"user","content":[
        {"type":"image","image":"chart.png"},
        {"type":"text","text":"What does this chart show?"},
    ]}],
    tokenize=True, return_tensors="pt", return_dict=True,
).to(m.device)

out = m.generate(**inputs, max_new_tokens=512)
print(proc.decode(out[0], skip_special_tokens=True))
```

In **vLLM** for production:

```bash
vllm serve google/gemma-4-31b-it \
    --quantization fp8 \
    --max-model-len 32768 \
    --limit-mm-per-prompt image=4
```

`--limit-mm-per-prompt image=4` caps images per request; without a limit, a malicious user can send 100 4K images and wreck your KV cache.

---

## 6. The image-token cost trap

**Always log image-token usage**. It surprises everyone.

A single 1024×1024 image:
- Anthropic Claude: ~1,500 tokens.
- OpenAI GPT (high detail): ~700-1,200 tokens.
- Gemini (per 768×768 tile, 258 each): ~258-1,032 tokens.
- Gemma 4: 70 / 140 / 280 / 560 / 1120 (you choose).

A typical UI screenshot is 5-15× the cost of a typical text turn. A multi-image batch can easily hit 30k tokens of image alone. This is why **every 2026 multimodal API charges per token, not per image**, and why you should *measure*.

Optimizations:

- **Resize before sending**: 768×768 is plenty for most tasks; 1024×1024 for OCR.
- **Crop / region-of-interest**: send the relevant area, not the whole screen.
- **Pick the lowest token-budget that gets you the right answer.** Eval at each setting.
- **Cache image embeddings.** Anthropic, OpenAI, Gemini all support image caching now; same-image repeat queries are cheap.
- **For OCR-heavy tasks**, prefer models with explicit doc-OCR training (Qwen-VL, InternVL3, Mistral Pixtral).

---

## 7. Fine-tuning a VLM

Same principles as text LLMs (chapter 20), with a few wrinkles.

### What to train, what to freeze

A typical recipe:

- **Freeze**: vision encoder (it's already great).
- **Unfreeze**: projector and LLM (or LoRA them).

Vision encoder fine-tuning is rare and usually hurts unless your domain (medical, satellite, microscopy) is far from natural images. Even then, you need a lot of labeled data.

### Data format

```json
{"messages": [
  {"role":"user","content":[
    {"type":"image","image":"path/to/img.png"},
    {"type":"text","text":"Caption this image."},
  ]},
  {"role":"assistant","content":"A red bicycle leaning against a stone wall."}
]}
```

5,000-50,000 of these. Quality matters more than quantity below 10k.

### QLoRA recipe for VLM (sketch)

```python
from transformers import AutoModelForVision2Seq, BitsAndBytesConfig, AutoProcessor
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import torch

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
m = AutoModelForVision2Seq.from_pretrained(
    "google/gemma-4-e4b-it",
    quantization_config=bnb, device_map="auto",
    attn_implementation="flash_attention_2",
)
proc = AutoProcessor.from_pretrained("google/gemma-4-e4b-it")

lora = LoraConfig(
    r=16, lora_alpha=32,
    # IMPORTANT: target only LLM layers; do NOT target vision_tower
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    lora_dropout=0.05, bias="none",
    modules_to_save=["multi_modal_projector"],   # train projector too
    task_type="CAUSAL_LM",
)

# ... SFTTrainer config (see chapter 20) ...
```

Two gotchas:
- **Don't accidentally LoRA the vision encoder.** Most VLMs name it `vision_tower` or `vision_model`. Use `target_modules` regex carefully.
- **`modules_to_save=["multi_modal_projector"]`** trains the projector fully (no LoRA), since it's small and benefits from full updates.

### When fine-tuning is worth it

- **Domain-specific images**: medical scans, schematics, branded UIs.
- **Custom output formats**: bounding boxes in your JSON shape, structured extractions.
- **Specific tasks**: OCR for a specific language, table extraction from your docs.

When **not** worth it: general VLM tasks. Frontier models are already great; better prompts and caching usually beat fine-tuning.

---

## 8. Object detection and grounding (text-out approach)

Modern VLMs can output bounding boxes as **text JSON**:

```
Q: Find all cars in the image.

A: [
  {"label":"car","bbox":[112, 234, 305, 412]},
  {"label":"car","bbox":[480, 220, 700, 410]},
]
```

Coordinates are usually **normalized to [0, 1000]** (Gemma 4 convention, also Qwen-VL); some models use [0, 1] floats. Always check the model card.

This is the 2026 way to do detection without training a YOLO-style model: ask a VLM, parse the JSON, render boxes. Works astonishingly well for natural images. For specialized domains (medical, industrial), you'll still want a dedicated detector.

**Molmo** (AI2) goes further with **point predictions**: ask "which point in the image is the kitchen sink?" and get `(x, y)` coordinates. Useful for UI agents.

---

## 9. Video understanding

Video = many frames. Three patterns:

1. **Frame sampling**: sample N frames uniformly (or by content), embed each as an image, feed all to LLM. Simple, works up to a few minutes.
2. **Temporal pooling**: special encoder that processes a stack of frames (Video-Llama, VideoMAE). Better for motion.
3. **Long-video architectures**: **Gemini 2.5** can process up to ~2 hours of video by sampling at low res + occasional high-res keyframes. Most others (Llama 4, Gemma 4, Qwen 3.6) handle a few minutes.

Cost: video is brutally expensive in tokens. A 5-minute video at 1 fps × 1024² ≈ 500-2000 minutes worth of image tokens. **Always downsample frames and reduce per-frame token budget.**

For voice + video together (multimodal calls): **Gemini Live API** and **OpenAI Realtime** are the only easy 2026 paths.

---

## 10. Audio handling

Two flavors:

- **ASR (speech to text)** — usually a separate model (Whisper-v3, NVIDIA Canary, Deepgram, AssemblyAI). Pipeline: audio → text → text LLM.
- **Native audio in** — model accepts audio directly: Gemma 4 E-tier, Qwen2.5-Omni, GPT-5 Realtime, Gemini Live, Moshi. Better for emotional cues, prosody, multi-speaker.
- **Native audio out** — model generates audio: GPT-5 Realtime, Moshi, Qwen-Omni, Gemini Live. Speech-to-speech latency under 800 ms.

For most products in 2026: **Whisper-v3-turbo** (open) or **Deepgram / AssemblyAI** (API) for ASR + a text LLM is still the cheapest, most flexible path. Switch to native multimodal when you need real-time bidirectional speech (voice agents).

```python
# Whisper-v3 transcription, then text LLM
import whisper
model = whisper.load_model("turbo")
text = model.transcribe("call.mp3")["text"]
# now feed `text` to your favorite LLM
```

---

## 11. Multimodal evaluation

Capability benchmarks specific to multimodal:

| Benchmark | What | Notable |
|-----------|------|---------|
| **MMMU / MMMU-Pro** | college-level, mixed images | the standard general MM benchmark |
| **MathVista** | math with figures | reasoning over images |
| **MATH-Vision** | competition math + diagrams | harder than MathVista |
| **MMBench** | many fine-grained capabilities | broad coverage |
| **DocVQA / InfographicsVQA** | document Q&A | OCR-heavy |
| **ChartQA / PlotQA** | charts | data extraction from plots |
| **OCRBench / OCRBench-v2** | OCR under varied conditions | low-level visual reading |
| **RealWorldQA** | natural images, common knowledge | gets at "useful in life" |
| **POPE / Hallucination Bench** | object-presence hallucination | catches "the model says there's a dog when there isn't" |
| **VideoMME / Video-LLaMA Bench** | video understanding | for video models |
| **AudioBench / VoiceBench** | speech / audio | for native audio models |

Capability-aside, evaluate hallucination explicitly: VLMs love to invent objects ("there's a clock on the wall" — there isn't). POPE and similar benchmarks measure exactly this. **Always include a hallucination eval in your VLM pipeline.**

---

## 12. The 2026 cheat sheet

- A VLM = **vision encoder → projector → text LLM**. The LLM does most of the work.
- **SigLIP-2** is the 2026 default vision backbone.
- **Native multi-resolution + variable token budget** is the 2026 standard.
- **Always log image-token usage**; an image is 5-15× a text turn.
- **Resize / crop before sending**; 768² is enough for most tasks.
- For OCR / docs: **Qwen-VL, InternVL3, Pixtral, Gemma 4 31B**.
- For agentic vision (UI, screenshots): **Claude Opus 4.7, GPT-5, Qwen 3.6**.
- For video / long context: **Gemini 2.5 Pro**.
- For on-device multimodal: **Gemma 4 E2B/E4B**.
- For ASR: **Whisper-v3-turbo** (open) or **Deepgram** (API).
- Fine-tune the **projector + LLM (LoRA)**, freeze the vision encoder.
- **Always evaluate hallucination** (POPE) on top of capability.

---

## Going deeper

- **SigLIP-2 paper** (Tschannen et al. 2025) — the canonical modern vision encoder.
- **Qwen-VL / Qwen2-VL / Qwen2.5-VL technical reports** — most readable open VLM papers.
- **InternVL3 paper** — fine-grained multi-resolution design.
- **Molmo paper** (AI2 2024) — open multimodal recipe with grounding.
- **Llama 3.2-Vision blog and code** — accessible reference open VLM.
- **`transformers` `AutoModelForVision2Seq` docs** — for quick experimentation.
- **vLLM multimodal serving docs** — production deployment.
- **MMMU and POPE leaderboards** — track frontier capabilities and hallucination tradeoffs.

Next: **[23-safety-and-alignment.md](./23-safety-and-alignment.md)** — keeping models helpful and harmless.
