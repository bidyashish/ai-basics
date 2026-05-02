# 22 · Multimodal Models — Vision, Audio, और Beyond

> **TL;DR** एक multimodal LLM (MLLM / VLM) एक text LLM है जिसमें images, audio, या video के लिए **encoder modules** हैं जो **soft tokens** (continuous embeddings) produce करते हैं और same residual stream में inject होते हैं। Architecturally: **vision encoder (ViT या SigLIP) → projector → LLM**। 2026 frontier (Gemma 4, Qwen 3.6-VL, Llama 4, GPT-5, Claude Opus 4.7, Gemini 2.5) सब native multi-resolution और (कुछ के लिए) audio के साथ ये pattern share करते हैं। ये chapter दिखाता है architecture कैसे काम करता है, practice में VLMs कैसे use करें, QLoRA के साथ एक VLM कैसे fine-tune करें, और eval benchmarks actually क्या मतलब करते हैं। Code English में रहता है; math small है।

---

## 1. Shared Mental Model

A VLM एक text LLM से **separate model नहीं है**। ये एक text LLM है extra ports के साथ।

```
   text tokens ───────────────┐
                              ▼
                        LLM (decoder)  ──→ output text tokens
                              ▲
   image patches → ViT → projector ─────┤
   audio frames  → conformer → projector┤
   video frames  → ViT(per-frame) ──────┘
```

हर non-text modality का **खुद का encoder**, एक small **projector** (1-2 linear layers) जो encoder output dim को LLM के hidden size पर map करे, और result LLM के input में drop हो जाता है जैसे वो normal tokens हों। फिर LLM उन सबके across attend करता है।

बस यही। कोई magic नहीं। Art इन में है:
- **कौन सा encoder** आप pick करते हो (ViT, SigLIP, native multi-resolution)।
- **Tokens कैसे form होते हैं** (per patch एक? pooled? per image variable?)।
- **Training के दौरान उन्हें text के साथ कैसे align करना** (alignment + instruction tuning)।
- **Inference पर एक image कितने tokens worth है** (cost lever)।

---

## 2. Vision Encoder

### Vision Transformer (ViT)

एक image को non-overlapping **patches** (e.g., 14×14 pixels) में split करो, हर एक को flatten करो, एक vector पर project करो, positional embeddings add करो, उन पर एक transformer run करो। Output per patch एक vector है (plus कुछ variants में एक learned [CLS] token)।

A 224×224 image के लिए 14×14 patches के साथ: `(224/14)² = 256 patches → 256 vision tokens`। Larger images = more tokens, more cost।

### CLIP / SigLIP — Text-image Alignment के लिए Pretrained

**CLIP** (OpenAI, 2021) और इसका better successor **SigLIP** (Google, 2023) ViTs हैं contrastive objective के साथ trained: matching pairs के text और image embeddings को साथ pull करो, non-matching को push करो। Result एक image encoder है जिसका output already text के साथ "compatible" space में रहता है।

VLMs almost always एक CLIP/SigLIP image encoder से start करते हैं। SigLIP-2 (mid-2024) और **SigLIP-2 SoViT** 2026 default backbones हैं — वो हर scale पर OpenAI CLIP को outperform करते हैं।

### Native Multi-resolution / Dynamic Patches

Older VLMs सब कुछ 224×224 या 448×448 पर resize करते थे। Modern ones (Qwen-VL, InternVL3, Gemma 4, Pixtral, Llama 4) **native aspect ratios** handle करते हैं:

- Large images को **multiple tiles** में split करके, हर एक को ViT से process करो।
- Vision encoder में **2D RoPE** use करके ताकि tiles के across position information preserved हो।
- **Variable token budgets**: user pick करता है कि एक image कितने vision tokens worth है (Gemma 4: 70 / 140 / 280 / 560 / 1120; more = better quality, more cost)।

ये practice में बहुत matter करता है। 70 tokens पर एक 4K screenshot unreadable है; 1120 tokens पर, model individual error messages read कर सकता है। आप जो choose करते हो उसके लिए pay करते हो।

---

## 3. Projector

A tiny linear layer (या 2-layer MLP) जो vision encoder dim → LLM hidden dim map करे:

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
- **Q-Former** (BLIP-2, Flamingo): learned query tokens जो vision features पर cross-attend करते हैं, फिर project। Token count reduce करता है।
- **Resampler / Perceiver**: similar — fixed number of output tokens।
- **Pixel-shuffle** (InternVL): projection से पहले 2×2 patches को एक में merge करो। Simpler, fast।

Modern open VLMs (Qwen-VL, Llama-3.2-Vision, Gemma 4) mostly **simple MLP projectors** plus pixel-shuffle use करते हैं। Complexity encoder में move हो गई है, projector में नहीं।

---

## 4. 2026 VLM Landscape

Headline models, उनके architectural choices, और rough capability:

| Model | Vision encoder | Projector | LLM | Tokens/image | Strength |
|-------|----------------|-----------|-----|--------------|----------|
| **Gemma 4 31B** | SigLIP-2 SoViT, native AR, multi-tile | MLP | Gemma 4 LLM | 70-1120 (configurable) | best open math/reasoning multimodal |
| **Gemma 4 E4B** | SigLIP-2 SoViT | MLP | Gemma 4 (4.5B effective) | 70-1120 | mobile / on-device multimodal |
| **Qwen 3.6-27B** | Qwen-VL-style 2D RoPE, dynamic tiles | MLP | Qwen 3.6 hybrid | up to ~16k | OCR + agentic के लिए best open VLM |
| **Llama 4** | NVIDIA NV-CLIP variant | MLP | Llama 4 dense | dynamic | open frontier multimodal |
| **GPT-5** | proprietary, multi-tile | proprietary | GPT-5 | dynamic | best general VLM |
| **Claude Opus 4.7** | proprietary | proprietary | Claude | ~1.5k @ 1024² | strongest agentic vision |
| **Gemini 2.5 Pro** | proprietary, very efficient | proprietary | Gemini | ~258 / tile | best long-video, 1M+ context |
| **Pixtral 12B** | Mistral's, native AR | MLP | Mistral Nemo | dynamic | strong open mid-size |
| **InternVL3** | InternViT-6B | pixel-shuffle MLP | various LLMs | dynamic | top open document understanding |
| **Molmo** | OpenAI CLIP, Olmo | MLP | OLMo | dynamic | open, point-and-click predictions |

Audio capability (separate या integrated):

| Model | Audio encoder | Audio in | Audio out |
|-------|---------------|----------|-----------|
| **Gemma 4 E2B/E4B** | USM-style conformer | yes | no |
| **GPT-5 / Realtime** | proprietary | yes | yes (speech-to-speech) |
| **Gemini 2.5** | USM | yes | yes |
| **Whisper-v3** | conformer encoder | (standalone ASR) | no |
| **Qwen2.5-Omni** | Qwen-Audio | yes | yes |
| **Moshi** (Kyutai) | RQ-Transformer | yes | yes (low-latency speech-to-speech) |

2026 audio applications के लिए: high-accuracy ASR के लिए **Whisper-v3 / -v3-turbo**, on-device multimodal-with-audio के लिए **Gemma 4 E**, sub-second speech-to-speech के लिए **OpenAI Realtime / Moshi / Qwen-Omni**।

---

## 5. एक VLM Use करना (API)

सारे major vision APIs same shape follow करते हैं:

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

OpenAI, Gemini, और Mistral image attachments के लिए slightly different field names के साथ same `messages` shape use करते हैं। `transformers` के via self-hosted:

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

Production में **vLLM** के via:

```bash
vllm serve google/gemma-4-31b-it \
    --quantization fp8 \
    --max-model-len 32768 \
    --limit-mm-per-prompt image=4
```

`--limit-mm-per-prompt image=4` per request images cap करता है; limit के बिना, एक malicious user 100 4K images भेज सकता है और आपके KV cache को wreck कर सकता है।

---

## 6. Image-token Cost Trap

**हमेशा image-token usage log करो**। ये सबको surprise करता है।

A single 1024×1024 image:
- Anthropic Claude: ~1,500 tokens।
- OpenAI GPT (high detail): ~700-1,200 tokens।
- Gemini (per 768×768 tile, 258 each): ~258-1,032 tokens।
- Gemma 4: 70 / 140 / 280 / 560 / 1120 (आप choose करते हो)।

A typical UI screenshot एक typical text turn का 5-15× cost है। एक multi-image batch easily image alone के 30k tokens hit कर सकता है। इसलिए **हर 2026 multimodal API per image नहीं per token charge करता है**, और इसलिए आपको *measure* करना चाहिए।

Optimizations:

- **भेजने से पहले resize करो**: 768×768 ज़्यादातर tasks के लिए plenty है; OCR के लिए 1024×1024।
- **Crop / region-of-interest**: relevant area भेजो, whole screen नहीं।
- **सबसे lowest token-budget pick करो जो right answer देता है।** हर setting पर eval करो।
- **Image embeddings cache करो।** Anthropic, OpenAI, Gemini अब सब image caching support करते हैं; same-image repeat queries cheap हैं।
- **OCR-heavy tasks के लिए**, explicit doc-OCR training वाले models prefer करो (Qwen-VL, InternVL3, Mistral Pixtral)।

---

## 7. एक VLM Fine-tuning

Same principles जैसे text LLMs (chapter 20), कुछ wrinkles के साथ।

### क्या Train करें, क्या Freeze

A typical recipe:

- **Freeze**: vision encoder (already great है)।
- **Unfreeze**: projector और LLM (या उन्हें LoRA)।

Vision encoder fine-tuning rare है और usually hurts unless आपका domain (medical, satellite, microscopy) natural images से far है। तब भी, आपको lots of labeled data चाहिए।

### Data Format

```json
{"messages": [
  {"role":"user","content":[
    {"type":"image","image":"path/to/img.png"},
    {"type":"text","text":"Caption this image."},
  ]},
  {"role":"assistant","content":"A red bicycle leaning against a stone wall."}
]}
```

5,000-50,000 ऐसे। Quality 10k examples से नीचे quantity से ज़्यादा matter करती है।

### VLM के लिए QLoRA Recipe (Sketch)

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
    # IMPORTANT: सिर्फ़ LLM layers target करो; vision_tower target मत करो
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    lora_dropout=0.05, bias="none",
    modules_to_save=["multi_modal_projector"],   # projector भी train करो
    task_type="CAUSAL_LM",
)

# ... SFTTrainer config (chapter 20 देखो) ...
```

दो gotchas:
- **Vision encoder को accidentally LoRA मत करो।** Most VLMs इसे `vision_tower` या `vision_model` name करते हैं। `target_modules` regex carefully use करो।
- **`modules_to_save=["multi_modal_projector"]`** projector को fully train करता है (no LoRA), क्योंकि ये small है और full updates से benefit करता है।

### कब Fine-tuning Worth है

- **Domain-specific images**: medical scans, schematics, branded UIs।
- **Custom output formats**: आपके JSON shape में bounding boxes, structured extractions।
- **Specific tasks**: एक specific language के लिए OCR, आपके docs से table extraction।

जब **worth नहीं**: general VLM tasks। Frontier models already great हैं; better prompts और caching usually fine-tuning को beat करते हैं।

---

## 8. Object Detection और Grounding (Text-out Approach)

Modern VLMs **text JSON** के रूप में bounding boxes output कर सकते हैं:

```
Q: image में सारे cars find करो।

A: [
  {"label":"car","bbox":[112, 234, 305, 412]},
  {"label":"car","bbox":[480, 220, 700, 410]},
]
```

Coordinates usually **[0, 1000] पर normalized** हैं (Gemma 4 convention, Qwen-VL भी); कुछ models [0, 1] floats use करते हैं। हमेशा model card check करो।

ये 2026 way है YOLO-style model train किए बिना detection करने का: एक VLM से ask करो, JSON parse करो, boxes render करो। Natural images के लिए astonishingly well काम करता है। Specialized domains (medical, industrial) के लिए, आप अभी भी एक dedicated detector चाहोगे।

**Molmo** (AI2) और आगे जाता है **point predictions** के साथ: ask करो "image में kitchen sink कौन से point पर है?" और `(x, y)` coordinates पाओ। UI agents के लिए useful।

---

## 9. Video Understanding

Video = कई frames। तीन patterns:

1. **Frame sampling**: uniformly N frames sample करो (या content से), हर एक को image के रूप में embed करो, सारे LLM को feed करो। Simple, कुछ minutes तक काम करता है।
2. **Temporal pooling**: special encoder जो frames का stack process करता है (Video-Llama, VideoMAE)। Motion के लिए better।
3. **Long-video architectures**: **Gemini 2.5** ~2 hours के video तक process कर सकता है low res पर sample करके + occasional high-res keyframes। Most others (Llama 4, Gemma 4, Qwen 3.6) कुछ minutes handle करते हैं।

Cost: video tokens में brutally expensive है। 1 fps × 1024² पर एक 5-minute video ≈ 500-2000 minutes worth image tokens। **हमेशा frames downsample करो और per-frame token budget reduce करो।**

Voice + video together (multimodal calls) के लिए: 2026 में सिर्फ़ easy paths **Gemini Live API** और **OpenAI Realtime** हैं।

---

## 10. Audio Handling

दो flavors:

- **ASR (speech to text)** — usually एक separate model (Whisper-v3, NVIDIA Canary, Deepgram, AssemblyAI)। Pipeline: audio → text → text LLM।
- **Native audio in** — model audio directly accept करता है: Gemma 4 E-tier, Qwen2.5-Omni, GPT-5 Realtime, Gemini Live, Moshi। Emotional cues, prosody, multi-speaker के लिए better।
- **Native audio out** — model audio generate करता है: GPT-5 Realtime, Moshi, Qwen-Omni, Gemini Live। 800 ms के नीचे speech-to-speech latency।

Most products के लिए 2026 में: **Whisper-v3-turbo** (open) या **Deepgram / AssemblyAI** (API) ASR के लिए + एक text LLM अभी भी cheapest, most flexible path है। Real-time bidirectional speech (voice agents) के लिए native multimodal पर switch करो।

```python
# Whisper-v3 transcription, फिर text LLM
import whisper
model = whisper.load_model("turbo")
text = model.transcribe("call.mp3")["text"]
# अब `text` को अपने favorite LLM को feed करो
```

---

## 11. Multimodal Evaluation

Multimodal-specific capability benchmarks:

| Benchmark | क्या | Notable |
|-----------|------|---------|
| **MMMU / MMMU-Pro** | college-level, mixed images | standard general MM benchmark |
| **MathVista** | figures के साथ math | images पर reasoning |
| **MATH-Vision** | competition math + diagrams | MathVista से harder |
| **MMBench** | many fine-grained capabilities | broad coverage |
| **DocVQA / InfographicsVQA** | document Q&A | OCR-heavy |
| **ChartQA / PlotQA** | charts | plots से data extraction |
| **OCRBench / OCRBench-v2** | varied conditions के तहत OCR | low-level visual reading |
| **RealWorldQA** | natural images, common knowledge | "useful in life" पर gets at |
| **POPE / Hallucination Bench** | object-presence hallucination | "model कहता है दीवार पर एक clock है" — नहीं है — catches |
| **VideoMME / Video-LLaMA Bench** | video understanding | video models के लिए |
| **AudioBench / VoiceBench** | speech / audio | native audio models के लिए |

Capability-aside, hallucination explicitly evaluate करो: VLMs objects invent करना love करते हैं ("एक clock है दीवार पर" — नहीं है)। POPE और similar benchmarks exactly ये measure करते हैं। **हमेशा अपने VLM pipeline में एक hallucination eval include करो।**

---

## 12. 2026 Cheat Sheet

- A VLM = **vision encoder → projector → text LLM**। LLM most work करता है।
- **SigLIP-2** 2026 default vision backbone है।
- **Native multi-resolution + variable token budget** 2026 standard है।
- **हमेशा image-token usage log करो**; एक image एक text turn का 5-15× है।
- **भेजने से पहले resize / crop करो**; 768² ज़्यादातर tasks के लिए enough है।
- OCR / docs के लिए: **Qwen-VL, InternVL3, Pixtral, Gemma 4 31B**।
- Agentic vision (UI, screenshots) के लिए: **Claude Opus 4.7, GPT-5, Qwen 3.6**।
- Video / long context के लिए: **Gemini 2.5 Pro**।
- On-device multimodal के लिए: **Gemma 4 E2B/E4B**।
- ASR के लिए: **Whisper-v3-turbo** (open) या **Deepgram** (API)।
- **Projector + LLM (LoRA)** fine-tune करो, vision encoder freeze करो।
- Capability के top पर **हमेशा hallucination evaluate करो** (POPE)।

---

## और गहराई से

- **SigLIP-2 paper** (Tschannen et al. 2025) — canonical modern vision encoder।
- **Qwen-VL / Qwen2-VL / Qwen2.5-VL technical reports** — most readable open VLM papers।
- **InternVL3 paper** — fine-grained multi-resolution design।
- **Molmo paper** (AI2 2024) — grounding के साथ open multimodal recipe।
- **Llama 3.2-Vision blog और code** — accessible reference open VLM।
- **`transformers` `AutoModelForVision2Seq` docs** — quick experimentation के लिए।
- **vLLM multimodal serving docs** — production deployment।
- **MMMU और POPE leaderboards** — frontier capabilities और hallucination tradeoffs track करो।

Next: **[23-safety-and-alignment.md](./23-safety-and-alignment.md)** — models को helpful और harmless रखना।
