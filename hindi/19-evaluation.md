# 19 · Evaluation — कैसे जानें कि आपका Model Actually अच्छा है

> **TL;DR** Evaluation LLM engineering का under-invested आधा है। तीन tiers हैं — **capability** (model चीज़ें जानता है?), **quality** (outputs अच्छे हैं?), और **reliability** (changes के बाद अच्छा रहता है?)। Capability के लिए **static benchmarks** (MMLU, GSM8K, HumanEval, SWE-bench), open-ended quality के लिए **LLM-as-judge**, और reliability के लिए अपने **golden + regression sets**। **Eval-driven development** — prompts से पहले evals लिखना — 2026 में LLM products ship करने के लिए single highest-leverage practice है। इसके बिना हर team slower और worse ship करती है।

---

## 1. Three Tiers of Evaluation

किसी भी tooling से पहले, clear हो जाओ कि measure क्या करना है।

| Tier | जो question answer करता है | Tool |
|------|---------------------|------|
| **Capability** | "क्या model X जानता है?" | static benchmarks (MMLU, GSM8K, HumanEval) |
| **Quality** | "क्या ये user के लिए good answer है?" | judge models, human raters, win-rate |
| **Reliability** | "क्या मेरे last change ने कुछ break किया?" | golden set + regression tests |

एक finished AI product को तीनों चाहिए। Capability बताती है कौन सा model pick करना है। Quality बताती है आपके prompts और harness अच्छे हैं या नहीं। Reliability बताती है कि आप ship कर सकते हो।

---

## 2. Capability Benchmarks — 2026 Short List

ये public benchmarks हैं जो हर model release report करता है। आपको सबकी need नहीं; अपने product के लिए सबसे relevant 4-6 pick करो।

| Benchmark | Tests | Notable |
|-----------|-------|---------|
| **MMLU / MMLU-Pro** | 57 / 14 fields के across broad knowledge & reasoning | canonical "general intelligence" test |
| **GPQA Diamond** | graduate-level science questions | harder, less contaminated |
| **BigBench Hard / BBEH** | mixed reasoning tasks | diversity के लिए great |
| **GSM8K / MATH-500 / AIME 2025-26** | math | GSM8K saturated है; AIME नया bar है |
| **HumanEval / MBPP / LiveCodeBench v6** | code generation | LCB modern, less-contaminated standard है |
| **SWE-bench Verified / SWE-bench Pro** | GitHub से real-world bug fixes | headline agentic-coding bench |
| **Terminal-Bench 2.0** | shell / agent tool use | newer, autonomous CLI use test करता है |
| **MMMU / MMMU Pro** | multimodal reasoning | vision models के लिए standard |
| **MathVista / MATH-Vision** | images के साथ math | long-context multimodal |
| **MRCR-v2** | long-context retrieval (128k में 8 needles) | post-needle-in-haystack standard |
| **OSWorld / GAIA** | desktop / web पर autonomous agent | agent benchmarks |
| **HELM / Open LLM Leaderboard** | aggregator | rankings sanity-check |

**Contamination warning.** Models public web पर train होते हैं, जिसमें कई benchmarks होते हैं। Recent alternates (GPQA Diamond, MMLU-Pro, AIME 2025-26, LiveCodeBench v6, SWE-bench Verified) leak करना harder design किए गए हैं। हमेशा एक *recent* benchmark पर cross-check करो; एक model जो सिर्फ़ पुरानों को ace करे शायद उन्हें memorize किया हो।

---

## 3. `lm-evaluation-harness` — Standard Runner

EleutherAI का `lm-evaluation-harness` (often `lm-eval`) capability evals के लिए open-source standard है।

```bash
pip install lm-eval[vllm]

# एक HF model को tasks के battery पर evaluate करो
lm_eval --model hf \
        --model_args pretrained=Qwen/Qwen3.6-27B,dtype=bfloat16 \
        --tasks mmlu,hellaswag,arc_challenge,gsm8k,humaneval \
        --batch_size auto

# faster: vLLM के via serve, OpenAI-compatible API के via evaluate
vllm serve Qwen/Qwen3.6-27B --quantization fp8 &
lm_eval --model local-completions \
        --model_args base_url=http://localhost:8000/v1,model=Qwen/Qwen3.6-27B \
        --tasks mmlu_pro,gpqa,gsm8k,math \
        --batch_size 256
```

Harness prompt formatting, few-shot examples, और accuracy computation handle करता है। Output reproducible numbers हैं जिन्हें आप published model cards के against compare कर सकते हो।

जब आप अपना khud का model train करो, हर interesting milestone cross करने वाले checkpoint पर lm-eval run करो। Eval-loss curve को benchmark accuracy के साथ plot करो ताकि divergence early catch कर सको।

---

## 4. LLM-as-judge — Open-ended Quality के लिए Workhorse

Chat, summarization, writing, code review — कुछ भी जो multiple-choice नहीं है — आप regex से score नहीं कर सकते। 2026 standard **LLM-as-judge** है: कोई और LLM (usually एक frontier model) आपके model का output पढ़ता है और rate करता है।

### Single-rating

```python
JUDGE_PROMPT = """You are an evaluator. Rate the assistant's response on a 1-5 scale
for {criterion}. 1 = terrible, 5 = excellent. Return only the number.

User question:
{question}

Assistant answer:
{answer}
"""
```

ये कुछ hundred (question, answer) pairs पर run करो और average करो। Cheap, biased, useful।

### Pairwise Win-rate (More Reliable)

Judge को दो answers (आपका vs baseline) दिखाओ और पूछो कौन better है।

```python
PAIRWISE = """Question: {q}

Answer A: {a}
Answer B: {b}

Which is better? Respond with exactly "A" or "B"."""
```

हमेशा **order randomize करो** (position bias के against — judges "A" को over-pick करते हैं)। कम से कम दो different judge models से run करो और average करो। Pairwise win-rate production में dominant metric है: "हम last week के prompt के against 62% जीते" exactly वो है जो stakeholders देखना चाहते हैं।

### MT-Bench / AlpacaEval / Arena-Hard

Public LLM-as-judge benchmarks. **Arena-Hard-Auto** (LMSYS Arena के साथ अच्छा correlate करता है) और **WildBench** 2026 favorites हैं। External models के against sanity check के रूप में use करो।

### Judge Biases — क्या Watch करना है

- **Length bias**: judges longer answers prefer करते हैं। Normalize करो, या ऐसा judge use करो जो इस के against trained है।
- **Position bias**: order matter करता है। हमेशा randomize करो।
- **Self-preference**: same family से model judge करते समय higher rate करता है। जब possible हो *different family* judge use करो।
- **Style bias**: judges formal, well-structured prose prefer करते हैं; अगर आपका product concise replies चाहता है, उसे rubric में build करो।

A useful sanity check: 100 examples human-label करो और judge की humans के साथ agreement check करो। ≥ 80% agreement aim करो; उसके नीचे, आपका rubric work चाहता है।

---

## 5. Golden Set और Regression Tests

ये वो eval है जो आपके product को day-to-day protect करता है।

### Golden Set

A curated list of **50-500 representative examples** known good answers (या judge rubrics) के साथ। Cover करता है:

- आपके users actually जो questions पूछते हैं।
- Edge cases जो past में broke हुए।
- Adversarial prompts।
- हर major use case (different personas, different intents)।

Version control में stored। Quarterly review किया जाता है। **Code की तरह treated।**

### Regression Test

हर prompt change, model upgrade, या harness refactor golden set run करता है। Output: pass/fail (vs. fixed rubric) या previous version vs. win-rate। **CI gate**: ship मत करो अगर win-rate < 50% या कोई high-priority example regress करे।

```python
# pseudo-code: baseline vs golden set run करो
def run_eval(version_a, version_b, golden_set, judge):
    wins = 0
    for example in golden_set:
        a = version_a(example.prompt)
        b = version_b(example.prompt)
        verdict = judge.compare(example.prompt, a, b)
        if verdict == "B": wins += 1
    return wins / len(golden_set)
```

A good golden set **brutal** है। अगर आपका eval वो bug catch नहीं करता जो आपने introduce किया, eval too easy है। हर बार जब आप production में एक failing case पाओ तो उसे add करो — set organically grow होता है।

---

## 6. Online Evals — Users के लिए Only Ones जो Count करते हैं

Offline evals proxies हैं। True measurement only ये है कि क्या real users better outcomes पाते हैं। Common online metrics:

- Responses पर **Thumbs up / thumbs down rate**।
- Assistant के suggestion और user जो actually ship करता है के बीच **Edit distance** (code completion के लिए)।
- **Retention** / **session length** / **return rate**।
- Agents के लिए **Task completion** (क्या user ने अपना goal achieve किया?)।
- Support bots के लिए **Time-to-resolve**।
- A sample पर **CSAT** survey।

Early A/B testing infrastructure set up करो। Full rollout से पहले हर meaningful prompt या model change को एक week के लिए A/B के रूप में run करो। **A/B हर बार intuition को beat करता है।** Tools: GrowthBook, Statsig, LaunchDarkly metric experiments के साथ।

---

## 7. Agent-specific Evaluation

Agents single-turn responses से evaluate करना harder हैं। तीन approaches:

### End-to-end Task Benchmarks

- **SWE-bench Verified / Pro** — real GitHub bug fixes; success = patch project के tests pass करता है।
- **OSWorld** — real desktop apps; success = task completed (file created, email sent, etc.)।
- **GAIA** — multi-step browsing requiring research questions।
- **AppWorld** — multi-app tool use।
- **BrowseComp** — web browsing benchmark।
- **Terminal-Bench 2.0** — autonomous CLI tasks।

इन्हें अपने agent पर nightly CI के रूप में run करो। Score, regress, fix।

### Trajectory Evaluation

Final answer judge मत करो; *trajectory* (tool calls और decisions का sequence) judge करो। इन्हें catch करने के लिए useful:

- **Wasted tool calls** (agent ने web search 6 बार किया जब docs use करनी चाहिए थी)।
- **Wrong tool choice** (`run_shell` use किया जब `read_file` ने काम कर दिया होता)।
- **Loops** — same tool call repeated।
- **Premature giving up**।

LangSmith, Braintrust, और Phoenix सब trajectory inspection support करते हैं। एक small custom scorer कई issues catch कर सकता है:

```python
def trajectory_score(traj):
    score = 1.0
    if len(traj) > 30: score -= 0.3                      # too many steps
    if has_loop(traj): score -= 0.5                       # looping
    if any(t.failed for t in traj): score -= 0.2          # tool errors
    if traj[-1].is_giveup_message: score -= 0.4
    return score
```

### Cost-and-latency Evaluation

Agents जो succeed करते हैं लेकिन 3 hours और $50 लेते हैं वो ship नहीं हो रहे। हमेशा log करो:

- Total tokens (input + output, cached + new)।
- Total cost।
- Wall-clock latency।
- Tool calls, retries, errors की number।

CI में budgets set करो: कोई भी task जो median cost के 10× में complete हो वो flagged।

---

## 8. Eval-driven Development

A 2026 best practice: **prompt या model change से पहले eval लिखो।** ये आपको "different" से पहले "better" define करने पर force करता है।

Loop:

1. Golden set में 5-20 नए examples add करो *new* behavior cover करते हुए।
2. Eval run करो — कई fail (ये correct है; feature अभी built नहीं)।
3. Prompts / fine-tunes / harness तब तक iterate करो जब तक pass न हों।
4. *Full* golden set run करो ताकि कोई regression न हो।
5. Ship।

ये same discipline है जैसे software में TDD, non-determinism पर applied। Eval-driven development वाली team उसके बिना वाली से 2-3× easily out-ship करती है।

---

## 9. Common Eval Pitfalls

**Practice में Goodhart's law।** जैसे ही benchmark target बन जाए, ये useful measure होना बंद हो जाता है। आपका model MMLU ace कर सकता है और फिर भी useless हो सकता है अगर आपने उस पर over-optimize किया। Benchmarks mix करो और rotate करो।

**Train-test contamination।** Benchmarks pretraining data में leak हो जाते हैं। हमेशा `train_eval_data_overlap` (Meta का tool, या n-gram overlap) check करो। *Recent* benchmarks (AIME 2025-26, LiveCodeBench v6, SWE-bench Verified) use करो जो आपके model के training cutoff के बाद हैं।

**Eval-set overfitting।** अगर आप वही set against prompts/models tune करते हो जिस पर measure करते हो, number meaningless है। हमेशा एक *separate* test set hold out करो जिसे आप shipping तक नहीं देखते।

**Brittle string matching।** "The answer is 42." vs "42." एक को fail मत करो। Numeric/extractive tasks के लिए lenient matching use करो।

**Single-judge bias।** एक judge model का अपना bias है। 2+ judges different families से use करो और दोनों numbers report करो।

**Tail behavior ignore करना।** Average misleading है। P95 quality और worst-case examples track करो। mean=4.2 / P5=1.0 वाला model mean=4.0 / P5=3.5 वाले से users के लिए *worse* है।

**Too few examples।** Public papers 200-500 prompts से run करते हैं और इसे done call करते हैं। Production के लिए, per slice 1000+ examples use करो; otherwise variance signal को swamp करती है।

**Expensive judging।** Frontier judges money cost करते हैं। High-volume eval के लिए, judge को distill करो: judge labels पर एक smaller classifier train करो, bulk scoring के लिए small वाले को use करो। Microsoft, Cohere, और Anthropic इस पर papers publish करते हैं।

---

## 10. Tooling — 2026 Stack

| Tool | Use |
|------|-----|
| **`lm-evaluation-harness`** | static capability benchmarks |
| **`promptfoo`** | prompt comparison, CI eval, web UI |
| **LangSmith / Langfuse / Braintrust / Helicone / Phoenix** | online eval, judge runs, dashboards, traces |
| **OpenAI Evals** | OpenAI-style eval framework, easy CI |
| **`inspect-ai`** (UK AISI) | safety + capability eval framework |
| **HELM** (Stanford) | meta-benchmark, broad coverage |
| **Arena-Hard-Auto** | LLM-as-judge harness, LMSYS के साथ correlate करता है |
| **`livebench.ai`** | continually-refreshed benchmark contamination dodge करने को |
| **`olmes`** (AllenAI) | reproducible OLMo-style eval |
| **WildBench / WildEval** | real user prompts, synthetic से harder |
| **Pydantic Logfire / OpenTelemetry GenAI** | structured agent traces |

A typical product setup के लिए: **offline / CI के लिए `promptfoo`**, **online traces के लिए LangSmith या Braintrust**, models pick करते समय capability sanity-checks के लिए **`lm-eval`**। Apna golden set Git-tracked YAML या JSONL में रखो; `promptfoo` directly read करता है।

---

## 11. एक Worked Eval Pipeline

LLM product के लिए आपकी daily eval routine:

```
1. हर PR पर CI
   ├── Static: golden set (50-200 examples) → main branch vs win-rate
   ├── Trajectory checks (agents के लिए)
   └── Latency / cost budget guards

2. Nightly
   ├── Larger golden set (1000+ examples)
   ├── Public benchmark sample (MMLU subset, GSM8K subset)
   ├── Agent benchmarks (SWE-bench Verified subset)
   └── Slices के across cost / latency P95

3. Weekly
   ├── Full golden set + judge (multiple judges)
   ├── User-feedback dashboard review
   ├── Worst-case example deep-dive
   └── Regression analysis (last week vs क्या worse हुआ)

4. Monthly
   ├── External benchmark refresh (contamination dodge करने के लिए rotate)
   ├── Staleness के लिए golden set का 5% re-label
   └── Eval-of-eval: 100 examples human-rate करो, judges के साथ compare करो
```

---

## 12. 2026 Cheat Sheet

- **Three tiers**: capability (benchmarks), quality (judges), reliability (regression)।
- **Eval-driven development** — पहले evals लिखो।
- **Git में Golden set**, code की तरह treated।
- Chat के लिए **Pairwise win-rate** single-rating को beat करता है।
- Different families से **Multiple judges**।
- Contamination dodge करने को **Recent benchmarks**।
- सिर्फ़ mean नहीं, **P95 track करो**।
- Agents के लिए **Trajectory + cost evals**, सिर्फ़ final answer नहीं।
- Users actually क्या चाहते हैं उसके लिए **Online A/B offline eval को beat करता है**।
- Most teams के लिए **`lm-eval` + `promptfoo` + Langfuse/Braintrust** working stack है।

---

## और गहराई से

- **`lm-evaluation-harness`** docs और source — कम से कम एक बार पढ़ो।
- **Arena-Hard-Auto** paper — modern LLM-as-judge benchmark methodology।
- **Anthropic & OpenAI eval cookbooks** — scale पर ये करने वाले लोगों से practical recipes।
- **"Beyond accuracy" papers** (Liang et al., HELM) — evals जो matter करते हैं design करना।
- **"Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"** (Zheng et al. 2023) — judge biases पर canonical paper।
- **`inspect-ai`** UK AISI — open-source safety & capability eval framework।
- **SWE-bench / OSWorld / GAIA** papers — agent eval design।

Next: **[20-fine-tuning-recipes.md](./20-fine-tuning-recipes.md)** — एक base model को exactly वो करवाना जो आप चाहते हो।
