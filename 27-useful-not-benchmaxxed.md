# 27 · Useful, Not Benchmaxxed — Building Models People Actually Want to Use

> **TL;DR** **Most LLM teams optimize the wrong number.** They chase MMLU, AIME, LiveCodeBench, Arena ELO — and ship models that benchmark beautifully but feel terrible to use. This is **Goodhart's Law** in production: as soon as a benchmark becomes the target, it stops measuring what matters. The actually-useful frontier is invisible to leaderboards: low latency, tight instruction following, honest uncertainty, the right output format, knowing when to ask vs. answer, and not padding 30-token responses to 800. **A model that gets 75 % on MMLU but feels right will out-retain a 90 %-MMLU model that doesn't.** This chapter is the philosophy + practice of building for usefulness instead of leaderboards. Read it before you ever publish a benchmark number.

---

## 1. The benchmark trap

A scenario that has played out at every LLM startup since 2023:

> **Week 1:** Team picks MMLU + GSM8K + HumanEval as the eval suite.
> **Week 12:** Model hits frontier numbers on those three. Champagne.
> **Week 13:** Beta users complain the model "feels worse than the old one."
> **Week 14:** Team realizes the model is now great at multiple-choice trivia and synthetic math problems, and slightly worse at *everything users actually do*.
> **Week 15:** Quiet rollback.

This isn't a hypothetical — it's the modal failure mode of LLM development. Every public benchmark, when targeted, breaks the same way:

| Benchmark | What it measures | What it stops measuring once you optimize for it |
|-----------|------------------|---------------------------------------------------|
| **MMLU / MMLU-Pro** | multiple-choice trivia | open-ended reasoning, real-world problem solving |
| **GSM8K / AIME** | school-style word problems | messy, ambiguous quantitative real work |
| **HumanEval / MBPP** | tiny single-function code | full-codebase navigation, refactors, debugging |
| **SWE-bench Verified** | known GitHub bugs with public tests | private codebases, vague tickets, judgment calls |
| **Chatbot Arena ELO** | "vibes" of short chat responses | long tasks, consistency, tool use, agency |
| **MT-Bench / Arena-Hard** | LLM-as-judge "is this answer good" | whether the user's actual goal got achieved |
| **Long-context "needle in a haystack"** | finding a planted sentence | actually using a long doc to think through a problem |

**Goodhart's Law:** *"When a measure becomes a target, it ceases to be a good measure."* It applies with remarkable precision to LLM benchmarks. The Brier score on MMLU goes up, the median user satisfaction goes down. Every. Single. Time.

---

## 2. What "useful" actually means

If not benchmark scores, then what? "Useful" decomposes into seven concrete dimensions, each measurable, none on a public leaderboard.

### 2.1 The user's job got done

The single most important question. Did the user achieve what they sat down to do?

- Did the code run?
- Did the email get a reply?
- Did the document explain the thing?
- Did the bug get fixed?

This is **task completion**, not "answer correctness." MMLU asks "what's the capital of Australia." Real life asks "I'm planning a 10-day trip to Australia next March, can you help me figure out the right cities to visit given that I want to do hiking and avoid crowds." The model that wins on the second is rarely the one that tops the first.

### 2.2 Time-to-useful-output

Latency is a feature, not a bug. A 200 tok/s model that's 90 % as accurate beats a 30 tok/s model that's 95 % as accurate for almost every product because:

- The user's attention budget is finite. Past ~10 seconds of waiting, they switch tabs.
- For agents, latency multiplies across steps. A 30-step agent at 30 tok/s vs 200 tok/s is the difference between "minutes" and "seconds."
- For tab-autocomplete (Cursor / Copilot), TTFT > 200 ms is unusable. A model that aces HumanEval at 50 ms TTFT loses to a less-accurate model at 20 ms TTFT.

You'll find this in chapter 17 §6 (HBM bandwidth) and chapter 24 (tokens/sec). Repeat it as a mantra: **bandwidth is destiny for what your users feel.**

### 2.3 Honest uncertainty

A useful model says "I don't know" when it doesn't, and "I'm sure" when it is. A benchmaxxed model says everything with the same confident voice — because the benchmark didn't measure calibration.

Useful behaviors:
- "I'm not sure about this; let me search."
- "Three plausible answers depending on what you mean by X — which is it?"
- "The docs say A, but my test run showed B. I'd verify before relying on either."

Benchmaxxed behaviors:
- Fabricated citations.
- Confident-sounding wrong answers when the right answer would be "this is unknowable from your prompt."
- "I cannot answer that" when actually the model can but is over-cautious.

### 2.4 Right format, right length

A useful model returns a 5-line answer to a 5-line question, and a 50-line answer to a 50-line question. It doesn't pad. It doesn't truncate.

- Asked for JSON, returns valid JSON, no preamble, no postamble.
- Asked "yes or no?", returns "yes" or "no", maybe one sentence of context.
- Asked to write an essay, writes an essay.
- Asked to refactor a function, returns the refactored function, not a lecture about refactoring.

Benchmaxxed models pad. They've been trained on long verbose answers because human raters often prefer "thorough" over "brief." This is the single most common usability complaint about post-2024 chat models.

### 2.5 Instruction precision

If the user said "use bullet points," the model uses bullet points. If the user said "no headers," there are no headers. If the user said "in the style of Hemingway," it's terse and declarative.

This is **adherence**, and it's hard to measure on a benchmark because compliance is binary per instruction; an MMLU score doesn't capture "did the model follow my 7 stylistic constraints."

### 2.6 Tool use that actually works

Useful agents:
- Pick the right tool the first time.
- Stop when done (don't loop).
- Recover from one failed call without unraveling.
- Say "I tried X, it failed because Y, want me to try Z?" instead of silently retrying.

Benchmaxxed agents:
- Run 30 tools when 3 would do (because the eval rewards "tool use").
- Loop on errors.
- Output a 3-page "I will now think about which tool to use" preamble.
- Pretend to use tools they don't have.

### 2.7 Cost and predictability

Useful = predictable. The same kind of question shouldn't cost $0.02 one week and $1.50 the next because the model decided to "think really hard" this time. Token-budget caps + thinking-mode discipline matter.

A model that *might* be brilliant for $5/query is less useful than one that's *consistently good* at $0.10/query for almost every product.

---

## 3. What real users complain about (vs. what benchmarks miss)

The actual top complaints from chat-product user feedback in 2026, in roughly descending order:

1. **"It writes too much."** Models pad answers. Users abandon mid-scroll.
2. **"It refuses things it shouldn't."** Over-aligned models declining legitimate requests.
3. **"It made stuff up confidently."** Hallucination without uncertainty signals.
4. **"It's slow."** Latency.
5. **"It lectured me when I just wanted the code."** Adding "context" when the user wanted output.
6. **"It changed its mind because I pushed back."** Sycophancy (chapter 26 §5).
7. **"It used the wrong format."** JSON broken, markdown when asked for plain text.
8. **"It forgot what we said earlier."** Long-context performance failures (different from "needle-in-haystack" benchmark).
9. **"It can't follow my style."** Adherence failures.
10. **"It's expensive for what I get."** Cost per useful interaction.

**None of these show up on MMLU, AIME, or HumanEval.** Most of them get *worse* when you optimize for those benchmarks, because:

- Optimizing for "correct answers" trains models to always have an opinion → hallucination + sycophancy.
- Optimizing for "thoroughness" → padding.
- Optimizing for "safety benchmarks" → over-refusal.
- Optimizing for "judge-rated quality" → sycophancy + length bias.
- Optimizing for "thinking benchmarks" → theatrical reasoning that wastes tokens.

This isn't paranoia. It's the **direct measured outcome** of leaderboard-driven development.

---

## 4. The contrastive examples — what useful looks like in practice

### 4.1 The benchmark champion

```
User: What's 2+2?

Model: Great question! Let me think step by step about this.
2+2 is a basic arithmetic operation involving the addition of
two numbers. To solve this, I need to combine the values:

Step 1: Identify the operands (2 and 2).
Step 2: Apply the addition operator.
Step 3: Calculate the sum.

Therefore, 2+2 = 4.

Note: Mathematics is a universal language and basic arithmetic
forms the foundation of more complex operations.
```

92 % MMLU. Zero usefulness. Nobody reads past line 3.

### 4.2 The useful model

```
User: What's 2+2?

Model: 4.
```

Same correctness. 50× less garbage. Users don't switch tabs.

---

### 4.3 The benchmark champion — coding edition

```
User: Add a `--verbose` flag to the script.

Model: I'd be happy to help you add a `--verbose` flag! Adding command-line flags
is a common pattern in Python scripts that allows users to control output
verbosity. Let me walk you through the considerations:

1. We have several options for argument parsing: `argparse` (built-in,
   recommended), `click` (third-party, more powerful), `typer` (FastAPI-style)...

[600 more tokens]

Here's the implementation:
```python
[the actual 4-line change]
```

Would you like me to also add logging configuration, a config file, ...
```

User wanted four lines. Got nine paragraphs.

### 4.4 The useful model — coding edition

```
User: Add a `--verbose` flag to the script.

Model: 
```python
parser.add_argument("--verbose", "-v", action="store_true")
# then use args.verbose to toggle output
```

Want me to wire it into the existing logger?
```

Three lines of code, one offered next step. **This is what users want.** The benchmark on this kind of interaction is "did they accept the suggestion." That's an A/B test, not a leaderboard.

---

## 5. How to evaluate for usefulness

If your eval doesn't measure usefulness, training optimizes you away from it. Five evals worth running, **all of which can be run in your existing tooling without depending on any public leaderboard**.

### 5.1 Task completion rate

Take 100-500 real tasks from your product. For each, define a verifiable success condition (code runs, doc compiles, deal closes, ticket resolves, output format validates). Measure the fraction the model hits.

This is what SWE-bench-style benchmarks try to capture, but at *your* product's distribution. Your "task completion rate on Cursor's actual tasks" is more useful than your SWE-bench Verified score.

### 5.2 User-feedback regression test

Real-user thumbs-up / thumbs-down + free-text comments, aggregated weekly. Run the candidate model on the same prompts as the deployed model, judge with a frontier model + spot-check humans. **Win-rate vs. last-shipped is the headline metric.**

Tool: LangSmith, Braintrust, Helicone, Phoenix all expose this dashboard.

### 5.3 Length-controlled win-rate

Pairwise judge eval, but **penalize length bias**. Anthropic's "length-controlled win-rate" technique: regress out the length effect from the judge's preference.

Without this, you'll think your model improved when actually it just got more verbose.

### 5.4 Refusal/false-refusal eval

Two contrasting sets:
- 100 prompts that *should* be refused (genuinely harmful).
- 100 prompts that look superficially adjacent but are legitimate ("how do I make my password unhackable" — tech support, not exploit).

Measure refusal rate on both. **You want high on the first, low on the second.** Most "safety" optimization breaks the second.

### 5.5 Latency-and-cost-controlled performance

For every benchmark or eval, normalize by tokens used and by latency. A model that gets 70 % on a task in 200 tokens is genuinely better than a model that gets 75 % in 2000 tokens — even though the second has the higher headline number.

Build this into your eval CI. Always report `(score, mean_tokens, mean_latency)`.

---

## 6. How to train for usefulness (and avoid benchmaxxing)

### 6.1 Data: real over synthetic

If you can use real user prompts (with consent and privacy controls), they beat any synthetic dataset for SFT. Synthetic data gives you a model that's good at *the synthetic distribution*. Real-user data gives you a model that's good at your actual product.

When you must use synthetic, **write the prompts the way users actually write** (typos, half-sentences, abbreviated, no preamble), not the way GPT-4 generates them (full grammar, polite phrasing, "I would like to know if you could please...").

### 6.2 Don't train on benchmark-shaped data

The fastest path to overfit:

```python
# DON'T
def make_sft_data():
    return [
        format_as_mmlu(...),
        format_as_humaneval(...),
        format_as_gsm8k(...),
    ]
```

Yes, your benchmarks will go up. Yes, your real users will hate the model. This is how leaderboards have collapsed into noise: too many teams trained on benchmark-shaped data.

If you must include benchmark-style examples, keep them at <5 % of the SFT mix and never DPO toward them.

### 6.3 Decontamination, but real

Standard 13-gram decontamination misses most leakage. Use:
- N-gram overlap at multiple lengths.
- Embedding-level similarity (a paraphrased benchmark question still embeds near the original).
- A held-out *private* benchmark you build yourself from real user data — never publish it, never include it in training.

The held-out private benchmark is the one number you actually trust.

### 6.4 Reward what you want, not what's easy to measure

For RLVR / GRPO, write a reward that measures usefulness, not correctness:

```python
def usefulness_reward(prompt, response, ground_truth):
    score = 0.0
    if answer_correct(response, ground_truth):
        score += 1.0
    if response_length(response) > expected_length(prompt) * 2:
        score -= 0.3                       # penalize padding
    if uses_appropriate_format(prompt, response):
        score += 0.3
    if has_unnecessary_disclaimers(response):
        score -= 0.5
    if asks_clarifying_question_when_ambiguous(prompt, response):
        score += 0.2
    return score
```

Almost no team does this. Most of them just reward `answer_correct` and wonder why their model lectures.

### 6.5 Length penalties in DPO

When constructing preference pairs, **make sure the chosen and rejected are length-matched**, or DPO learns "longer = better." Anthropic, OpenAI, and Cohere all have explicit length-controlled DPO recipes now (~2024-2026); use them.

### 6.6 Ship to staging, not main

A version that beats yours on offline evals can still lose in production. Always:
- A/B test in staging on real traffic for at least 1 week.
- Track the seven complaints in §3 explicitly.
- Roll back if user satisfaction drops, even if benchmarks went up.

This is uncomfortable. The benchmark went up. The leaderboard rewards you. Your CEO's slide deck wants the bigger number. Roll back anyway. **The only honest signal is whether real users prefer it.**

---

## 7. The anti-pattern catalog (smell tests for a benchmaxxed model)

If your model does these, you've over-optimized for benchmarks:

1. **"Great question!"** — almost all the time.
2. **"Let me think step by step"** — for trivial questions.
3. **"Note: this is for educational purposes only"** — on harmless requests.
4. **Comprehensive overviews** when a one-liner was asked.
5. **Hedging** ("It depends...", "There are many ways...") on factual questions.
6. **Refusing technical questions** that touch any sensitive word.
7. **Apologizing for past answers** when the user pushes back, even when right.
8. **Stating confidence on hallucinated facts.**
9. **Using the same answer template** regardless of question style.
10. **Producing thinking blocks longer than the answer**, on questions that don't need thinking.
11. **Long preambles before code**, every time.
12. **Refusing to commit to a recommendation** when the user explicitly asked for one.
13. **Tool-use loops** when no tool is needed.
14. **Following a benchmark format** when the user wants free-form.

Each of these is a symptom. Each came from optimizing the wrong eval signal.

---

## 8. The pattern catalog (smell tests for a useful model)

These are quiet, easily missed, and they're what users actually love:

1. **Length matches question length.** Not always. Mostly.
2. **Asks "X or Y?"** when the prompt is genuinely ambiguous, instead of guessing.
3. **Says "I don't know — want me to look it up?"** instead of fabricating.
4. **Returns code in the format requested** (with or without comments, with or without imports).
5. **Stops talking when the answer is given.**
6. **Names its uncertainty:** "I'm pretty sure...", "I think but verify..."
7. **Disagrees with the user** when the user is wrong (no sycophancy).
8. **Uses tools sparingly** but reaches for them when truly needed.
9. **Remembers the chat's context** without re-explaining it.
10. **Owns up:** "I was wrong about X earlier; here's the correct answer."
11. **Follows constraints** ("under 100 words," "no markdown," "in JSON") on the first try.
12. **Knows its own latency budget** — short answer when speed matters, long when depth matters.

A model that passes most of these will out-retain a model that aces every public benchmark. **Build for these.**

---

## 9. Real-product case studies

### 9.1 Cursor / Windsurf

What they emphasize publicly: **Tab-completion latency, codebase indexing, inline diffs.** What they barely mention: the underlying model's HumanEval score. Why? Because their users don't care about HumanEval. They care about whether tab-complete *feels* native.

Their internal eval suite reportedly tests:
- Per-character latency under load.
- Acceptance rate of suggestions on real customer codebases.
- Edit distance between suggestion and what the user kept.
- Time-to-task-complete on instrumented real workflows.

Zero of those are public benchmarks. **All of them are useful metrics.**

### 9.2 Claude Code

Claude Code (the CLI) does talk about SWE-bench publicly, but the team has been explicit that they optimize for:
- "How often does the agent finish without me having to intervene?"
- "Does it write the right amount of code?"
- "Does it ask before destructive actions?"
- "Does it maintain context across the session?"

Anthropic's user-facing complaint reduction is more about behavior shape (sycophancy, padding, over-refusal) than benchmark scores. They publish small benchmark wins; they don't publish what most of post-training is actually doing.

### 9.3 ChatGPT vs. all the "we beat GPT-4" releases

The public LLM cemetery is full of models that beat GPT-4 on benchmarks and still got zero adoption. ChatGPT's moat was never "highest MMLU." It was: low latency, predictable behavior, decent UX, and a massive feedback loop turning real-user complaints into improvements faster than any competitor.

The lesson: **benchmark numbers move spreadsheets, retention moves businesses.** If you're writing a press release about your benchmark wins, you're behind on the work that actually matters.

### 9.4 The reasoning-model usefulness paradox

Reasoning models (o1-style, R1, Qwen3-Thinking) ace AIME and GPQA. Real users use them for chat *less*, because:
- Their thinking blocks waste seconds on questions that don't need them.
- Their final answers are often longer and more hedged.
- Their cost per query is 5-20× a non-thinking model.

Useful 2026 deployments use **routed** reasoning: easy queries → fast non-thinking model, hard queries → reasoning model. Forcing reasoning everywhere is benchmaxxing the wrong axis.

---

## 10. The one-line philosophy

If you remember one thing from this chapter:

> **Build the model your user would pay for using their own money — not the model that wins a leaderboard you wouldn't pay to read.**

Everything in §§1-9 follows from that.

---

## 11. The 2026 cheat sheet

- **Goodhart's Law applies to every benchmark.** Every. Single. One.
- **MMLU / AIME / HumanEval are weak proxies.** Use them as sanity checks, not goals.
- **Real-user task completion rate** is the only metric that actually matters.
- **Latency, format adherence, honesty, length-discipline** are all part of "useful." Measure them.
- **Don't train on benchmark-shaped data.** Less than 5 % of SFT mix at most.
- **Length-control your DPO.** Otherwise you teach the model to pad.
- **Reward usefulness, not correctness.** Penalize padding, hallucinated confidence, unnecessary disclaimers.
- **Build a private held-out benchmark from real user data.** Never publish it. That's the one number you trust.
- **A/B in production for a week** before shipping any model improvement.
- **Roll back when users hate the new model**, even if your offline numbers went up.
- **Route reasoning to where it helps**, not everywhere.
- **Latency is a feature.** Treat HBM bandwidth, quantization, and routing as first-class usefulness levers.
- **The smell test of a benchmaxxed model**: padding, theatrics, sycophancy, over-refusal, "Great question!"
- **The smell test of a useful model**: short when short is right, asks when ambiguous, says "I don't know" when it doesn't.

---

## 12. Going deeper

- **"Goodhart's Law in Reinforcement Learning"** — Skalse et al. The formal version of why this happens.
- **Anthropic's "Sleeper Agents"** and **"Sycophancy"** posts — case studies of benchmark-driven failure modes.
- **Length-controlled win-rate** — the methodology paper from Dubois et al. 2024.
- **Tülu 3 technical report** (AI2) — explicitly discusses choosing useful over benchmarked metrics.
- **Cursor / Continue / Aider blog posts on latency** — reading these is more practical than any leaderboard.
- **HELM "Holistic Evaluation"** — Stanford's attempt at a multi-dimensional eval that's harder to gamify.
- **The Chatbot Arena research papers** — and their honest acknowledgments of judge biases.

This chapter is the philosophical complement to the rest of the book. Chapters 04 (data), 14 (training), 19 (eval), and 25-26 (release & debug) tell you *how* to build models. **Chapter 27 tells you *what for*.** Without it, the rest of the curriculum optimizes a number that doesn't matter.

Build models people use. The leaderboards will fade; the retained users won't.
