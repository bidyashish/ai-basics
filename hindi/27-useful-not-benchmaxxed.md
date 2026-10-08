# 27 · Useful, Not Benchmaxxed — Models बनाओ जो Log Actually Use करना चाहें

> **TL;DR** **ज़्यादातर LLM teams ग़लत number optimize करती हैं।** वो MMLU, AIME, LiveCodeBench, Arena ELO chase करती हैं — और models ship करती हैं जो benchmark beautifully करते हैं लेकिन use करने में terrible feel होते हैं। ये **Goodhart's Law** है production में: जैसे ही एक benchmark target बनती है, वो वो measure करना बंद कर देती है जो matter करता है। Actually-useful frontier leaderboards के लिए invisible है: low latency, tight instruction following, honest uncertainty, right output format, कब ask करना बनाम answer देना, और 30-token responses को 800 तक pad न करना। **एक model जो MMLU पर 75 % पाता है लेकिन right feel होता है वो 90 %-MMLU model से out-retain करेगा जो नहीं।** ये chapter benchmaxxing के बजाय usefulness के लिए build करने का philosophy + practice है। कोई भी benchmark number publish करने से पहले इसे पढ़ो।

---

## 1. Benchmark Trap

एक scenario जो 2023 के बाद हर LLM startup में play out हुआ है:

> **Week 1:** Team eval suite के रूप में MMLU + GSM8K + HumanEval pick करती है।
> **Week 12:** Model उन तीनों पर frontier numbers hit करता है। Champagne।
> **Week 13:** Beta users complain करते हैं model "old वाले से worse feel होता है।"
> **Week 14:** Team realize करती है कि model अब multiple-choice trivia और synthetic math problems पर great है, और *जो भी users actually करते हैं* उस पर slightly worse है।
> **Week 15:** Quiet rollback।

ये hypothetical नहीं है — ये LLM development का modal failure mode है। हर public benchmark, target किए जाने पर, same way break होती है:

| Benchmark | क्या measure करती है | Optimize करने पर क्या measure करना बंद करती है |
|-----------|------------------|---------------------------------------------------|
| **MMLU / MMLU-Pro** | multiple-choice trivia | open-ended reasoning, real-world problem solving |
| **GSM8K / AIME** | school-style word problems | messy, ambiguous quantitative real work |
| **HumanEval / MBPP** | tiny single-function code | full-codebase navigation, refactors, debugging |
| **SWE-bench Verified** | public tests वाले known GitHub bugs | private codebases, vague tickets, judgment calls |
| **Chatbot Arena ELO** | short chat responses की "vibes" | long tasks, consistency, tool use, agency |
| **MT-Bench / Arena-Hard** | LLM-as-judge "क्या ये answer good है" | क्या user का actual goal achieved हुआ |
| **Long-context "needle in a haystack"** | एक planted sentence find करना | actually एक long doc use करके problem solve करना |

**Goodhart's Law:** *"जब एक measure target बन जाती है, तो वो good measure होना बंद हो जाती है।"* ये LLM benchmarks पर remarkable precision के साथ apply होता है। MMLU पर Brier score ऊपर जाता है, median user satisfaction नीचे जाती है। हर। Single। बार।

---

## 2. "Useful" का Actually मतलब क्या है

अगर benchmark scores नहीं तो क्या? "Useful" सात concrete dimensions में decompose होता है, हर एक measurable, कोई public leaderboard पर नहीं।

### 2.1 User का Job हो गया

Single most important question। क्या user ने वो achieve किया जो वो करने बैठा था?

- क्या code run हुआ?
- क्या email को reply मिला?
- क्या document ने वो thing explain की?
- क्या bug fix हुआ?

ये **task completion** है, "answer correctness" नहीं। MMLU पूछता है "Australia की capital क्या है।" Real life पूछती है "मैं अगले March 10 दिन के Australia trip plan कर रहा हूं, क्या आप right cities figure out करने में help कर सकते हो ये देखते हुए कि मैं hiking करना चाहता हूं और crowds avoid करना चाहता हूं।" वो model जो दूसरे पर wins करता है rarely वो होता है जो पहले पर top करता है।

### 2.2 Time-to-useful-output

Latency एक feature है, bug नहीं। एक 200 tok/s model जो 90% accurate है वो 30 tok/s model जो 95% accurate है से almost हर product के लिए beat करता है क्योंकि:

- User का attention budget finite है। ~10 seconds waiting के बाद, वो tabs switch करते हैं।
- Agents के लिए, latency steps के across multiply होती है। 30 tok/s vs 200 tok/s पर 30-step agent "minutes" और "seconds" का difference है।
- Tab-autocomplete (Cursor / Copilot) के लिए, TTFT > 200 ms unusable है। एक model जो 50 ms TTFT पर HumanEval ace करता है वो less-accurate model से 20 ms TTFT पर loses है।

आप ये chapter 17 §6 (HBM bandwidth) और chapter 24 (tokens/sec) में पाओगे। एक mantra की तरह repeat करो: **जो आपके users feel करते हैं उसके लिए bandwidth destiny है।**

### 2.3 Honest Uncertainty

A useful model कहता है "मुझे नहीं पता" जब उसे नहीं पता, और "मैं sure हूं" जब है। A benchmaxxed model सब कुछ same confident voice में कहता है — क्योंकि benchmark calibration measure नहीं करती।

Useful behaviors:
- "मुझे इस बारे में sure नहीं; मुझे search करने दो।"
- "तीन plausible answers depending on आप X से क्या मतलब करते हो — कौन सा है?"
- "Docs A कहते हैं, लेकिन मेरा test run B दिखाया। मैं किसी पर भी relying से पहले verify करूंगा।"

Benchmaxxed behaviors:
- Fabricated citations।
- Confident-sounding wrong answers जब right answer "ये आपके prompt से unknowable है" होता।
- "मैं वो answer नहीं कर सकता" जब actually model कर सकता है लेकिन over-cautious है।

### 2.4 Right Format, Right Length

A useful model 5-line question को 5-line answer return करता है, और 50-line question को 50-line answer। ये pad नहीं करता। ये truncate नहीं करता।

- JSON ask करने पर, valid JSON return करता है, no preamble, no postamble।
- "yes या no?" ask करने पर, "yes" या "no" return करता है, maybe context का एक sentence।
- Essay लिखने को ask करने पर, essay लिखता है।
- Function refactor करने को ask करने पर, refactored function return करता है, refactoring पर lecture नहीं।

Benchmaxxed models pad करते हैं। उन्हें long verbose answers पर trained किया गया है क्योंकि human raters often "thorough" को "brief" से prefer करते हैं। ये post-2024 chat models के बारे में single most common usability complaint है।

### 2.5 Instruction Precision

अगर user ने कहा "bullet points use करो," model bullet points use करता है। अगर user ने कहा "no headers," कोई headers नहीं हैं। अगर user ने कहा "in the style of Hemingway," ये terse और declarative है।

ये **adherence** है, और इसे एक benchmark पर measure करना hard है क्योंकि compliance per instruction binary है; एक MMLU score "क्या model ने मेरे 7 stylistic constraints follow किए" capture नहीं करता।

### 2.6 Tool Use जो Actually काम करे

Useful agents:
- First time right tool pick करते हैं।
- Done होने पर stop करते हैं (loop नहीं)।
- एक failed call से unraveling के बिना recover करते हैं।
- Silently retrying के बजाय "मैंने X try किया, ये Y की वजह से failed, क्या मैं Z try करूं?" कहते हैं।

Benchmaxxed agents:
- 30 tools run करते हैं जब 3 enough थे (क्योंकि eval "tool use" reward करती है)।
- Errors पर loop करते हैं।
- एक 3-page "मैं अब सोचूंगा कौन सा tool use करूं" preamble output करते हैं।
- ऐसे tools use करने का pretend करते हैं जो उनके पास नहीं हैं।

### 2.7 Cost और Predictability

Useful = predictable। Same kind का question एक week $0.02 और दूसरे $1.50 cost नहीं करना चाहिए क्योंकि model ने decide किया इस बार "really hard think" करूं। Token-budget caps + thinking-mode discipline matter करते हैं।

A model जो *might* $5/query के लिए brilliant है वो one less useful है जो $0.10/query पर almost हर product के लिए *consistently good* है।

---

## 3. Real Users क्या Complain करते हैं (vs. Benchmarks क्या Miss करते हैं)

2026 में chat-product user feedback से actual top complaints, roughly descending order में:

1. **"ये बहुत ज़्यादा लिखता है।"** Models answers pad करते हैं। Users mid-scroll abandon करते हैं।
2. **"ये उन चीज़ों को refuse करता है जिन्हें नहीं करना चाहिए।"** Over-aligned models legitimate requests decline कर रहे।
3. **"इसने confidently चीज़ें बना दीं।"** Uncertainty signals के बिना hallucination।
4. **"ये slow है।"** Latency।
5. **"इसने मुझे lecture दी जब मैं बस code चाहता था।"** "Context" add करना जब user output चाहता था।
6. **"मैंने push back किया तो इसने अपना mind change किया।"** Sycophancy (chapter 26 §5)।
7. **"इसने wrong format use किया।"** JSON broken, plain text ask करने पर markdown।
8. **"ये भूल गया हमने earlier क्या कहा था।"** Long-context performance failures ("needle-in-haystack" benchmark से अलग)।
9. **"ये मेरा style follow नहीं कर सकता।"** Adherence failures।
10. **"जो मुझे मिलता है उसके लिए ये expensive है।"** Per useful interaction cost।

**इनमें से कोई भी MMLU, AIME, या HumanEval पर show नहीं होता।** उनमें से ज़्यादातर तब *worse* हो जाते हैं जब आप उन benchmarks के लिए optimize करते हो, क्योंकि:

- "Correct answers" के लिए optimizing models को हमेशा एक opinion रखने सिखाता है → hallucination + sycophancy।
- "Thoroughness" के लिए optimizing → padding।
- "Safety benchmarks" के लिए optimizing → over-refusal।
- "Judge-rated quality" के लिए optimizing → sycophancy + length bias।
- "Thinking benchmarks" के लिए optimizing → theatrical reasoning जो tokens waste करता है।

ये paranoia नहीं है। ये leaderboard-driven development का **directly measured outcome** है।

---

## 4. Contrastive Examples — Useful Practice में कैसा दिखता है

### 4.1 Benchmark Champion

```
User: 2+2 क्या है?

Model: Great question! इसके बारे में step by step सोचने दो।
2+2 दो numbers की addition involve करने वाला एक basic arithmetic
operation है। इसे solve करने के लिए, मुझे values combine करनी हैं:

Step 1: Operands identify करो (2 और 2)।
Step 2: Addition operator apply करो।
Step 3: Sum calculate करो।

इसलिए, 2+2 = 4।

Note: Mathematics एक universal language है और basic arithmetic
ज़्यादा complex operations की foundation बनती है।
```

92% MMLU। Zero usefulness। कोई line 3 के बाद नहीं पढ़ता।

### 4.2 Useful Model

```
User: 2+2 क्या है?

Model: 4।
```

Same correctness। 50× less garbage। Users tabs switch नहीं करते।

---

### 4.3 Benchmark Champion — Coding Edition

```
User: Script में एक `--verbose` flag add करो।

Model: मैं आपको `--verbose` flag add करने में happy हूं! Command-line flags add
करना Python scripts में एक common pattern है जो users को output verbosity
control करने देता है। Considerations आपके through walk करूं:

1. हमारे पास argument parsing के लिए कई options हैं: `argparse` (built-in,
   recommended), `click` (third-party, more powerful), `typer` (FastAPI-style)...

[600 more tokens]

यहां implementation है:
```python
[the actual 4-line change]
```

क्या आप चाहते हो मैं logging configuration भी add करूं, एक config file...
```

User चार lines चाहता था। नौ paragraphs मिले।

### 4.4 Useful Model — Coding Edition

```
User: Script में एक `--verbose` flag add करो।

Model: 
```python
parser.add_argument("--verbose", "-v", action="store_true")
# फिर output toggle करने के लिए args.verbose use करो
```

क्या आप चाहते हो मैं इसे existing logger में wire करूं?
```

तीन lines code, एक offered next step। **यही users चाहते हैं।** इस तरह के interaction पर benchmark "क्या उन्होंने suggestion accept किया" है। ये एक A/B test है, leaderboard नहीं।

---

## 5. Usefulness के लिए Evaluate कैसे करें

अगर आपकी eval usefulness measure नहीं करती, training आपको उससे away optimize करती है। पांच evals जो run करने worth हैं, **सब अपनी existing tooling में run हो सकती हैं किसी भी public leaderboard पर depend किए बिना**।

### 5.1 Task Completion Rate

अपने product से 100-500 real tasks लो। हर एक के लिए, एक verifiable success condition define करो (code runs, doc compiles, deal closes, ticket resolves, output format validates)। Fraction measure करो जो model hit करता है।

ये वो है जो SWE-bench-style benchmarks capture करने try करते हैं, लेकिन *आपके* product के distribution पर। आपका "Cursor के actual tasks पर task completion rate" आपके SWE-bench Verified score से ज़्यादा useful है।

### 5.2 User-feedback Regression Test

Real-user thumbs-up / thumbs-down + free-text comments, weekly aggregated। Candidate model को deployed model वाले same prompts पर run करो, frontier model + spot-check humans से judge करो। **Last-shipped vs. Win-rate headline metric है।**

Tool: LangSmith, Braintrust, Helicone, Phoenix सब ये dashboard expose करते हैं।

### 5.3 Length-controlled Win-rate

Pairwise judge eval, लेकिन **length bias penalize करो**। Anthropic की "length-controlled win-rate" technique: judge की preference से length effect regress out करो।

इसके बिना, आप सोचोगे आपका model improve हुआ जब actually ये बस ज़्यादा verbose हो गया।

### 5.4 Refusal/false-refusal Eval

दो contrasting sets:
- 100 prompts जो *should* refuse हों (genuinely harmful)।
- 100 prompts जो superficially adjacent दिखते हैं लेकिन legitimate हैं ("मैं अपना password unhackable कैसे बनाऊं" — tech support, exploit नहीं)।

दोनों पर refusal rate measure करो। **आप पहले पर high चाहते हो, दूसरे पर low।** ज़्यादातर "safety" optimization दूसरे को break करती है।

### 5.5 Latency-and-cost-controlled Performance

हर benchmark या eval के लिए, used tokens और latency से normalize करो। एक model जो 200 tokens में एक task पर 70% पाता है वो 2000 tokens में 75% पाने वाले model से genuinely better है — even though दूसरे का higher headline number है।

इसे अपनी eval CI में build करो। हमेशा `(score, mean_tokens, mean_latency)` report करो।

---

## 6. Usefulness के लिए कैसे Train करें (और Benchmaxxing से बचें)

### 6.1 Data: Synthetic से Real

अगर आप real user prompts use कर सकते हो (consent और privacy controls के साथ), वो SFT के लिए किसी भी synthetic dataset को beat करते हैं। Synthetic data आपको एक model देता है जो *synthetic distribution* पर अच्छा है। Real-user data आपको एक model देता है जो आपके actual product पर अच्छा है।

जब आपको synthetic use करना ही पड़े, **prompts ऐसे लिखो जैसे users actually लिखते हैं** (typos, half-sentences, abbreviated, no preamble), GPT-4 जैसे generate करता है (full grammar, polite phrasing, "मैं जानना चाहूंगा अगर आप कृपया कर सकें...") नहीं।

### 6.2 Benchmark-shaped Data पर Train मत करो

Overfit का fastest path:

```python
# मत करो
def make_sft_data():
    return [
        format_as_mmlu(...),
        format_as_humaneval(...),
        format_as_gsm8k(...),
    ]
```

हां, आपकी benchmarks ऊपर जाएंगी। हां, आपके real users model से नफ़रत करेंगे। ये है कैसे leaderboards noise में collapse हो गई हैं: too many teams ने benchmark-shaped data पर train किया।

अगर आपको benchmark-style examples include करने ही हैं, उन्हें SFT mix के <5% पर रखो और कभी उनकी ओर DPO मत करो।

### 6.3 Decontamination, लेकिन Real

Standard 13-gram decontamination ज़्यादातर leakage miss करता है। Use करो:
- Multiple lengths पर N-gram overlap।
- Embedding-level similarity (एक paraphrased benchmark question अभी भी original के पास embed होती है)।
- एक held-out *private* benchmark जो आप real user data से खुद build करते हो — कभी publish मत करो, कभी training में include मत करो।

Held-out private benchmark वो number है जिस पर आप actually trust करते हो।

### 6.4 जो आप चाहते हो उसे Reward करो, जो measure करना easy है उसे नहीं

RLVR / GRPO के लिए, एक reward लिखो जो usefulness measure करे, correctness नहीं:

```python
def usefulness_reward(prompt, response, ground_truth):
    score = 0.0
    if answer_correct(response, ground_truth):
        score += 1.0
    if response_length(response) > expected_length(prompt) * 2:
        score -= 0.3                       # padding penalize
    if uses_appropriate_format(prompt, response):
        score += 0.3
    if has_unnecessary_disclaimers(response):
        score -= 0.5
    if asks_clarifying_question_when_ambiguous(prompt, response):
        score += 0.2
    return score
```

Almost कोई team ये नहीं करती। उनमें से ज़्यादातर बस `answer_correct` reward करते हैं और wonder करते हैं उनका model lecture क्यों देता है।

### 6.5 DPO में Length Penalties

Preference pairs construct करते समय, **make sure chosen और rejected length-matched हैं**, otherwise DPO "longer = better" सीखता है। Anthropic, OpenAI, और Cohere सब के पास अब explicit length-controlled DPO recipes हैं (~2024-2026); उन्हें use करो।

### 6.6 Main को नहीं, Staging को Ship करो

A version जो आपको offline evals पर beat करता है production में अभी भी lose कर सकता है। हमेशा:
- Real traffic पर at least 1 week के लिए staging में A/B test करो।
- §3 के सात complaints को explicitly track करो।
- Roll back करो अगर user satisfaction drop हो, even if benchmarks ऊपर गए।

ये uncomfortable है। Benchmark ऊपर गई। Leaderboard आपको reward करता है। आपके CEO का slide deck bigger number चाहता है। फिर भी roll back करो। **The only honest signal real users prefer करते हैं या नहीं ये है।**

---

## 7. Anti-pattern Catalog (Benchmaxxed Model के Smell Tests)

अगर आपका model ये करता है, तो आपने benchmarks के लिए over-optimize किया है:

1. **"Great question!"** — almost हर समय।
2. **"Let me think step by step"** — trivial questions के लिए।
3. **"Note: this is for educational purposes only"** — harmless requests पर।
4. **Comprehensive overviews** जब one-liner ask किया था।
5. **Hedging** ("It depends...", "There are many ways...") factual questions पर।
6. **Technical questions Refuse करना** जो किसी sensitive word को touch करते हैं।
7. **Past answers के लिए Apologize करना** जब user push back करे, even when right।
8. **Hallucinated facts पर confidence state करना।**
9. **Same answer template use करना** question style के regardless।
10. **Answer से longer thinking blocks produce करना**, जिन questions को thinking नहीं चाहिए।
11. **हर बार code से पहले long preambles**।
12. **एक recommendation के लिए commit refuse करना** जब user ने explicitly एक के लिए ask किया।
13. **Tool-use loops** जब कोई tool नहीं चाहिए।
14. **एक benchmark format follow करना** जब user free-form चाहता है।

इनमें से हर एक एक symptom है। हर एक wrong eval signal optimize करने से आया।

---

## 8. Pattern Catalog (Useful Model के Smell Tests)

ये quiet हैं, easily missed हैं, और users actually इन्हें love करते हैं:

1. **Length question length से match करती है।** Always नहीं। Mostly।
2. **जब prompt genuinely ambiguous हो "X या Y?" ask करता है**, guess करने के बजाय।
3. **"मुझे नहीं पता — क्या मैं look up करूं?" कहता है** fabricate करने के बजाय।
4. **Code requested format में return करता है** (comments के साथ या बिना, imports के साथ या बिना)।
5. **जब answer दे दिया तो बात करना stop करता है।**
6. **अपनी uncertainty name करता है:** "मैं pretty sure हूं...", "मुझे लगता है लेकिन verify करो..."
7. **User से Disagree करता है** जब user wrong है (no sycophancy)।
8. **Tools sparingly use करता है** लेकिन जब truly needed हो तो उनके लिए reach करता है।
9. **Chat के context को remember करता है** उसे re-explain किए बिना।
10. **Owns up:** "मैं X के बारे में earlier wrong था; यहां correct answer है।"
11. **Constraints follow करता है** ("100 words के नीचे," "no markdown," "in JSON") first try पर।
12. **अपनी latency budget जानता है** — जब speed matter करे short answer, जब depth matter करे long।

A model जो इनमें से ज़्यादातर pass करे वो every public benchmark ace करने वाले model से out-retain करेगा। **इनके लिए build करो।**

---

## 9. Real-product Case Studies

### 9.1 Cursor / Windsurf

वो publicly क्या emphasize करते हैं: **Tab-completion latency, codebase indexing, inline diffs।** वो barely क्या mention करते हैं: underlying model का HumanEval score। क्यों? क्योंकि उनके users HumanEval के बारे में care नहीं करते। वो care करते हैं क्या tab-complete native *feel* होता है।

उनकी internal eval suite reportedly test करती है:
- Load के तहत per-character latency।
- Real customer codebases पर suggestions की acceptance rate।
- Suggestion और जो user ने keep किया उसके बीच edit distance।
- Instrumented real workflows पर time-to-task-complete।

इनमें से zero public benchmarks हैं। **उनमें से सब useful metrics हैं।**

### 9.2 Claude Code

Claude Code (CLI) publicly SWE-bench के बारे में बात करता है, लेकिन team explicit रही है कि वो optimize करती है:
- "कितनी बार agent मुझे intervene करवाए बिना finish करता है?"
- "क्या ये right amount of code लिखता है?"
- "क्या ये destructive actions से पहले ask करता है?"
- "क्या ये session के across context maintain करता है?"

Anthropic का user-facing complaint reduction benchmark scores से ज़्यादा behavior shape (sycophancy, padding, over-refusal) के बारे में है। वो small benchmark wins publish करते हैं; वो ज़्यादातर post-training actually क्या कर रही है ये publish नहीं करते।

### 9.3 ChatGPT vs. सारे "हमने GPT-4 को beat किया" Releases

Public LLM cemetery models से full है जिन्होंने benchmarks पर GPT-4 को beat किया और फिर भी zero adoption मिला। ChatGPT का moat कभी "highest MMLU" नहीं था। ये था: low latency, predictable behavior, decent UX, और एक massive feedback loop जो real-user complaints को किसी भी competitor से faster improvements में बदलता है।

Lesson: **benchmark numbers spreadsheets move करते हैं, retention businesses move करता है।** अगर आप अपनी benchmark wins के बारे में press release लिख रहे हो, आप उस work पर behind हो जो actually matter करता है।

### 9.4 Reasoning-model Usefulness Paradox

Reasoning models (o1-style, R1, Qwen3-Thinking) AIME और GPQA ace करते हैं। Real users उनका chat के लिए *less* use करते हैं, क्योंकि:
- उनके thinking blocks उन questions पर seconds waste करते हैं जिन्हें नहीं चाहिए।
- उनके final answers often longer और more hedged होते हैं।
- उनका per query cost एक non-thinking model का 5-20× है।

Useful 2026 deployments **routed** reasoning use करते हैं: easy queries → fast non-thinking model, hard queries → reasoning model। Reasoning को हर जगह force करना wrong axis को benchmaxx करना है।

---

## 10. One-line Philosophy

अगर आप इस chapter से एक चीज़ remember करो:

> **वो model build करो जिसके लिए आपका user अपने खुद के पैसे से pay करेगा — वो model नहीं जो एक leaderboard win करता है जिसे आप पढ़ने के लिए pay नहीं करोगे।**

§§1-9 में सब कुछ उससे follow होता है।

---

## 11. 2026 Cheat Sheet

- **Goodhart's Law हर benchmark पर applies होता है।** हर। Single। एक।
- **MMLU / AIME / HumanEval weak proxies हैं।** उन्हें sanity checks के रूप में use करो, goals नहीं।
- **Real-user task completion rate** एकमात्र metric है जो actually matter करती है।
- **Latency, format adherence, honesty, length-discipline** सब "useful" का part हैं। उन्हें measure करो।
- **Benchmark-shaped data पर train मत करो।** SFT mix का at most 5% से कम।
- **अपने DPO को length-control करो।** Otherwise आप model को pad करना सिखाते हो।
- **Usefulness reward करो, correctness नहीं।** Padding, hallucinated confidence, unnecessary disclaimers penalize करो।
- **Real user data से एक private held-out benchmark build करो।** कभी publish मत करो। यही एक number है जिस पर आप trust करते हो।
- **एक week production में A/B करो** किसी भी model improvement ship करने से पहले।
- **जब users new model से नफ़रत करें Roll back करो**, even if your offline numbers ऊपर गए।
- **Reasoning वहां route करो जहां ये help करे**, हर जगह नहीं।
- **Latency एक feature है।** HBM bandwidth, quantization, और routing को first-class usefulness levers के रूप में treat करो।
- **Benchmaxxed model का smell test**: padding, theatrics, sycophancy, over-refusal, "Great question!"।
- **Useful model का smell test**: short जब short right है, ambiguous हो तो asks, "मुझे नहीं पता" कहता है जब नहीं पता।

---

## 12. और गहराई से

- **"Goodhart's Law in Reinforcement Learning"** — Skalse et al. ये क्यों होता है का formal version।
- **Anthropic's "Sleeper Agents"** और **"Sycophancy"** posts — benchmark-driven failure modes के case studies।
- **Length-controlled win-rate** — Dubois et al. 2024 का methodology paper।
- **Tülu 3 technical report** (AI2) — explicitly benchmarked पर useful चुनने के बारे में discuss करता है।
- **Cursor / Continue / Aider blog posts on latency** — इन्हें पढ़ना किसी भी leaderboard से ज़्यादा practical है।
- **HELM "Holistic Evaluation"** — Stanford का एक multi-dimensional eval का attempt जो harder to gamify है।
- **Chatbot Arena research papers** — और judge biases के उनके honest acknowledgments।

ये chapter book के बाकी का philosophical complement है। Chapters 04 (data), 14 (training), 19 (eval), और 25-26 (release & debug) आपको बताते हैं *कैसे* models build करें। **Chapter 27 आपको बताता है *किस लिए*।** उसके बिना, बाकी curriculum एक number optimize करती है जो matter नहीं करता।

Models बनाओ जो लोग use करें। Leaderboards fade हो जाएंगी; retained users नहीं होंगे।
