# 04 · Data — Collect, Process, और Prepare

> **TL;DR** 2026 में, एक strong small model का easiest path **FineWeb-Edu** या **DCLM-Baseline** से शुरू करना है (दोनों open, दोनों pre-cleaned), code (StarCoder2 / The-Stack-v2), math (FineMath), और synthetic instruction data (Cosmopedia / OpenHermes / Tülu 3) मिक्स करना है। फिर dedupe करो, quality के लिए filter करो, और tokenized binary files में shard करो जिन्हें `DataLoader` stream कर सके। **आपके tokens की quality आपके tokens की quantity को beat करती है।**

## 1. Data ही model क्यों है

Modern LLMs ज़्यादातर अपने data से define होते हैं। Same architecture और same compute use करने वाली दो teams wildly different models produce कर सकती हैं क्योंकि एक के पास cleaner data था। 2020 में ये कम obvious था जब architecture bottleneck था। 2026 में, **architecture small-model regime के लिए essentially solved है; data lever है।**

तीन data axes matter करते हैं:

1. **Quality** — क्या text correct, well-written, informative है?
2. **Diversity** — क्या ये कई topics, styles, languages, modalities cover करता है?
3. **Quantity** — tokens में measured, GB में नहीं। English के लिए ~4 chars/token।

Better data smaller models को fewer steps के लिए train करने देता है और फिर भी bigger models को benchmarks पर beat करते हो।

---

## 2. 2026 का Open-data Landscape

आप almost कभी web खुद crawl नहीं करते। HuggingFace से curated open datasets use करो।

### General web text

- **FineWeb** (~15T tokens) — HuggingFace का clean Common Crawl. Default base.
- **FineWeb-Edu** (~1.3T tokens) — एक educational-content classifier से filtered FineWeb. अपनी weight से *way* ऊपर punch करता है; small models जो इस पर trained हैं वो 5× more raw FineWeb पर trained models को match करते हैं।
- **DCLM-Baseline** (~4T tokens) — MIT/Apple का release; FineWeb-Edu के साथ competitive, slightly different filtering style।
- **The Pile** — पुराना लेकिन reproducibility और ablations के लिए अभी useful।

### Code

- **StarCoder2 / The-Stack-v2** — GitHub से permissively-licensed code। ~900B tokens।
- **CodeContests, OpenCodeReasoning** — reasoning traces के साथ competition-style code।

### Math और Reasoning

- **FineMath** — एक classifier द्वारा filtered math web pages। Surprisingly hard to find at scale.
- **OpenWebMath**, **InfiMM-WebMath**, **MathPile**।
- **Synthetic math** जैसे NuminaMath, OpenR1-Math, AceMath।

### Multilingual

- **HPLT v2**, **CulturaX**, **MADLAD-400** — non-English crawls।
- एक strong English model के लिए, आप अभी भी 5-15% non-English चाहते हो "reasoning patterns" diverse रखने के लिए।

### Instruction / Chat (SFT के लिए, chapter 14 देखो)

- **Tülu 3 SFT mix**, **Llama-Nemotron post-training**, **Open-Orca-2**, **OpenHermes-2.5**।
- **Cosmopedia v2** — fully synthetic, extremely clean instruction data Mixtral & Llama से generated।
- **UltraFeedback / HelpSteer3** — preference optimization के लिए।

### Multimodal (अगर relevant हो)

- **OBELICS** (interleaved image-text), **DataComp**, **LAION-5B-en-aesthetic-v2**।

---

## 3. अपना khud का data sourcing करना (जब आपको चाहिए)

अगर आप ऐसा data चाहते हो जो open releases में नहीं है (एक niche domain, एक private corpus), pipeline है:

1. **Crawl** — `commoncrawl.org` standard source है। **WARC** files use करो। `warcio` उन्हें parse करता है।
2. **Text extract करो** — `trafilatura` या `resiliparse` HTML से boilerplate cleanly strip करते हैं। **Boilerplate poison है;** menus, footers, और billions बार repeated copyright notices आपके model को tank कर देंगे।
3. **Language detect करो** — `fasttext-langdetect` fast और accurate है।
4. **Filter करो** — short docs, gibberish, repetitive boilerplate।
5. **Deduplicate करो** — documents के across और documents के अंदर।
6. **Quality-classify करो** — educational keep करो, spam drop करो।

Realistic numbers: एक clean pipeline raw Common Crawl HTML का **5-15%** final training data के रूप में keep करता है। बाकी junk है।

---

## 4. Cleaning Rules of Thumb

Bad data models को crush करता है। कुछ heuristics जो काम करती हैं:

- **Length filter**: 200 chars से कम या 100k chars से ज़्यादा docs drop करो (per page)।
- **Mean word length**: typical English के लिए 3-10 chars। बाहर gibberish है।
- **Symbol/word ratio**: > 0.1 suspicious है (prose data में tables, code dumps)।
- **Bullet/ellipsis ratio**: > 90% bullets usually navigation page है।
- **Repeated n-gram ratio**: top 5 4-grams text का > 20% cover करते हैं → spam/SEO।
- **Profanity / NSFW classifiers**: subjective; आपके goal पर depend करता है।
- **PII redaction**: emails, phone numbers, credit cards अगर आप data ship कर रहे हो।

Gopher-style filters और C4-clean rules open-source recipes हैं जिन्हें आप reuse कर सकते हो।

---

## 5. Deduplication: Cheap, Big Win

LLMs memorize करते हैं। अगर आप same sentence पर 100× train करते हो, model उसको remember करने में capacity waste करता है instead of learning। **Deduplication often एक free perplexity win देती है जो 2× more compute पर training के equivalent है।**

दो flavors:

### Exact dedup
हर document को hash करो (SHA-256 या just MD5)। Duplicates drop करो। Easy।

### Near-duplicate dedup (MinHash + LSH)
Documents जो 90% same हैं लेकिन identical नहीं (e.g., same article different header के साथ reprinted)। Standard tool **MinHash with locality-sensitive hashing** है:

1. हर doc को n-grams में tokenize करो (words के 5-grams)।
2. MinHash signature compute करो (e.g., 256 random hashes, हर एक से n-grams के ऊपर min hash store करो)।
3. Documents को overlapping signature bands द्वारा bucket करो; candidates को true Jaccard similarity > 0.8 के लिए check करो।
4. हर bucket के अंदर duplicates drop करो।

Tools: **`datasketch`** (Python), **`text-dedup`** (HuggingFace), **`spark-rapids-ml`** large scale के लिए।

कुछ TB text के लिए, near-dedup beefy CPU machine पर hours में run होता है।

### In-document dedup

कुछ docs same paragraph दो बार paste करते हैं। एक simple `set()` per doc paragraph hashes का help करता है।

---

## 6. Classifiers से Quality Filtering

Heuristics obvious junk catch करते हैं; **classifiers boring-but-low-information stuff catch करते हैं**।

2026 standard recipe:

1. एक small **gold set** (~100k docs) लो high/low quality labeled. Sources:
   - High: Wikipedia, books, arXiv abstracts, academic papers।
   - Low: random Common Crawl sample (mostly forums, ads, SEO)।
2. एक fast classifier train करो (hashed features पर logistic regression, या small fastText / DistilBERT model) "is this educational/high-quality?" predict करने के लिए।
3. हर document को score करो, top X% keep करो।

**FineWeb-Edu** exactly इसी तरह build हुआ था: Llama-3-70B-labeled "educational value" scores पर एक classifier train करो, फिर score ≥ 3 out of 5 वाले docs के लिए FineWeb को filter करो।

DCLM ने different model से similar किया। दोनों अब standard tools हैं जिन्हें आप या तो directly use कर सकते हो या replicate कर सकते हो।

```python
# fasttext के साथ rough sketch
import fasttext
clf = fasttext.train_supervised('quality_train.txt',
                                lr=1.0, epoch=5, wordNgrams=2, minCount=3,
                                dim=64, loss='hs')
clf.save_model('quality.bin')

# scale पर
def keep(doc):
    label, prob = clf.predict(doc.replace('\n', ' '))
    return label[0] == '__label__high' and prob[0] > 0.5
```

---

## 7. Decontamination

आपके model को benchmarks पर evaluate किया जाएगा (MMLU, GSM8K, HumanEval, etc.)। अगर वो exact questions आपके training data में हैं, आप अपने आप को cheat कर रहे हो। **Decontamination** चलाओ:

1. सारे benchmark prompts को hash करो (length 13 के n-grams standard है)।
2. Training data में किसी भी doc के लिए search करो जो benchmark n-gram contain करता है।
3. ऐसे docs को remove करो (या at least flag करो)।

ये standard practice है। Tools: HuggingFace का `decontaminate` script, `datatrove`।

---

## 8. Data Mixing

आपका final pretraining corpus एक **mixture** है। 2026 small model के लिए common ratios:

| Source | Share |
|--------|-------|
| Clean general web (FineWeb-Edu / DCLM) | 60-70% |
| Code (StarCoder2 filtered) | 15-25% |
| Math (FineMath, OpenWebMath) | 5-10% |
| Books / academic / Wikipedia | 5-10% |
| Multilingual (small) | 0-10% |

Higher code share non-code benchmarks पर भी reasoning में help करता है (papers जैसे "Code Pre-training Improves Reasoning" ये दिखाते हैं)।

आप **curriculum learning** भी कर सकते हो: noisier general data पहले, फिर बाद के epochs में cleaner / harder data (math, instruct-style) पर। 2026 में बहुत common — pretraining tokens के last ~10% को often "anneal phase" कहा जाता है और heavily filtered data use करता है।

---

## 9. Training के लिए Tokenizing और Sharding

Data clean हो जाने के बाद, training loop चाहता है **token IDs के fixed-size sequences, fast to read, cheaply shuffled**। Standard format:

1. हर document को आपके model के tokenizer (chapter 6) से **tokenize** करो।
2. Documents के बीच end-of-document tokens के साथ **concatenate** करो: `[BOS] doc1 tokens [EOS] doc2 tokens [EOS] ...`
3. `block_size` (e.g., 4096) के fixed-length sequences में **pack करो**। No padding, no waste।
4. **Binary shards** के रूप में save करो e.g. 1B tokens each। vocab size के depend uint16 या uint32।

Throughput के लिए fixed-size packing critical है — alternative (variable-length with padding) compute waste करता है।

```python
# minimal tokenize-and-shard sketch
import numpy as np
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
shard_size = 1_000_000_000
buf = []
shard_id = 0
def flush():
    global shard_id, buf
    arr = np.array(buf, dtype=np.uint32)
    np.save(f'shard_{shard_id:05d}.npy', arr)
    shard_id += 1
    buf = []

for doc in stream_docs():           # cleaned docs पर आपका iterator
    ids = tok.encode(doc) + [tok.eos_token_id]
    buf.extend(ids)
    if len(buf) >= shard_size:
        flush()
flush()
```

Real run के लिए, **`datatrove`** (HuggingFace) या **`mosaic-streaming`** use करो जो distributed processing, sharding, resumable runs, और S3 streaming handle करते हैं।

---

## 10. Training Time पर Streaming

आप 15T tokens RAM में load नहीं करते। आप उन्हें stream करते हो।

दो mainstream options:

- **`mosaicml-streaming`** — S3/GCS पर huge datasets के लिए designed। हर rank disjoint shards read करता है। Excellent shuffling. PyTorch `DataLoader` के साथ काम करता है।
- **`HuggingFace datasets` with `streaming=True`** — start करने के लिए easier, mid-scale तक fine।

एक minimal streaming dataset:

```python
from torch.utils.data import IterableDataset, DataLoader
import numpy as np, glob, random

class ShardedDataset(IterableDataset):
    def __init__(self, shard_glob, block_size=4096):
        self.shards = sorted(glob.glob(shard_glob))
        self.block_size = block_size

    def __iter__(self):
        random.shuffle(self.shards)
        for shard in self.shards:
            data = np.load(shard, mmap_mode='r')
            n = (len(data) - 1) // self.block_size
            order = np.random.permutation(n)
            for i in order:
                start = i * self.block_size
                x = data[start : start + self.block_size]
                y = data[start + 1 : start + 1 + self.block_size]
                yield x.astype(np.int64), y.astype(np.int64)

loader = DataLoader(ShardedDataset('shards/*.npy'),
                    batch_size=8, num_workers=4, pin_memory=True)
```

`mmap_mode='r'` आपको 1B-token shard को बिना RAM में load किए stream करने देता है। OS उसे page in करता है।

---

## 11. Data Experiment है, Fixed Input नहीं

एक बार आपका training pipeline काम करने लगे, **data ablations** ही actually improve करने का तरीका हैं:

- 5% more code add करो → क्या MMLU ऊपर गया? क्या GSM8K?
- 100B FineWeb को 100B FineWeb-Edu से replace करो → क्या सब कुछ ऊपर गया?
- Multilingual drop करो → क्या English perplexity improve हुई? (Often yes 1-2% से, लेकिन आप multilingual ability खोते हो।)
- Curated high-quality math के साथ anneal phase → क्या GSM8K spike करता है?

Ablations early set up करो। हर variant पर एक 100M-300M token "scout" model कुछ hours के लिए train करो, compare करो। 2025-2026 के कई papers *बस* data ablations हैं।

---

## 12. एक 2026 Cheat Sheet

- Crawl मत करो। **FineWeb-Edu या DCLM-Baseline** को अपने base के रूप में use करो।
- StarCoder2-filtered code, FineMath, Cosmopedia synthetic के लिए mix करो।
- अपने full mix पर **MinHash near-dedup** चलाओ।
- अगर आप पहले से curated set use नहीं कर रहे तो **classifier-based quality filter** चलाओ।
- आप जिन benchmarks पर evaluate करोगे उनके against **decontaminate** करो।
- अपने chosen tokenizer से एक बार tokenize करो; 1B-token uint32 binary files में shard करो।
- `mosaicml-streaming` या custom `IterableDataset` से stream करो।
- Extra-clean data पर training के last ~10% को **anneal phase** के रूप में use करो।
- Code की तरह data lineage track करो: कौन सा model कौन से data version पर trained। आपको ये जानना पड़ेगा।

---

## और गहराई से

- **datatrove** (HuggingFace) — production data pipeline. उनके repo में FineWeb cookbook पढ़ो।
- **FineWeb technical report** by Penedo et al. — modern data pipeline का best public writeup।
- **DCLM paper** by Li et al. — same problem पर different angle; great filtering analysis।
- **Common Crawl docs** at `commoncrawl.org/the-data` — जब आप upstream जाओ।
- **`spark_rapids_ml`** अगर आप कभी TB-scale dedup करो।

Next: **[05-model-scale.md](./05-model-scale.md)** — आपका model कितना big होना चाहिए?
