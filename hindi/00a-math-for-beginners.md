# 00a · AI Math को आसान भाषा में समझो (High Schoolers के लिए)

> **अगर `00-essential-math.md` heavy लगा, तो पहले ये पढ़ो।**
> यहाँ बस वही use करते हैं जो आप school में पढ़ चुके हो: numbers, lists, basic algebra, और थोड़ा सा calculus (जो ज़रूरी है, हम यहीं सिखा देंगे)।

---

## इस math की ज़रूरत क्यों है?

एक neural network **बस math है**: addition, multiplication, exponents, और logs — बस ये चार चीज़ें billions बार दोहराई जाती हैं। कोई **जादू नहीं है**। अगर आप math समझ गए, तो AI रहस्य नहीं रहता।

ज़्यादातर लोग math skip करते हैं, tutorial copy-paste करते हैं, और जब कुछ टूटता है तो फँस जाते हैं। आप वो मत बनो। यहाँ कुछ घंटे लगा दो। ये investment ज़िंदगी भर return देता है।

अच्छी खबर: **आपको सिर्फ़ 7 ideas चाहिए**।

1. Vectors, matrices, tensors (numbers की lists और grids)
2. Matrix multiplication
3. Broadcasting (PyTorch shape mismatch कैसे handle करता है)
4. `exp`, `log`, `softmax` functions
5. Loss (model कितना ग़लत है?)
6. Derivatives और gradients (model को कम ग़लत कैसे करें)
7. Chain rule (कई steps के derivatives को कैसे जोड़ें)

बस इतना। ये master कर लिए तो हर AI paper पढ़ लोगे।

---

## Section 1: Numbers, Lists, और Grids

### Scalar = बस एक number

जैसे `5`, `-2.7`, या `3.14`। एक value। एक thought।

AI में हम इन्हें "scalar" बोलते हैं lists से अलग पहचानने के लिए।

### Vector = numbers की एक list

**Vector** बस numbers की ordered list है, जैसे:

```
[3, 5, 2]
```

ये length 3 का vector है।

**Real-world example.** आपके 4 subjects के marks:

```
math = 92, english = 78, science = 85, art = 90
```

इसको हम vector ऐसे लिखेंगे:

```
grades = [92, 78, 85, 90]
```

Length 4. हर position एक subject को represent करती है।

**AI example.** जब model "cat" शब्द देखता है, वो उसे एक vector में बदलता है। एक छोटे model में **768 numbers** हो सकते हैं; Claude या Llama जैसे frontier model **4,096 से 16,384** numbers use करते हैं:

```
cat = [0.12, -0.45, 1.07, ..., 0.33]   # सैकड़ों से हज़ारों numbers
```

इतने सारे numbers क्यों? क्योंकि meaning capture करने के लिए ज़्यादा dimensions चाहिए ("animal", "furry", "small", "pet", "predator", और बहुत कुछ — सब इन numbers में फैला हुआ)। Model training में खुद ये numbers सीखता है; हम hand से नहीं देते।

### Vectors के साथ क्या-क्या कर सकते हैं

**Addition** — line-up करो और position-by-position add करो:

```
[1, 2, 3] + [4, 5, 6] = [1+4, 2+5, 3+6] = [5, 7, 9]
```

**किसी number से multiply** (scaling):

```
3 * [2, 4, 6] = [6, 12, 18]
```

**Length** of a vector (Pythagorean theorem!)। `[3, 4]` के लिए:

```
length = sqrt(3² + 4²) = sqrt(9 + 16) = sqrt(25) = 5
```

लंबे vectors के लिए भी same idea: हर number को square करो, जोड़ो, square root लो।

**Dot product** — position-by-position multiply करो, फिर जोड़ो:

```
[1, 2, 3] · [4, 5, 6] = (1×4) + (2×5) + (3×6) = 4 + 10 + 18 = 32
```

Dot product बताता है कि दो vectors कितने "aligned" हैं।
- बड़ा positive number = same direction में जा रहे हैं
- Zero = perpendicular (कोई relation नहीं)
- बड़ा negative number = opposite directions में

**Cosine similarity** — dot product को `(length of A × length of B)` से divide करो। Result हमेशा `-1` से `+1` के बीच होता है:

- `+1` = same direction (बहुत similar)
- `0` = perpendicular (unrelated)
- `-1` = opposite directions

Chatbots इसी से "similar texts" या "similar images" ढूँढते हैं — हर एक को vector में convert करो, फिर cosine similarity से compare करो।

### Matrix = numbers का grid

**Matrix** numbers का 2D arrangement है — rows और columns, spreadsheet जैसा।

**Example.** 3 students × 4 subjects:

```
              Math  English  Science  Art
Alice    [    92      78       85      90  ]
Bob      [    65      82       71      88  ]
Carol    [    88      90       75      80  ]
```

ये एक **3 × 4** matrix है। हम इसकी "shape" `(3, 4)` बोलते हैं। 3 rows, 4 columns।

AI में, model का सारा *learned knowledge* matrices में रहता है, जिन्हें **weights** बोलते हैं। जब लोग कहते हैं "Llama 3 के 405 billion parameters हैं", तो उनका मतलब है कि सारी matrices में मिलाकर 405 billion numbers हैं।

### Tensor: बस stacks (इस word से डरो मत)

**Tensor** एक umbrella word है। बस इतना मतलब है: "numbers का multi-dimensional array"।

| Shape | क्या है | Example |
|-------|---------|---------|
| `()` | scalar | `5` |
| `(5,)` | vector | `[1, 2, 3, 4, 5]` |
| `(3, 4)` | matrix | 3×4 का grid |
| `(2, 3, 4)` | 3-D tensor | 2 stacks of 3×4 grids |
| `(32, 3, 224, 224)` | 4-D tensor | 32 photos, 3 colors, 224×224 pixels |

Color photo एक 3-D tensor है। PyTorch में convention **channels first** है: `3 colors × height × width`, तो एक 224×224 image की shape `(3, 224, 224)` होती है। ऐसी 32 photos का batch `(32, 3, 224, 224)` होगा।

तो जब कोई कहे "tensor of shape (8, 16, 768)", उसका मतलब बस इतना है: numbers का 3D box जिसकी ये dimensions हैं। बस।

### Code में (PyTorch)

```python
import torch

# scalar
s = torch.tensor(5.0)                     # shape ()

# vector
v = torch.tensor([1.0, 2.0, 3.0])         # shape (3,)

# matrix
M = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])            # shape (2, 2)

# 3D tensor (random numbers)
T = torch.randn(2, 3, 4)                  # shape (2, 3, 4)

print(T.shape)
```

जब भी dimensions confuse करें, **shape print कर दो**। ये नंबर 1 debugging trick है।

---

## Section 2: Matrix Multiplication (AI का सबसे important operation)

LLM जब चलता है, उसका लगभग **95% time** matrix multiplication में जाता है। तो ये समझना ज़रूरी है।

### Shape rule

अगर matrix `A` की shape `(m, k)` है और matrix `B` की shape `(k, n)` है, तो `A × B` (Python में हम `A @ B` लिखते हैं) की shape `(m, n)` होगी।

**अंदर के numbers match होने चाहिए**:

```
(2, 3) @ (3, 4) → (2, 4)   ✓  (3 = 3, drop them, बाहरी 2 और 4 रखो)
(2, 3) @ (4, 5) → ERROR    ✗  (3 ≠ 4)
```

### हर entry कैसे compute होती है

Result की हर entry एक **dot product** है — A की एक row और B की एक column का:

```
C[i, j] = (A की i-th row)  ·  (B का j-th column)
```

बस यही पूरा story है। Matrix multiplication = dot products का grid।

### Worked example (एक बार hand से ज़रूर करो)

```
A = [1  2  3]       shape (2, 3)
    [4  5  6]

B = [1  0]
    [0  1]          shape (3, 2)
    [1  1]
```

Result shape: `(2, 2)`। चलो हर entry calculate करते हैं।

```
C[0,0] = A की row 0 · B का col 0 = 1·1 + 2·0 + 3·1 = 1 + 0 + 3 = 4
C[0,1] = A की row 0 · B का col 1 = 1·0 + 2·1 + 3·1 = 0 + 2 + 3 = 5
C[1,0] = A की row 1 · B का col 0 = 4·1 + 5·0 + 6·1 = 4 + 0 + 6 = 10
C[1,1] = A की row 1 · B का col 1 = 4·0 + 5·1 + 6·1 = 0 + 5 + 6 = 11
```

तो:

```
C = [4    5]
    [10  11]
```

Code में:

```python
A = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])
B = torch.tensor([[1., 0.],
                  [0., 1.],
                  [1., 1.]])
print(A @ B)
# tensor([[ 4.,  5.],
#         [10., 11.]])
```

### ये क्यों matter करता है?

एक neural network "layer" **literally** बस ये है:

```
output = input @ W + b
```

- `input` एक vector (या batch of vectors) है
- `W` एक learned weight matrix है
- `b` एक learned bias vector है
- `output` layer का response है

ऐसी कई layers stack करो → deep neural network। एक specific arrangement में stack करो → transformer (GPT, Claude, Gemini, etc. के पीछे यही architecture है)।

जब हम कहते हैं "model train कर रहे हैं", उसका मतलब है: **`W` और `b` के numbers adjust करो जब तक outputs जो हम चाहते हैं वो आ जाएँ।**

---

## Section 3: Broadcasting (जब shapes exactly match नहीं करतीं)

कभी-कभी आप एक vector को matrix की हर row में add करना चाहते हो। PyTorch ये automatic handle करता है — "broadcasting" करके। मतलब, छोटी shape को pretend करके repeat कर देता है।

**Example:**

```python
A = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])      # shape (2, 3)
b = torch.tensor([10, 20, 30])     # shape (3,)
print(A + b)
# [[11, 22, 33],
#  [14, 25, 36]]
```

PyTorch ने `[10, 20, 30]` को `A` की हर row में add कर दिया।

**Broadcasting के rules:**

1. Shapes को **right** से line up करो।
2. हर pair of dimensions के लिए, sizes match होनी चाहिए या एक `1` होनी चाहिए।

ये clean code के लिए बढ़िया है, लेकिन **silent bugs** भी कर सकता है — PyTorch कोई error नहीं देगा, लेकिन result ग़लत होगा:

```python
loss = torch.randn(4)       # shape (4,)   — हर sample का एक loss
weight = torch.randn(4, 1)  # shape (4, 1) — ग़लती से extra dim आ गई
result = loss * weight       # shape (4, 4) — (4,) नहीं! Silent outer product.
```

PyTorch ने `(4,)` को `(1, 4)` मानकर `(4, 1)` के साथ broadcast किया → `(4, 4)` बना दिया। कोई error नहीं, बस ग़लत answer।

**आदत बनाओ: नए हो तो shapes लगातार print करते रहो।**

---

## Section 4: `exp`, `log`, और `softmax`

### `exp(x) = eˣ`

`e` एक special number है ≈ `2.71828`। तो:

- `exp(0) = 1`
- `exp(1) ≈ 2.718`
- `exp(2) ≈ 7.39`
- `exp(-2) ≈ 0.135`

**Properties:** हमेशा positive, बहुत तेज़ बढ़ता है, कभी zero नहीं होता।

### `log(x)`

`log` `exp` का **inverse** है। तो अगर `exp(2) = 7.39`, तो `log(7.39) = 2`।

- `log(1) = 0`
- `log(e) = 1`
- `log(10) ≈ 2.30`

सिर्फ़ `x > 0` के लिए defined है (zero या negative का log नहीं ले सकते)।

**Magic property:** `log` multiplication को addition में बदल देता है।

```
log(a × b) = log(a) + log(b)
```

AI में हमें ये क्यों पसंद है: probabilities बहुत छोटे numbers होते हैं, और कई छोटे numbers multiply करने पर इतने छोटे हो जाते हैं कि computer उन्हें zero समझ ले ("underflow" बोलते हैं इसे)। अगर हम log-space में काम करें और **add** करें multiply की जगह, तो underflow नहीं होता।

### Softmax: raw scores को probabilities में बदलना

Neural network की last layer raw scores देती है — इनको **logits** बोलते हैं। Logits कोई भी real number हो सकते हैं — positive, negative, बड़े, छोटे।

लेकिन हमें usually **probabilities** चाहिए: 0 और 1 के बीच के numbers जो जोड़ने पर 1 होते हैं।

Softmax यही करता है:

```
softmax(x)[i] = exp(x[i]) / sum_of_exp_of_all_x
```

**Worked example.** मान लो model ने तीन options के लिए logits दिए: `[2, 1, 0.1]`।

Step 1: हर एक का `exp` लो।
- `exp(2)  = 7.39`
- `exp(1)  = 2.72`
- `exp(0.1) = 1.11`

Step 2: जोड़ो।
- `total = 7.39 + 2.72 + 1.11 = 11.22`

Step 3: हर एक को total से divide करो।
- `7.39 / 11.22 ≈ 0.659`
- `2.72 / 11.22 ≈ 0.243`
- `1.11 / 11.22 ≈ 0.099`

Result: `[0.659, 0.243, 0.099]`। जोड़ने पर 1 आता है। इसको ऐसे पढ़ सकते हैं: **"model 65.9% sure है option A का, 24.3% sure है option B का, 9.9% sure है option C का।"**

```python
def softmax(x):
    e = torch.exp(x - x.max())   # numerical safety के लिए max subtract
    return e / e.sum()

x = torch.tensor([2.0, 1.0, 0.1])
print(softmax(x))   # tensor([0.6590, 0.2424, 0.0986])
```

**Max subtract क्यों करते हैं?** क्योंकि `exp(1000)` एक astronomical number है जो आपके computer में overflow कर देगा। पहले max subtract करो → सब numbers 0 या negative हो जाते हैं, और `exp(negative)` छोटा और safe होता है। Result mathematically same रहता है:

```
exp(x) / Σ exp(x)  =  exp(x - m) / Σ exp(x - m)
```

(दोनों `exp(m)` factors cancel हो जाते हैं।) हर framework यही trick use करता है।

### Sigmoid (softmax का special case)

जब बस दो choices हों, हम **sigmoid** use करते हैं:

```
sigmoid(x) = 1 / (1 + exp(-x))
```

ये किसी भी real number को `(0, 1)` range में squish कर देता है:

- `sigmoid(0) = 0.5`
- `sigmoid(2) ≈ 0.88`
- `sigmoid(-2) ≈ 0.12`
- `sigmoid(10) ≈ 1.0`
- `sigmoid(-10) ≈ 0.0`

S-shaped curve दिखती है। Use होती है binary "yes/no" predictions में, और modern architectures के अंदर "gates" में जो decide करते हैं कि कितनी information pass हो।

---

## Section 5: Loss — model कितना ग़लत है?

हमें एक number चाहिए जो बताए "ये prediction कितनी ख़राब है?" कम = बेहतर। इस number को **loss** बोलते हैं।

Language models (और लगभग सब classification) के लिए, हम **cross-entropy loss** use करते हैं।

### Simple version

अगर model ने सही answer को probability `p` दी, तो:

```
loss = -log(p)
```

बस। एक formula।

Examples:

| Model ने सही answer को कितनी probability दी | Loss |
|---|---|
| `1.00` (perfect) | `-log(1.00) = 0`        — कोई penalty नहीं |
| `0.90` | `-log(0.90) ≈ 0.105`              — छोटी penalty |
| `0.50` (50/50 guess) | `-log(0.50) ≈ 0.693` — moderate penalty |
| `0.10` | `-log(0.10) ≈ 2.30`               — बड़ी penalty |
| `0.01` | `-log(0.01) ≈ 4.61`               — बहुत बड़ी penalty |
| `0.0001` | `-log(0.0001) ≈ 9.21`           — massive penalty |

तो loss **छोटा होता है जब model confident है और सही है**, और **बहुत बड़ा होता है जब confident है और ग़लत है**। बिल्कुल वही जो हमें चाहिए।

### Code में

```python
import torch.nn.functional as F

logits = torch.tensor([[2.0, 1.0, 0.1]])    # shape (1, 3) — 3 classes
target = torch.tensor([0])                   # सही class index 0 है

loss = F.cross_entropy(logits, target)
print(loss.item())   # ≈ 0.417
```

`cross_entropy` raw **logits** लेता है (probabilities नहीं) और अंदर ही softmax + log apply करता है। ऐसे करना faster है और numerically ज़्यादा stable भी।

ये LLM pretraining का **core loss function** है। Fine-tuning stages (RLHF, DPO) इसके ऊपर और objectives add करते हैं, लेकिन cross-entropy foundation है।

---

## Section 6: Derivatives — change का rate

एक **derivative** इस सवाल का answer है:
> *"अगर मैं `x` को थोड़ा सा हिलाऊँ, तो `f(x)` कितना बदलेगा?"*

Geometrically: derivative उस point पर curve का **slope** है।

### 5 rules जो आपको चाहिए

| Function | Derivative | आसान भाषा में |
|----------|-----------|----------|
| `f(x) = c` (कोई constant) | `0` | constants बदलते नहीं |
| `f(x) = x` | `1` | x 1-for-1 बदलता है |
| `f(x) = xⁿ` | `n × xⁿ⁻¹` | "power rule" |
| `f(x) = eˣ` | `eˣ` | exp का derivative खुद है! |
| `f(x) = log(x)` | `1/x` | |

Plus 3 rules functions को combine करने के लिए:

- **Sum rule:** `(f + g)` का derivative `f' + g'` है
- **Product rule:** `(f × g)` का derivative `f' × g + f × g'` है
- **Chain rule:** `f(g(x))` का derivative `f'(g(x)) × g'(x)` है ← **deep learning का सबसे important rule**

### Quick worked example

```
f(x) = 3x² + 5x + 7
```

Rules apply करो:
- `3x²` का derivative = `3 × 2x = 6x` (power rule × constant multiple)
- `5x` का derivative = `5`
- `7` का derivative = `0` (constant)

जोड़ो: `f'(x) = 6x + 5`।

अगर `x = 2`, तो `f'(2) = 12 + 5 = 17`। तो `x = 2` पर curve का slope 17 है। अगर हम `x` को `0.001` हिलाएँ, तो `f(x)` लगभग `17 × 0.001 = 0.017` बढ़ जाएगा।

### Chain rule (backpropagation की आत्मा)

मान लो आपके पास nested function है:

```
f(x) = sin(x²)
```

**बाहरी** function `sin` है, और **अंदरूनी** function `x²` है।

Chain rule:

```
f'(x) = (बाहरी का derivative)  ×  (अंदरूनी का derivative)
      = cos(x²)                ×  2x
```

बाहर से अंदर तक differentiate करो, multiply करते जाओ।

एक neural network functions का deep nesting है:

```
loss = cross_entropy(softmax(linear(linear(linear(input)))))
```

किसी भी weight को थोड़ा हिलाने पर loss कितना बदलेगा — ये निकालने के लिए हम chain rule layer-by-layer apply करते हैं, loss से शुरू करके input तक। **Backpropagation बस यही है** — एक huge graph पर chain rule apply करना।

PyTorch ये automatic करता है। आप hand से कभी नहीं करते।

---

## Section 7: Gradients — कई inputs के लिए derivatives

AI में ज़्यादातर functions **बहुत सारे** inputs लेते हैं। एक neural network 768 numbers in ले सकती है।

**Gradient** बस partial derivatives की list है — हर input के लिए एक।

**Example:**

```
f(x₁, x₂, x₃) = x₁² + 3x₂ + x₃
```

हर variable के respect में partial derivative निकालो (बाक़ियों को constant मानो):

- `∂f/∂x₁ = 2x₁`
- `∂f/∂x₂ = 3`
- `∂f/∂x₃ = 1`

तो gradient है `[2x₁, 3, 1]`। बस derivatives का vector है।

### Gradient uphill point करता है

Geometrically, point `x` पर gradient उस direction में point करता है जहाँ `f` सबसे **तेज़** बढ़ता है।

Imagine करो आप एक hilly landscape (loss surface) पर खड़े हो। Gradient एक compass है जो हमेशा uphill point करता है। अगर हमें loss **कम** करना है, तो हम **opposite** direction में चलते हैं — downhill।

### Gradient descent — AI कैसे सीखता है

Update rule:

```
new_x = old_x  -  learning_rate × gradient
```

**Learning rate** (`η` या `lr` लिखते हैं) एक छोटा number होता है, जैसे `0.001`। बहुत सारे छोटे steps downhill लो, और आप एक low point तक पहुँच जाते हो। यही training है।

### PyTorch आपके लिए ये करता है

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()       # y = 1 + 4 + 9 = 14
y.backward()             # PyTorch gradients भर देता है
print(x.grad)            # tensor([2., 4., 6.])
```

`requires_grad=True` बोलता है "इस tensor के gradients track करो"। फिर `.backward()` उन्हें compute करता है, और आप `x.grad` से पढ़ सकते हो।

क्या हुआ अभी? `y = x₁² + x₂² + x₃²`, तो gradient है `[2x₁, 2x₂, 2x₃] = [2, 4, 6]` जब `x = [1, 2, 3]`। Math match हो गया।

---

## Section 8: Probability की basics

### Probability क्या है?

0 और 1 के बीच का number।
- `0` = impossible
- `1` = certain
- `0.5` = 50/50

एक **distribution** अलग-अलग outcomes की probabilities की list है; ये जोड़कर 1 बनती है।

Examples:
- Fair coin: `[P(heads)=0.5, P(tails)=0.5]`
- Loaded die जो 6 पर ज़्यादा गिरता है: `[0.1, 0.1, 0.1, 0.1, 0.1, 0.5]`

### Expectation = average outcome

अगर random variable `X` values `v_i` लेता है probabilities `p_i` के साथ, तो expected value:

```
E[X] = Σ p_i × v_i
```

Fair die के लिए: `E[X] = (1/6)(1+2+3+4+5+6) = 21/6 = 3.5`। Average में 3.5 आता है।

AI में, हम जो loss minimize करते हैं वो सारे training data पर **expected** loss है। हम इसे एक छोटे batch (मान लो 32 examples) पर average करके approximate करते हैं। इसीलिए इसे **stochastic** gradient descent बोलते हैं — हर batch एक random sample है।

### Distribution से sample करना

Probabilities मिल गईं → हम pick (sample) कर सकते हैं। `[0.5, 0.3, 0.2]` के लिए:
- 50% time option 0 pick करते हैं
- 30% time option 1 pick करते हैं
- 20% time option 2 pick करते हैं

Language models में, softmax के बाद हमें next word पर distribution मिलती है। Pick करने की strategies:

- **Argmax (greedy)** — हमेशा सबसे ज़्यादा probability pick करो। Boring, repetitive।
- **Sample** — probabilities के according random pick करो। ज़्यादा creative।
- **Top-k** — सिर्फ़ `k` सबसे likely options consider करो, उनसे sample।
- **Top-p (nucleus)** — सबसे छोटा set रखो जिसकी probabilities sum `p` बने, उससे sample।
- **Min-p** — सिर्फ़ वो tokens रखो जिनकी probability कम-से-कम `p × top_token_probability` हो। Model की confidence के हिसाब से automatically adjust होता है — 2025+ inference engines में standard।
- **Temperature** — softmax से पहले सब logits को `T` से divide करो।
  - `T < 1` sharp करता है (ज़्यादा deterministic)
  - `T > 1` flat करता है (ज़्यादा random)

```python
def sample(logits, temperature=1.0):
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1)
```

### KL divergence (बस word देख लो)

KL divergence measure करता है कि **दो distributions कितनी अलग हैं**:

```
KL(P || Q) = Σ P(x) × log(P(x) / Q(x))
```

`P == Q` होने पर `0`, और जैसे-जैसे अलग होते हैं, बढ़ता है।

**Cool fact:** cross-entropy loss minimize करना *बिल्कुल वही है* जो model की distribution और data की distribution के बीच KL divergence minimize करना (एक constant के अलावा)। तो **training का मतलब है model को data की distribution match करवाना।**

---

## Section 9: Real LLM code में जो shapes दिखेंगी

Transformer code में ये letters **हर जगह** आते हैं:

| Letter | मतलब | Typical size |
|--------|-------|--------------|
| `B` | batch (एक साथ कितने examples) | 1 से 1024 |
| `T` | time = sequence length (tokens की संख्या) | 1 से 1M+ |
| `D` (या `d_model`) | model dimension | 768 से 16k |
| `H` (या `n_heads`) | attention heads की संख्या | 8 से 128 |
| `d` (या `head_dim`) | per-head dimension | 64 से 128 |
| `V` | vocabulary size | 32k से 256k |
| `F` (या `d_ff`) | feed-forward inner dim | 2-8 × D |

Note: `D = H × d`। Model dimension heads में बँट जाती है।

Typical attention input की shape `(B, T, D)` होती है। Heads में split करने के बाद ये `(B, H, T, d)` बन जाती है। ज़्यादातर "shape gymnastics" बस इन forms के बीच reshape है।

```python
B, T, D, H = 2, 5, 128, 8
d = D // H

x = torch.randn(B, T, D)                       # (2, 5, 128)
x_heads = x.view(B, T, H, d).transpose(1, 2)   # (2, 8, 5, 16)
```

Memorize मत करो — shapes print करते रहो जब तक click न हो जाए।

---

## Section 10: Numerical tricks (training को blow up होने से बचाने के लिए)

ये छोटी details training में **NaN** (not-a-number) errors से बचाती हैं। इनके बिना training silently टूट जाती है।

1. **Softmax से पहले max subtract करो।** Same answer, no overflow.
2. **Log-sum-exp use करो** exp को directly sum करने की जगह:
   ```
   log(Σ exp(x))  =  m + log(Σ exp(x − m))
   ```
   जहाँ `m = max(x)`। Numerically stable.
3. **Mixed precision.** Activations को `bf16` (16-bit floats) में store करो, but sums को `fp32` (32-bit) में accumulate करो। PyTorch ये `autocast` से handle करता है।
4. **Gradients clip करो।** Gradient कितना बड़ा हो सकता है, उसकी limit लगाओ, ताकि एक bad batch training को blow up न करे:
   ```python
   torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
   ```
5. **`log(0)` से बचो।** एक tiny epsilon add करो: `log(p + 1e-10)`, या better, सब जगह log-space में काम करो।

जब तक कुछ टूटे नहीं, इसे ignore कर सकते हो। जब टूटे, ये list पहली जगह है जहाँ देखोगे।

---

## Section 11: One-page Cheat Sheet

- **Vector** = numbers की list।
- **Matrix** = numbers का 2D grid।
- **Tensor** = numbers का multi-D array।
- सबसे common operation है `output = input @ W + b`। LLM का ज़्यादातर हिस्सा यही है, बार-बार।
- **Softmax** raw scores (logits) को probabilities में बदलता है जो sum 1 करती हैं।
- **Cross-entropy** = `-log(p_correct)` = loss। सही पर छोटा, confident-ग़लत पर बड़ा।
- **Derivative** slope है: input हिलाने पर output कितना बदला?
- **Gradient** हर input dimension के लिए एक derivative। Uphill point करता है — हम downhill जाते हैं।
- **Chain rule** = nested functions को differentiate करना। Backprop बस chain rule है पूरी network पर।
- **KL divergence** और **cross-entropy** लगभग twins हैं। Training model की distribution को data की match करवाती है।

बस यही deep learning का पूरा math foundation है। बाक़ी सब इसके ऊपर का plumbing है।

---

## Section 12: ये सब Google Colab पर चलाओ (free, बिना install)

आपको Python या PyTorch अपने computer पर install करने की ज़रूरत नहीं। **Google Colab** आपको cloud में एक free Python notebook देता है, PyTorch pre-installed के साथ, और एक free GPU भी।

### Step-by-step setup

1. Browser खोलो और जाओ **https://colab.research.google.com**।
2. किसी भी Google account से sign in करो।
3. **"New notebook"** click करो। एक empty cell blinking cursor के साथ दिखेगी।
4. Cell में code type या paste करो। For example:
   ```python
   import torch
   v = torch.tensor([1.0, 2.0, 3.0])
   print(v)
   print(v.shape)
   ```
5. Cell run करो — उसके left पर **▶** button click करो, या `Shift + Enter` press करो।
6. ऊपर **+ Code** button से एक new cell add करो। Matrix multiplication try करो:
   ```python
   A = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
   B = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
   print(A @ B)
   ```

### Optional: free GPU on करो

इस guide के छोटे examples के लिए GPU की ज़रूरत नहीं, लेकिन कुछ बड़ा train करना हो तो:

1. Menu: **Runtime → Change runtime type**।
2. Hardware accelerator: **T4 GPU** select करो (free)।
3. **Save** click करो।

Verify करो कि काम कर रहा है:

```python
import torch
print(torch.cuda.is_available())          # True print होना चाहिए
print(torch.cuda.get_device_name(0))      # जैसे 'Tesla T4'
```

### ये exercise try करो (इस guide का हर concept use होता है)

ये पूरा block एक Colab cell में paste करो और run करो। हर line पढ़ो — हर concept जो हमने cover किया।

```python
import torch

# एक simple "neural network" 1 layer के साथ:
# input 3 numbers, output 2 numbers।
W = torch.randn(3, 2, requires_grad=True)   # weight matrix
b = torch.randn(2, requires_grad=True)      # bias vector
x = torch.tensor([1.0, 2.0, 3.0])           # input vector

# Forward pass: y = x @ W + b   (Section 2)
y = x @ W + b

# Loss compute करो: y target से कितना दूर है?
target = torch.tensor([1.0, 0.0])
loss = ((y - target) ** 2).sum()            # squared error

# Backward pass: PyTorch gradients compute करता है (Section 7)
loss.backward()

print("y     =", y)
print("loss  =", loss.item())
print("dL/dW =", W.grad)        # W के respect में gradient
print("dL/db =", b.grad)        # b के respect में gradient
```

क्या हो रहा है:
- एक 3-input, 2-output "layer" बनाया।
- एक forward pass करके `y` निकाला।
- `y` को target से compare करके loss निकाला।
- `loss.backward()` ने chain rule run करके `W` और `b` पर `.grad` भर दिया।

Congrats — अब आपने इस guide का हर concept literally 10 lines code में execute कर लिया।

### Colab use करने के tips

- Free GPU की limits हैं (कुछ घंटे per day)। Learning के लिए बहुत है।
- Notebooks auto-save होते हैं Google Drive में।
- कुछ weird हो जाए तो सब reset करो: **Runtime → Restart runtime**।
- Code cells के बीच **+ Text** button से notes add करो। Code के साथ-साथ notes लेने में बढ़िया।

---

## आगे क्या पढ़ें

अब जब ये click हो गया, तो **[00-essential-math.md](./00-essential-math.md)** का compact version fast पढ़ा जाएगा — same content, tighter form में।

फिर **[01-neural-networks.md](./01-neural-networks.md)** पर move करो actually एक neural network बनाने के लिए।

अगर कोई future chapter कभी confuse करे, **यहाँ वापस आ जाओ**। Math ground floor है। दो बार इसे skip मत करना।
