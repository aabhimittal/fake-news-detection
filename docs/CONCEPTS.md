# Concepts — the theory behind the code

This document explains the ideas the project implements, from first principles.
It complements the code comments and the step-by-step tour in the main README.

---

## 1. Fake-news detection as text classification

### 1.1 The learning problem

We are given a corpus of documents `x₁ … xₙ` with binary labels `yᵢ ∈ {0, 1}`
(0 = credible, 1 = fake). We want a function `f(x) → {0, 1}` that generalises to
unseen documents. This is **supervised binary classification**.

Text can't be fed to a numerical model directly, so the real work is turning a
string into a vector — *feature extraction* — and then fitting a decision
boundary in that vector space.

### 1.2 TF-IDF: representing "what is said"

**Bag-of-words** ignores word order and represents a document by which words it
contains. Raw counts over-weight common words, so we use **TF-IDF**:

```
tfidf(t, d) = tf(t, d) · idf(t)
idf(t)      = log( N / (1 + df(t)) )
```

- `tf(t, d)` — how often term `t` appears in document `d` (we use the
  `sublinear` variant `1 + log(tf)` so a word appearing 100× isn't 100× as
  important).
- `idf(t)` — the inverse document frequency: rare terms (high `idf`) are more
  discriminative than words that appear everywhere (low `idf`).

We include **bigrams** (`ngram_range=(1,2)`) so multi-word tells like
`"share before"` or `"one weird trick"` become single features.

**Why it works for fake news:** fabricated stories reuse a recognisable
vocabulary of sensational and clickbait phrasing that TF-IDF surfaces as
high-weight features.

### 1.3 Stylometry: representing "how it is said"

Vocabulary isn't the whole story — two articles on the same topic can differ in
*credibility signalled by style*. We add nine scale-free statistics:

| Feature | Signal |
|---------|--------|
| `uppercase_ratio` | SHOUTING is a classic clickbait tell |
| `exclamation_ratio`, `question_ratio` | Emotional / rhetorical punctuation |
| `clickbait_ratio` | Hits against a curated sensationalism lexicon |
| `unique_word_ratio` | Lexical diversity (repetitive filler vs. substance) |
| `avg_word_len`, `digit_ratio`, `word_count`, `char_count` | General register |

These are **interpretable and robust**: they transfer across topics far better
than any single vocabulary feature, because they measure *manner*, not *subject*.

### 1.4 Why a linear classifier

TF-IDF vectors live in a very high-dimensional, sparse space where classes are
close to linearly separable. A **linear model** — logistic regression by
default — is therefore both accurate and fast, and its per-feature weights give
us free explanations:

```
score(d) = Σ  wⱼ · featureⱼ(d)   →   P(fake) = σ(score)
```

The contribution of feature `j` to a specific verdict is exactly
`wⱼ · featureⱼ(d)`, which is what `FakeNewsDetector.explain` reports.

Alternatives the project also ships:

- **Passive-Aggressive** — an online learner well suited to streaming news.
- **Linear SVM** — maximum-margin boundary, strong on sparse text.
- **Multinomial Naïve Bayes** — a fast probabilistic baseline.

### 1.5 Measuring success honestly

For a moderation system the costs are asymmetric:

- A **false negative** (fake labelled real) lets misinformation spread → we
  watch **recall on the fake class**.
- A **false positive** (real labelled fake) censors legitimate news and erodes
  trust → we watch **precision**.

`F1` balances the two, and the **confusion matrix** shows exactly where errors
fall. Accuracy alone can hide a model that never catches the fake class on an
imbalanced corpus.

### 1.6 Cross-validation: comparing models fairly

A single train/test split yields one number that depends on *which* rows landed
in the test set. **Stratified k-fold cross-validation** removes that luck: the
data is partitioned into `k` folds (each preserving the class balance), and each
model is trained on `k-1` folds and evaluated on the held-out one, rotating
through all `k`. Reporting **mean ± standard deviation** across folds captures
both how good a model is and how *stable* that estimate is — a model with high
mean but huge variance is not something to deploy. Because every model is scored
on the *same* folds, the comparison is apples-to-apples. This is what
`fakenews.benchmark` does.

### 1.7 Beyond bag-of-words: fine-tuned transformers

TF-IDF discards word order and context: "not fake at all" and "fake, not at all"
share a bag of words. **Transformers** (BERT, DistilBERT) fix this with
**self-attention** — each token's representation is a learned, weighted blend of
every other token in the sentence, so a word is encoded *in context*.

The decisive practical idea is **transfer learning via fine-tuning**:

1. **Pretraining** (done for us, once, at great expense) — the model learns
   general language structure from billions of words of unlabelled text via
   self-supervised objectives (masked-word prediction).
2. **Fine-tuning** (what we do) — we attach a small classification head and
   continue training for a *couple of epochs* on our few hundred labelled
   fake/real examples. Because the model already "knows English", it needs very
   little task-specific data to specialise.

This buys accuracy on subtle, paraphrased misinformation that dodges obvious
clickbait vocabulary. The costs are real, though: orders of magnitude more
compute, and the loss of the linear model's free per-feature explanations (a
transformer's decision is distributed across millions of weights). The
interpretable linear model and the contextual transformer are complementary, not
rivals — `fakenews.transformer.TransformerDetector` deliberately shares the
`FakeNewsDetector` interface so you can swap between them.

---

## 2. Propagation: from a label to a stopped cascade

### 2.1 Social networks are scale-free

Empirically, follower counts follow a power law: a few accounts are massive
hubs, most are small. The **Barabási–Albert** model reproduces this via
*preferential attachment* — new users are more likely to connect to already
popular users. Scale-free structure is why a single hub reshare can ignite a
cascade, and why removing a few hubs can smother one.

### 2.2 The Independent Cascade diffusion model

Misinformation spreads like a contagion. Each user is in one of three states
(an **SIR**-style model):

- **S**usceptible — hasn't seen the story.
- **I**nfected — actively sharing it.
- **R**ecovered — saw it and stopped, or was immunised.

Per time step:

1. **Infect** — each infected user `u` independently infects each susceptible
   neighbour `v` with probability `p` (the activation / virality probability).
2. **Recover** — each infected user stops sharing with probability `r`.

Because the process is stochastic, we run it many times (**Monte-Carlo**) and
average. Two summary numbers matter: **total reached** (how many users ever
shared) and **peak active** (the worst simultaneous load — the "how loud did it
get" metric).

### 2.3 Containment = targeted immunisation

We can protect a small budget `k` of nodes: a **monitor / fact-checker** node,
once it would be infected, instead debunks and refuses to propagate. Formally it
becomes recovered and blocks every path through it. The optimisation problem —
*which `k` nodes minimise expected spread* — is **NP-hard** (it is the dual of
influence maximisation), so we use principled heuristics:

| Strategy | Selection rule | Cost |
|----------|----------------|------|
| **Degree** | The `k` highest-degree hubs | Needs the full degree sequence |
| **Betweenness** | The `k` nodes on the most shortest paths (best bridges) | Expensive centrality computation |
| **Acquaintance** | Repeatedly: sample a random node, immunise a random neighbour | **Local only** |
| **Random** | `k` uniformly random nodes | Trivial (null baseline) |

### 2.4 The friendship paradox (why acquaintance immunisation works)

"Your friends have more friends than you do." Formally, sampling a node by
picking a **random neighbour** of a random node biases toward high-degree nodes,
because hubs are named as a neighbour by many people. So acquaintance
immunisation concentrates the budget on influential accounts **without ever
computing degrees or seeing the whole graph** — the realistic setting for a
platform that can only observe local interactions. That's why, in our results,
it lands between random and full-knowledge degree targeting.

### 2.5 Greedy immunisation, and a submodularity trap

The centrality strategies are heuristics — cheap stand-ins for the real
objective, *minimise expected spread*. The **greedy** strategy optimises that
objective directly: repeatedly immunise whichever node reduces the simulated
cascade the most (its **marginal gain**), estimated by Monte-Carlo.

The interesting twist is about **submodularity**. A set function `f` is
submodular if marginal gains shrink as the set grows ("diminishing returns"):

```
gain(v | S)  ≥  gain(v | T)     whenever  S ⊆ T
```

- **Influence maximisation** — choosing *seeds* to maximise spread — is monotone
  and submodular (Kempe–Kleinberg–Tardos). Submodularity guarantees the greedy
  solution is within `(1 − 1/e) ≈ 63%` of optimal, **and** it licenses the
  **CELF** speed-up: a stale marginal gain is a valid *upper bound*, so you can
  keep candidates in a max-heap and re-evaluate only the current top one — often
  a handful of simulations per round instead of hundreds.

- **Node immunisation** — choosing *blockers* to minimise spread — is **not**
  submodular in general. Removing one node can *increase* another node's
  marginal value, because a previously-redundant node may now sit on the only
  remaining path. When that happens, CELF's upper-bound assumption is violated
  and its lazy skipping locks in a worse set.

We hit exactly this: the lazy CELF variant selected a strictly worse monitor set
than the exact greedy, so `fakenews.propagation` runs the **exact** greedy
(re-evaluating every candidate each round), trading CELF's speed for correctness
and capping cost with a high-degree candidate pool instead. The lesson is
general: *the fast algorithm is only correct when its structural assumption
holds — check submodularity before you reach for CELF.*

### 2.6 Reading the result

```
strategy      reached   reduction vs none
none            45.4          0.0%
degree          29.9         34.3%   ← hub targeting: best heuristic, needs global info
betweenness     30.0         33.9%   ← bridge targeting: comparable
greedy          29.9         34.3%   ← optimises spread directly; matches degree here
acquaintance    39.2         13.6%   ← local-only, still 3× better than random
random          43.1          5.0%   ← spreading budget thinly barely helps
```

Greedy tying `degree` is itself informative: on scale-free graphs the cascade
must funnel through hubs, so hub-immunisation is already near-optimal and greedy
confirms it — while making *no* structural assumption, which is what keeps it
robust when a network is *not* cleanly hub-dominated.

**Takeaway:** structure beats volume. A handful of well-chosen fact-checkers on
the right accounts contains a cascade far better than many placed at random —
the single most important practical lesson for platform-scale moderation.

---

## 3. Production hardening

### 3.1 Adversarial evasion is a normalisation problem

A bag-of-words model has a specific, exploitable weakness: it matches *tokens*.
Anything that changes the codepoints while preserving human legibility — a
Cyrillic `а`, a zero-width space, `v4cc1n3`, `shoooocking` — turns a known
feature into an unknown one, and the model silently loses the evidence it was
relying on.

The cheapest robust answer is **normalisation, not detection**. Rather than try
to judge whether obfuscation was malicious (an arms race), we map every variant
onto one canonical form so the model sees the same token either way. This is
deterministic, costs one pass over the string, and cannot be defeated by tuning
the obfuscation rate.

The engineering difficulty is entirely in the **false-positive direction**: an
over-eager normaliser corrupts legitimate text. Three concrete rules earn their
place in `fakenews.adversarial`:

* Collapse repeated **letters** only, so `!!!` — a real stylometric signal —
  survives. Runs of 3+ collapse to *one* letter, because genuine doubles
  ("coffee") are runs of 2 and are never touched. Measured on the sample corpus
  this recovers the original token in ~52% of attacked documents with **zero**
  corruption of clean text; collapsing to two recovers none.
* De-leet a digit only when it is not part of a multi-digit run *unless* that run
  is embedded between letters — so `exp05ed` is recovered while `covid19` and
  `2024` are not touched. Tokens of 2 characters are skipped entirely, protecting
  `5G`, `4K`, `3D`.
* Substitute punctuation-leet (`$`, `@`, `!`) only when a letter *follows* it.
  `$hocking` is recovered; `BREAKING!!!` keeps its exclamation marks.

A useful empirical result falls out of measuring rather than assuming: the
**stylometric features are incidentally robust**. Shouting and exclamation spam
survive homoglyph and leetspeak substitution, so a model carrying them degrades
noticeably less than a lexical-only one. Defence in depth applies to features
too.

### 3.2 Calibration, abstention and the cost of being wrong

Three separate ideas hide behind "how confident is the model?".

**Calibration** asks whether a score of 0.9 corresponds to being right 90% of the
time. Margin-based classifiers are systematically over-confident, so raw scores
are not probabilities and any threshold set on them means something other than
what you think. *Expected Calibration Error* buckets predictions by confidence
and averages |confidence − accuracy| across buckets; the *Brier score* is the
mean squared error of the probabilities. Isotonic regression (flexible,
needs data) or Platt scaling (two parameters, safe on small sets) fixes the
mapping — fitted on a **held-out** split, since calibrating on training scores
merely relearns the model's own over-confidence.

**Abstention** asks which calls are safe to make at all. A selective classifier
answers only outside an uncertainty band and routes the rest to a human. The
diagnostic is the **risk–coverage curve**: as you abstain more, does the error
rate on what remains fall steeply? If it does not, the model's confidence is not
informative, which is itself a crucial finding.

`fit_policy` inverts the usual question into the one operations teams actually
plan against: *given a tolerated automated-error budget, how much of the queue
can we automate?* On separable data that is 100%; on noisy data it correctly
collapses toward "review everything" rather than quietly exceeding the budget.
A caveat worth stating: a policy fitted on finite validation data still shows a
generalisation gap, so it must be re-validated and monitored, not set once.

**Cost asymmetry** is a policy input, not a modelling one. Missing a fake story
and censoring a real one are not equally bad, and the ratio moves the optimal
cutoff away from 0.5 — downward as false negatives get more expensive.

### 3.3 Detecting by structure: the confound that makes it easy to fool yourself

Cascade-shape detection rests on a real empirical finding (Vosoughi et al.,
*Science* 2018): false stories travel deeper, through longer person-to-person
chains, while true stories are more often broadcast once and stop. **Structural
virality** — mean pairwise distance in the diffusion tree — is the measure that
separates the two even at equal cascade size: a star scores ~1.8, a chain ~4.0.

The methodological trap is worth more than the result. The intuitive setup —
seed "real" cascades from hubs and "fake" ones from ordinary users — produces a
detector with 98% accuracy after a **single** step. It looks superb and is
worthless: it is reading the poster's follower count, and would collapse the
moment a fake story is posted by a popular account. The size gap at step 1 was
4x, and no shape feature was doing any work.

Removing the confound (identical seeding, identical hop-0 dynamics, differing
only in whether chains persist) yields an honest curve: **near chance at one
step, ~0.82 by step three, then a plateau**. Depth-based features carry the
signal, exactly as the theory predicts. The general lesson: when a structural
model performs suspiciously well *immediately*, check whether it has access to a
proxy for the label that has nothing to do with the structure you meant to
measure.

---

## 4. How the two halves connect

In a deployed system the loop is:

1. **Detect** — the classifier flags a story as likely fake (Part 1).
2. **Locate** — identify the accounts currently spreading it (the seed set).
3. **Contain** — deploy the limited fact-checking budget on the structurally
   critical accounts using a strategy from Part 2.
4. **Measure** — the simulation predicts how much spread each intervention
   averts, so the budget goes where it buys the most reduction.

Detection without containment is a smoke alarm with no sprinklers; this project
implements both, and shows how much the second half is worth.
