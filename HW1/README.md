# Spelling Correction using Edit Distance and Language Models

## Implementation of a Noisy Channel Model for Bulgarian Text

**Author:** Georgi H. Lazov
**Student ID:** 0MI0600299
**Course:** Information Retrieval and Deep Learning
**Instructor:** Prof. Stoyan Mihov
**Semester:** Winter 2025/2026

---

## Abstract

This work presents an implementation of a spelling correction system for Bulgarian text based on the noisy channel model. The system combines a modified Levenshtein edit distance with an n-gram language model to identify and correct spelling errors. The implementation includes functions for computing edit distance, weighted edit distance, optimal alignment, edit weight training, candidate generation, and final spelling correction using interpolated language model probabilities.

**Keywords:** Spelling Correction, Edit Distance, Levenshtein Distance, N-gram Language Model, Noisy Channel Model, Bulgarian Language

---

## 1. Introduction

### 1.1 Motivation

Spelling correction is a fundamental task in natural language processing with applications in search engines, text editors, and machine translation. The noisy channel model provides a principled probabilistic framework for this task.

### 1.2 Project Objectives

1. Implement modified Levenshtein edit distance with transposition operations
2. Implement weighted edit distance using learned operation costs
3. Find optimal alignments between strings
4. Train edit weights from a corpus of spelling errors
5. Generate correction candidates within edit distance 2
6. Combine edit distance with language model for final correction

### 1.3 Corpus Used

The system was trained and evaluated using the **Corpus of Journalistic Texts for Southeastern Europe**, provided by the Institute for Bulgarian Language at the Bulgarian Academy of Sciences:
- **Number of documents:** 35,337
- **Size:** 7.9 MB
- **Language:** Bulgarian
- **Source:** http://dcl.bas.bg/BulNC-registration/

---

## 2. Theoretical Background

### 2.1 Noisy Channel Model

The spelling correction problem is modeled as finding the most likely intended word $w$ given an observed (possibly misspelled) word $r$:

$$\hat{w} = \arg\max_w P(w|r) = \arg\max_w P(r|w) \cdot P(w)$$

where:
- $P(r|w)$ is the **error model** (probability of typing $r$ when intending $w$)
- $P(w)$ is the **language model** (prior probability of word $w$)

### 2.2 Modified Levenshtein Distance

The standard Levenshtein distance allows three operations:
- **Insertion:** Insert a character
- **Deletion:** Delete a character
- **Substitution:** Replace one character with another

This implementation extends the distance with:
- **Split:** One character becomes two (e.g., typing "ab" instead of "a")
- **Merge:** Two characters become one (e.g., typing "a" instead of "ab")

The recurrence relation becomes:

$$M[i,j] = \min \begin{cases}
M[i-1,j-1] + \mathbb{1}[s_1[i] \neq s_2[j]] & \text{(substitution/match)} \\
M[i-1,j] + 1 & \text{(deletion)} \\
M[i,j-1] + 1 & \text{(insertion)} \\
M[i-2,j-1] + 1 & \text{(merge)} \\
M[i-1,j-2] + 1 & \text{(split)}
\end{cases}$$

### 2.3 Weighted Edit Distance

Instead of unit costs, we use learned weights based on the negative log probability of each operation:

$$w(a \to b) = -\log P(a \to b)$$

The weights are estimated from a corpus of spelling errors using maximum likelihood estimation.

### 2.4 N-gram Language Model

The language model uses interpolated n-gram probabilities:

$$P_{\text{interp}}(w|c) = \alpha \cdot P_{\text{MLE}}(w|c) + (1-\alpha) \cdot P_{\text{interp}}(w|c')$$

where $c'$ is the context with the first word removed (backoff).

### 2.5 Perplexity

Model quality is measured using perplexity:

$$\text{PPL} = 2^{H(P)}$$

where $H(P)$ is the cross-entropy rate of the model on test data.

---

## 3. Implementation

### 3.1 Edit Distance (`a1.py`)

#### 3.1.1 editDistance

Computes the modified Levenshtein distance matrix:

```python
def editDistance(s1: str, s2: str) -> np.ndarray:
    M = np.zeros((len(s1)+1, len(s2)+1))

    # Base cases
    for i in range(1, len(s1)+1):
        M[i, 0] = i
    for j in range(1, len(s2)+1):
        M[0, j] = j

    # Fill matrix
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            M[i, j] = min(
                M[i-1, j-1] + (s1[i-1] != s2[j-1]),  # substitution
                M[i-1, j] + 1,                        # deletion
                M[i, j-1] + 1                         # insertion
            )
            if i > 1:
                M[i, j] = min(M[i, j], M[i-2, j-1] + 1)  # merge
            if j > 1:
                M[i, j] = min(M[i, j], M[i-1, j-2] + 1)  # split

    return M
```

**Complexity:** O(n × m) where n and m are string lengths

#### 3.1.2 editWeight

Computes weighted edit distance using learned operation costs:

```python
def editWeight(s1: str, s2: str, Weight: dict) -> float:
    M = np.zeros((len(s1)+1, len(s2)+1))

    for i in range(1, len(s1)+1):
        M[i, 0] = M[i-1, 0] + Weight[(s1[i-1], '')]

    for j in range(1, len(s2)+1):
        M[0, j] = M[0, j-1] + Weight[('', s2[j-1])]

    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            M[i, j] = min(
                M[i-1, j-1] + Weight[(s1[i-1], s2[j-1])],
                M[i-1, j] + Weight[(s1[i-1], '')],
                M[i, j-1] + Weight[('', s2[j-1])]
            )
            # Merge and split operations...

    return M[len(s1), len(s2)]
```

#### 3.1.3 bestAlignment

Finds the optimal alignment by backtracking through the edit distance matrix:

```python
def bestAlignment(s1: str, s2: str) -> list[tuple[str, str]]:
    M = editDistance(s1, s2)
    alignment = []
    i, j = len(s1), len(s2)

    while i > 0 and j > 0:
        if M[i-1, j] == M[i, j] - 1:
            alignment.append((s1[i-1], ''))  # deletion
            i -= 1
        elif M[i, j-1] == M[i, j] - 1:
            alignment.append(('', s2[j-1]))  # insertion
            j -= 1
        # ... other operations

    return list(reversed(alignment))
```

**Example:**
```
bestAlignment('редакция', 'рдашиа') =
[('р','р'), ('е',''), ('д','д'), ('а','а'), ('кц','ш'), ('и','и'), ('я','а')]
```

### 3.2 Weight Training

#### 3.2.1 trainWeights

Learns edit operation weights from a corpus of (error, correction) pairs:

```python
def trainWeights(corpus: list[tuple[str, str]]) -> dict:
    ids = subs = ins = dels = splits = merges = 0

    for q, r in corpus:
        alignment = bestAlignment(q, r)
        for op in alignment:
            # Count operation types...

    N = ids + subs + ins + dels + splits + merges

    weight = {}
    for a in alphabet:
        weight[(a, a)] = -math.log(ids / N)
        weight[(a, '')] = -math.log(dels / N)
        weight[('', a)] = -math.log(ins / N)
        # ...

    return weight
```

### 3.3 Candidate Generation

#### 3.3.1 generateEdits

Generates all strings at edit distance 1 from the input:

```python
def generateEdits(q: str) -> list[str]:
    edits = []

    for i in range(len(q) + 1):
        for a in alphabet:
            edits.append(q[:i] + a + q[i:])  # insertion

    for i in range(len(q)):
        edits.append(q[:i] + q[i+1:])  # deletion
        for a in alphabet:
            edits.append(q[:i] + a + q[i+1:])  # substitution
            for b in alphabet:
                edits.append(q[:i] + a + b + q[i+1:])  # split

    for i in range(len(q) - 1):
        for a in alphabet:
            edits.append(q[:i] + a + q[i+2:])  # merge

    return edits
```

#### 3.3.2 generateCandidates

Generates all valid candidates within edit distance 2:

```python
def generateCandidates(query: str, dictionary: dict) -> list[str]:
    L = []
    if allWordsInDictionary(query):
        L.append(query)

    for query1 in generateEdits(query):
        if allWordsInDictionary(query1):
            L.append(query1)
        for query2 in generateEdits(query1):
            if allWordsInDictionary(query2):
                L.append(query2)

    return L
```

### 3.4 Spelling Correction

#### 3.4.1 correctSpelling

Combines edit distance with language model:

```python
def correctSpelling(r: str, model: MarkovModel, weights: dict,
                    mu: float = 1.0, alpha: float = 0.9):
    result = ''
    maxProbability = float('-inf')
    candidates = generateCandidates(r, model.kgrams[tuple()])

    for candidate in candidates:
        curProbability = (
            -editWeight(r, candidate, weights) +
            mu * model.sentenceLogProbability(candidate.split(" "), alpha)
        )
        if curProbability > maxProbability:
            result = candidate
            maxProbability = curProbability

    return result
```

**Scoring Function:**
$$\text{score}(c) = -w(r, c) + \mu \cdot \log P_{\text{LM}}(c)$$

where:
- $w(r, c)$ is the weighted edit distance
- $P_{\text{LM}}(c)$ is the language model probability
- $\mu$ controls the weight of the language model

### 3.5 Markov Language Model (`langmodel.py`)

The `MarkovModel` class implements an interpolated n-gram language model with:
- Dictionary extraction with frequency cutoff
- Unknown word handling
- K-gram extraction for contexts up to length K
- Maximum likelihood probability estimation
- Interpolated probability with backoff
- Perplexity evaluation

---

## 4. Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| K | 3 | N-gram order (trigram) |
| dictionaryLimit | 10,000 | Maximum vocabulary size |
| alpha | 0.9 | Interpolation coefficient |
| mu | 1.0 | Language model weight |
| maxEditDistance | 2 | Maximum edit distance for candidates |

---

## 5. Results

### 5.1 Edit Distance Examples

| String 1 | String 2 | Distance |
|----------|----------|----------|
| "редакция" | "рдашиа" | 4 |
| "корекция" | "корекцияа" | 1 |
| "българси" | "български" | 1 |

### 5.2 Alignment Example

For strings "редакция" → "рдашиа":
```
р → р  (match)
е → ∅  (deletion)
д → д  (match)
а → а  (match)
кц → ш (merge + substitution)
и → и  (match)
я → а  (substitution)
```

---

## 6. Conclusions

### 6.1 Achieved Results

1. **Successful implementation** of modified Levenshtein distance with merge/split operations
2. **Weighted edit distance** using learned operation probabilities
3. **Optimal alignment** extraction through backtracking
4. **Candidate generation** within edit distance 2
5. **Noisy channel spelling correction** combining edit distance with language model

### 6.2 Future Work

- Implement phonetic similarity for candidate generation
- Use character-level neural language models
- Add context-aware correction
- Implement real-word error correction

---

## 7. Technical Details

### 7.1 Requirements

```
Python >= 3.5
numpy
```

### 7.2 Project Structure

```
HW1/
├── README.md                 # This document
├── TI_HW1.pdf               # Assignment description
├── a1/
│   ├── a1.pdf               # Detailed instructions
│   ├── README.txt           # Setup instructions
│   ├── langmodel.py         # Markov language model
│   ├── corpus/              # Training corpus
│   └── ...
├── solution/
│   ├── a1.py                # Implementation
│   └── TI_HW1.pdf          # Theoretical proofs
└── venv/                    # Virtual environment
```

### 7.3 Execution

```bash
# Create and activate virtual environment (first time only)
cd HW1
python3 -m venv venv
source venv/bin/activate
pip install numpy

# Navigate to working directory
cd a1

# Run in Python interpreter
python
>>> import langmodel
>>> import a1
>>>
>>> # Load corpus and build language model
>>> corpus = [...]  # Load from corpus file
>>> model = langmodel.MarkovModel(corpus, K=3)
>>>
>>> # Compute edit distance
>>> M = a1.editDistance("редакция", "рдашиа")
>>> print(M[-1, -1])  # 4.0
>>>
>>> # Get optimal alignment
>>> alignment = a1.bestAlignment("редакция", "рдашиа")
>>> print(list(alignment))
>>>
>>> # Correct spelling
>>> weights = a1.trainWeights(error_corpus)
>>> corrected = a1.correctSpelling("българси език", model, weights)
>>> print(corrected)  # "български език"
```

---

## References

1. Jurafsky, D., & Martin, J. H. (2024). *Speech and Language Processing*. Chapter 2: Minimum Edit Distance.
2. Kernighan, M. D., Church, K. W., & Gale, W. A. (1990). *A Spelling Correction Program Based on a Noisy Channel Model*. COLING 1990.
3. Brill, E., & Moore, R. C. (2000). *An Improved Error Model for Noisy Channel Spelling Correction*. ACL 2000.
4. Corpus of Journalistic Texts for Southeastern Europe, Institute for Bulgarian Language, Bulgarian Academy of Sciences.

---

*This document was created as part of Homework Assignment 1 for the course "Information Retrieval and Deep Learning", Faculty of Mathematics and Informatics, Sofia University "St. Kliment Ohridski", 2025/2026*