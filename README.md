# Clothing Search Engine (Information Retrieval Assignment 1)
**Course**: CSD358 - Information Retrieval  
**Topic**: Vector Space Model (VSM `lnc.ltc`) & Positional Inverted Indexing on 100-Document Clothing Corpus  
**GitHub Repository**: [https://github.com/Saathvik-Mullapudi/IR_assgn1](https://github.com/Saathvik-Mullapudi/IR_assgn1)

---

## 1. Project Overview & Deliverables Summary

This project implements a complete, self-contained clothing search engine over an official 100-document product corpus across 10 apparel categories (*T-shirt, Shirt, Jeans, Kurta, Saree, Dress, Hoodie, Jacket, Leggings, Sweatshirt*). 

### Deliverables Checklist (As Specified in Assignment Sheet)
- [x] **Source Code with Full Documentation**: Clean, modular, well-commented Python architecture in `src/`, `app.py`, and `scripts/`.
- [x] **Dictionary / Inverted-Index Output**: Exported to [`inverted_index.json`](inverted_index.json) and [`output/inverted_index.json`](output/inverted_index.json).
- [x] **Positional-Index Output**: Exported to [`positional_index.json`](positional_index.json) and [`output/positional_index.json`](output/positional_index.json).
- [x] **Screenshots of the Application & Representative Queries**: Saved in [`screenshots/`](screenshots/) demonstrating VSM search, phrase search, proximity search, and dual comparison.
- [x] **Evaluation Benchmark Report**: Complete results for all 10 free-text, 5 exact phrase, 4 proximity, and 1 non-corpus query in [`evaluation_report.md`](evaluation_report.md).
- [x] **Blackboard Submission ZIP Archive**: Bundled in [`IR_Assignment1_CSD358.zip`](IR_Assignment1_CSD358.zip).

---

## 2. Core Concepts Implemented

| Component / Concept | Technical Specification & Implementation |
| :--- | :--- |
| **Corpus Management (Part A)** | Ingests the 100-document collection (`N=100`), parsing structured fields (`<DOCID>`, `<CATEGORY>`, `<TITLE>`, `<TEXT>`). Enforces unique sequential IDs (`D001` to `D100`) and equal category distribution (10 items each). |
| **Pre-Processing Pipeline (Part A)** | Lowercasing, possessive removal (`Men's` $\to$ `men`), hyphen splitting (`t-shirt` $\to$ `t shirt`), stop-word removal with documented domain protection (retains `wear` and `fit`), and Porter Stemming (`shirts` $\to$ `shirt`, `breathable` $\to$ `breathabl`). Tracks 1-indexed token positions. |
| **Standard Inverted Index (Part A & B)** | Dictionary mapping unique vocabulary terms $t \to df_t$ (Document Frequency). Postings lists storing sorted records of $(docID, tf_{t,d})$. Pre-computes Euclidean document vector lengths $\|d\|$ for instant cosine normalization. |
| **Ranked Retrieval: VSM (Part B)** | Implements the exact **`lnc.ltc`** SMART weighting scheme:<br>• **Document ($lnc$)**: $w_{d,t} = 1 + \log_{10}(tf)$ for $tf > 0$, no idf, cosine normalized by $\|d\|$.<br>• **Query ($ltc$)**: $w_{q,t} = (1 + \log_{10}(tf_{q,t})) \times \log_{10}(N / df_t)$, cosine normalized by $\|q\|$.<br>• Cosine score: $\frac{\vec{w_q} \cdot \vec{w_d}}{\|q\| \cdot \|d\|}$. Results sorted by decreasing score; ties broken by **increasing docID**. Returns top 10. |
| **Positional Index (Part C)** | Extends inverted postings to store occurrence offsets: $(term \to df \to [(docID, tf, [p_1, p_2, \dots])])$. Supports multi-word Exact Phrase Search via linear positional intersect ($pos_{i+1} = pos_i + 1$) and Ordered Proximity Search ($1 \le pos_2 - pos_1 \le k$). |
| **Interactive Web UI (Part D)** | Lightweight, modern web application with free-text VSM search, phrase/proximity search, and explicit display of matching token positions as proof of positional index usage. |
| **Novelty Feature (+3 Marks)** | **Side-by-Side Dual Comparison Mode**: Simultaneously executes both models on the query, highlighting VSM false positives that lack syntactic word adjacency, providing direct visual and analytical proof of why positional indexing improves precision. |

---

## 3. Directory Structure

```text
IR_assgn1/
├── app.py                      # Flask web application & search API
├── inverted_index.json         # Standard inverted index deliverable (Part A/B)
├── positional_index.json       # Positional inverted index deliverable (Part C)
├── evaluation_report.md        # Comprehensive report of all 19 mandatory test queries (Part E)
├── data/
│   ├── clothing_corpus_raw.txt # Original SGML/XML corpus (100 documents)
│   └── clothing_corpus.json    # Structured JSON corpus
├── output/
│   ├── inverted_index.json     # Mirrored index deliverable
│   ├── positional_index.json   # Mirrored positional index deliverable
│   └── evaluation_report.md    # Mirrored evaluation report
├── screenshots/                # Application & query screenshots
│   ├── vsm_search_results.png
│   ├── phrase_search_results.png
│   ├── proximity_search_results.png
│   └── dual_comparison_results.png
├── src/                        # Core search engine modules
│   ├── __init__.py
│   ├── corpus.py               # Document & CorpusManager classes
│   ├── preprocessor.py         # Tokenizer, stop-word policy, and Porter stemmer
│   ├── inverted_index.py       # InvertedIndex & Posting classes
│   ├── vsm_search.py           # VSMSearchEngine with lnc.ltc weighting
│   └── positional_index.py     # PositionalIndex & phrase/proximity algorithms
├── templates/
│   └── index.html              # Search engine frontend interface
├── scripts/
│   └── run_benchmarks.py       # Automated benchmark script for all 19 mandatory queries
└── tests/                      # Automated unit test suite (26 passing tests)
    ├── __init__.py
    ├── test_preprocessor.py
    ├── test_inverted_index.py
    ├── test_vsm_search.py
    ├── test_positional_index.py
    └── test_app.py
```

---

## 4. How to Run the Project

### Prerequisites
- Python 3.8+
- Standard dependencies: `flask`, `nltk` (pure rule-based Porter stemmer, no extra downloads needed)

```bash
pip install flask nltk
```

### 1. Run Automated Unit Tests (26 Tests)
```bash
python -m unittest discover tests
```

### 2. Run the Benchmark Suite (Part E)
Executes all 10 free-text, 5 phrase, 4 proximity, and 1 non-corpus query, regenerating `evaluation_report.md`:
```bash
python scripts/run_benchmarks.py
```

### 3. Launch the Interactive Web Application (Part D)
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```
- Click between **Free-Text (VSM)**, **Phrase & Proximity**, and **Dual Comparison** tabs.
- Click any of the pre-configured sample query chips for instant testing.

---

## 5. Stop-Word Policy Justification

1. **Noise Reduction**: High-frequency grammatical function words (articles, prepositions, auxiliary verbs, conjunctions like *the, is, at, of, with, from*) occur uniformly across nearly all documents. Keeping them inflates index size and causes artificial dot-product inflation without adding semantic value.
2. **Preservation of Clothing Domain Terms**: Generic stop-word lists often remove terms like *wear* or *fit*. In clothing retrieval, these are domain-critical keywords (*festive wear*, *winter wear*, *regular fit*, *slim fit*). They are explicitly retained in our stop-word policy.
3. **Symmetric Consistency**: The exact same pre-processing pipeline is applied symmetrically to both documents at index time and queries at search time.

---

## 6. Key Findings from Comparative Analysis

1. **Exact Phrase Precision (`"cotton shirt"`)**:
   - In VSM (lnc.ltc), documents like `D001` (T-shirt: *"Men's Cotton Crew Neck T-Shirt... 100% cotton, this t-shirt..."*) receive high scores due to high frequency of "cotton" and stemmed "shirt".
   - Positional Index strictly isolates true checked shirts (`D002`, `D022`, `D042`, etc.) where `cotton` is immediately followed by `shirt` ($pos_2 = pos_1 + 1$).
2. **Semantic Binding (`"stretch denim"`)**:
   - VSM accumulates scores for denim jackets (`D018`) and stretch leggings (`D009`).
   - Positional Index matches only true stretch denim jeans (`D003`, `D013`, `D033`, etc.) where words are consecutive.
3. **Proximity Flexibility (`cotton WITHIN/3 shirt`)**:
   - Relaxing the distance window to $k=3$ expands recall from 5 strict phrase shirts to 15 products, capturing items like "cotton crew neck t-shirt" ($pos=(17, 19)$) while still enforcing proper word order.
