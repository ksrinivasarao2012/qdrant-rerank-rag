# 📘 Project Master Log & Handover Guide
**Project:** Qdrant Rerank RAG Portfolio Engine  
**Live Deployment:** [Hugging Face Spaces](https://huggingface.co/spaces/Srinivasa12/rag-portfolio)  
**GitHub Repository:** [ksrinivasarao2012/qdrant-rerank-rag](https://github.com/ksrinivasarao2012/qdrant-rerank-rag)  
**Last Updated:** August 20, 2026  

---

## 1. 🏗️ Architecture & Scale Overview

* **Corpus Scale:** **93,455 StackExchange Q&A discussions (218,456 indexed chunks)** from Cross Validated (`stats.stackexchange.com`).
* **Vector & Sparse Storage:** **Qdrant Cloud**
  * **Dense Model:** `BAAI/bge-small-en-v1.5` (384-dimensional dense vectors).
  * **Sparse Model:** Qdrant Native Sparse Vectors (Fast BM25 keyword matching with sub-second execution, eliminating in-memory RAM bloat).
  * **Fusion:** **Reciprocal Rank Fusion (RRF)** merging top dense and sparse candidate pools ($K=10$ to $K=15$).
* **Neural Re-Ranking:** **Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)**
  * Decouples coarse retrieval from fine relevance scoring.
  * Re-scores candidate pairs to filter semantic distractors before generation.
* **Generation Engine:** **Groq LPU (`openai/gpt-oss-20b` via `gptoss_simple_v1`)**
  * Fast token-by-token streaming with low reasoning delay.
  * Multi-model cascade fallbacks (`openai/gpt-oss-120b`, `deepseek-r1-distill-llama-70b`, `gemma2-9b-it`).
* **Serving & Interfaces:**
  * **FastAPI Backend:** Asynchronous NDJSON streaming endpoint (`/chat/stream`) with conversational query rewriting and sub-second TTFT.
  * **Gradio Web Interface:** Deployed on **Hugging Face ZeroGPU** with MathJax/LaTeX support and verified StackExchange source citations.

---

## 2. 🚀 Deployment & Bug Fix History

| Component | Issue Encountered | Root Cause | Permanent Resolution |
| :--- | :--- | :--- | :--- |
| **HF Build** | `Exit code: 137 (OOMKilled)` | Heavy offline testing packages (`deepeval`, `black`, `pytest`) triggered pip resolver memory exhaustion. | Cleaned production [`requirements.txt`](file:///d:/Rag-Portfolio/requirements.txt) down to lean production runtime dependencies. |
| **Gradio Launch** | `TypeError: argument of type 'bool' is not iterable` | Gradio 4.44 `get_api_info()` crashed parsing boolean `"additionalProperties": true` from Pydantic 2.9+ schemas. | Passed `show_api=False` in `demo.launch()` and pinned `pydantic>=2.0,<2.9.0`. |
| **Gradio Chatbot** | `Data incompatible with tuples format` | Gradio 4.44 `gr.Chatbot` defaults to 2-element tuples `[user, bot]` unless configured for message dicts. | Initialized `gr.Chatbot(type="messages")` in [`app.py`](file:///d:/Rag-Portfolio/app.py). |
| **Groq LLM** | `Error code: 404 / 400 (model_decommissioned)` | Legacy Llama 3 / 3.1 70B & 8B tags were decommissioned on Groq. | Migrated to Groq's active flagship model **`openai/gpt-oss-20b`** (`gptoss_simple_v1`) with resilient multi-model fallbacks. |

---

## 3. 🎯 Golden Benchmark Curation & LLM Audit Status

Total Dataset: **294 evaluation test cases across 10 categories**.

### **Category 1: `multi_turn` (18 Conversational Cases)** — `[STATUS: 100% COMPLETE & VERIFIED]`
* **Fixes Applied:**
  * Reclassified 4 standalone comparisons (`mturn_04`, `mturn_11`, `mturn_17`, `mturn_19`) to `multi_hop`.
  * Expanded single gold answers to multi-gold sibling sets across identical StackExchange discussion threads.
  * Aligned `mturn_15` (learning rate decay annealing) and `mturn_16` (zero-deviation variance shrinkage under mean substitution).
* **Audit Result:** **18 / 18 (100% PASS)** via `Qwen-2.5-7B` judge.

### **Category 2: `negation` (20 Exclusion Cases)** — `[STATUS: 100% COMPLETE & APPLIED]`
* **Fixes Applied:**
  * Replaced 16 forbidden-method definitions with verified alternative techniques:
    * `neg_01` (Normality without Shapiro-Wilk) $\rightarrow$ Anderson-Darling, KS, QQ plots (`1648`, `74954`, `52297`).
    * `neg_04` (Dimensionality reduction without PCA) $\rightarrow$ Isomap, Autoencoders, UMAP (`124545`, `262060`, `410405`).
    * `neg_05` (Feature selection without stepwise) $\rightarrow$ Lasso, Elastic Net, RF (`13698`, `27310`).
    * `neg_06` (Time series without ARIMA) $\rightarrow$ Exponential Smoothing, State Space (`125016`, `157379`).
    * `neg_07` (Clustering without $k$) $\rightarrow$ DBSCAN, Hierarchical dendrogram (`195481`, `3692`).
    * `neg_10` (Non-parametric correlation without Pearson) $\rightarrow$ Spearman $\rho$, Kendall $\tau$ (`3733`, `3744`, `3946`).
    * `neg_11` (Dependency without Mutual Info) $\rightarrow$ Distance Correlation, MIC (`394490`, `77016`, `50680`).
    * `neg_19` (Categorical mapping without one-hot) $\rightarrow$ Feature Hashing, Target Encoding (`411775`, `298951`).
* **Audit Result:** Applied to [`evaluation/golden_dataset.json`](file:///d:/Rag-Portfolio/evaluation/golden_dataset.json).

### **Category 3: `multi_hop` (24 Comparison Cases)** — `[STATUS: AUDITED & QUEUED]`
* Full LLM judge audit completed in [`multi_hop_audit_report.md`](file:///C:/Users/Srinivasa/.gemini/antigravity-ide/brain/2aac9f65-d485-4026-bc73-ba733300333b/multi_hop_audit_report.md).
* Next step: Apply candidate replacement IDs and verify database existence.

### **Category 4: `niche_topic` (22 Domain-Specific Cases)** — `[STATUS: AUDITED & QUEUED]`
* Full LLM judge audit completed in [`niche_topic_audit_report.md`](file:///C:/Users/Srinivasa/.gemini/antigravity-ide/brain/2aac9f65-d485-4026-bc73-ba733300333b/niche_topic_audit_report.md).
* Next step: Apply candidate replacement IDs and verify database existence.

---

## 4. 🛠️ Essential Command Reference

### Run Automated LLM-as-a-Judge Audits:
```powershell
# Audit multi_turn category
python evaluation/audit_golden_dataset.py --category multi_turn

# Audit negation category
python evaluation/audit_golden_dataset.py --category negation

# Audit both multi_turn and negation combined
python evaluation/audit_golden_dataset.py --category multi_turn,negation

# Audit all 5 curated categories
python evaluation/audit_golden_dataset.py
```

### Run Contextual Recall & Retrieval Verification:
```powershell
# Evaluate Contextual Recall on negation (local index)
python evaluation/eval_contextual_recall.py --category negation --n 20 --local

# Evaluate Contextual Recall on multi_turn (local index)
python evaluation/eval_contextual_recall.py --category multi_turn --n 18 --local

# Run standard Recall@K & MRR retriever evaluation
python evaluation/eval_retriever.py --category standard --k 3 --local
```

### Deploy Updates to Hugging Face Spaces:
```powershell
git push -f hf main
```

---

## 5. 🗺️ Next Steps Roadmap

1. **Step 1:** Review & apply **Category 3 (`multi_hop` - 24 cases)** into `evaluation/golden_dataset.json`.
2. **Step 2:** Review & apply **Category 4 (`niche_topic` - 22 cases)** into `evaluation/golden_dataset.json`.
3. **Step 3:** Execute full **Contextual Recall & MRR Benchmarks** across the entire 294-case dataset and generate the final benchmark score report.
4. **Step 4:** Phase 2 retrieval optimizations (Multi-hop RRF query decomposition & negation token filtering).

---

## 6. 📝 Living Chronological Change Log & Decision History

*This section is updated after **EVERY** modification to maintain full transparency on WHAT was changed, WHY it was changed, and the verified OUTCOME.*

| Timestamp | Component / File | What Was Changed | Why We Did It (Rationale) | Verified Outcome / Impact |
| :--- | :--- | :--- | :--- | :--- |
| **2026-08-18** | `requirements.txt`, `app.py` | Stripped offline testing packages; added `show_api=False` and `type="messages"`. | HF Space crashed during build (`OOMKilled exit 137`) and launch (`Pydantic 2.9 schema bug` + `Chatbot tuples error`). | Space successfully built, launched, and started serving on ZeroGPU. |
| **2026-08-18** | `system_prompts.yaml`, `llm_service.py` | Switched active answer prompt to `openai/gpt-oss-20b` (`gptoss_simple_v1`) with fallback cascade. | Groq decommissioned legacy `llama-3.1-70b-versatile` tag, causing HTTP 400 `model_decommissioned`. | Restored instant token-by-token answer streaming on live Space. |
| **2026-08-18** | `golden_dataset.json` (Multi-Turn) | Reclassified 4 standalone comparisons to `multi_hop` and added sibling multi-golds across 18 cases. | Single-gold labels caused artificial recall failures when the retriever fetched valid sibling answers from the same thread. | Multi-turn category expanded with verified sibling post IDs. |
| **2026-08-19** | `golden_dataset.json` (Negation) | Updated 16 negation cases with verified alternative methods (Isomap, Anderson-Darling, DBSCAN, etc.). | 80% of original negation golds explained the *forbidden* method rather than the *alternative*. | All 20 negation cases mapped to verified alternative posts in `posts.jsonl`. |
| **2026-08-20** | `golden_dataset.json` (`mturn_15`) | Updated gold IDs to `["235052", "384202"]`. | Old gold (`352037`) was a general debugging checklist; new posts directly explain learning rate annealing and convergence. | LLM Judge evaluated and marked **`mturn_15: [+] OK / PASS`**. |
| **2026-08-20** | `golden_dataset.json` (`mturn_16`) | Preserved exact original query "Why does this artificially reduce variance?" with multi-golds `["78065", "440047", "45868"]`. | Kept question 100% untouched while mapping to the 3 posts in database discussing zero-deviation variance shrinkage and data regularity. | Question preserved with verified variance reduction answers. |
| **2026-08-20** | `audit_golden_dataset.py` | Supported comma-separated categories (e.g. `--category multi_turn,negation`). | User requested a single unified command to audit multiple categories sequentially without manual runs. | Enables single-command multi-category audits. |

