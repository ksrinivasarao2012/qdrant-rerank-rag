---
title: Qdrant Rerank RAG Assistant
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# ⚡ Qdrant Rerank RAG Assistant

[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue?style=for-the-badge)](https://huggingface.co/spaces/Srinivasa12/rag-portfolio)
[![Qdrant](https://img.shields.io/badge/Qdrant-Cloud%20VectorDB-red?style=for-the-badge)](https://qdrant.tech/)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.1--8B-orange?style=for-the-badge)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **🚀 Live Interactive Application**: **[https://huggingface.co/spaces/Srinivasa12/rag-portfolio](https://huggingface.co/spaces/Srinivasa12/rag-portfolio)**
>
> **A High-Precision Dual-Stage RAG Pipeline featuring Hybrid Retrieval (Qdrant + BM25), Cross-Encoder Re-Ranking, Real-Time Token Streaming, and Multi-Turn Conversational Memory.**

---

## 📐 Architecture Overview

```text
                        ┌───────────────────────────────────┐
                        │      Uploaded PDF Document        │
                        └─────────────────┬─────────────────┘
                                          │
                                          ▼
                        ┌───────────────────────────────────┐
                        │    Document Processor (PyMuPDF)   │
                        │    Text Extraction & Chunking     │
                        └─────────────────┬─────────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    │                                           │
                    ▼                                           ▼
  ┌──────────────────────────────────┐        ┌──────────────────────────────────┐
  │   Dense Vector Store (Qdrant)    │        │   Sparse Keyword Index (BM25)    │
  │   Model: BAAI/bge-small-en-v1.5  │        │   Tokenizer: Alphanumeric        │
  └─────────────────┬────────────────┘        └─────────────────┬────────────────┘
                    │ (Top 15 Dense)                            │ (Top 15 Sparse)
                    └─────────────────────┬─────────────────────┘
                                          │
                                          ▼
                        ┌───────────────────────────────────┐
                        │  Reciprocal Rank Fusion (RRF)     │
                        │  Deduplication & Rank Merging     │
                        └─────────────────┬─────────────────┘
                                          │ (15 Fused Candidates)
                                          ▼
                        ┌───────────────────────────────────┐
                        │    Cross-Encoder Re-Ranker        │
                        │ Model: ms-marco-MiniLM-L-6-v2     │
                        └─────────────────┬─────────────────┘
                                          │ (Top K Filtered Chunks)
                                          ▼
                        ┌───────────────────────────────────┐
                        │    Groq LLM (Llama 3.1 8B)        │
                        │ Multi-Turn Chat History Injected  │
                        └─────────────────┬─────────────────┘
                                          │
                                          ▼
                        ┌───────────────────────────────────┐
                        │    Native Gradio Streaming UI     │
                        │    ZeroGPU Accelerated on HF      │
                        └───────────────────────────────────┘
```

---

## ✨ Key Technical Features

- **Dual-Stage Retrieval Architecture**:
  - **Stage 1a (Dense Vector Search)**: Semantic retrieval powered by Qdrant Cloud & HuggingFace `BAAI/bge-small-en-v1.5` embeddings.
  - **Stage 1b (Sparse Lexical Search)**: Keyword precision powered by `rank-bm25`.
  - **Stage 1c (Reciprocal Rank Fusion)**: Combines dense & sparse candidate rankings into a unified score list.
  - **Stage 2 (Cross-Encoder Re-Ranking)**: Re-scores candidate chunks with `cross-encoder/ms-marco-MiniLM-L-6-v2` for factual precision.
- **ZeroGPU Acceleration**: Accelerated on Hugging Face Spaces using `@spaces.GPU`.
- **Multi-Turn Conversational Memory**: Preserves context history across queries using LangChain message objects.
- **Fast PDF Processing**: Page text and structure extraction via PyMuPDF (`pymupdf`).
- **Filter by Document**: Filter questions across all indexed documents or specific PDF files.

---

## 🛠️ Tech Stack

* **UI & Deployment**: Gradio 4.44.0, Hugging Face ZeroGPU
* **Vector Database**: Qdrant Cloud
* **Embeddings**: HuggingFace (`BAAI/bge-small-en-v1.5`)
* **Keyword Search**: `rank-bm25` (BM25Okapi)
* **Re-Ranker**: SentenceTransformers (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
* **LLM Engine**: Groq (`llama-3.1-8b-instant`) via LangChain
* **PDF Engine**: PyMuPDF (`fitz`)

---

## 🚀 Local Setup Guide

### 1. Clone the Repository
```bash
git clone https://huggingface.co/spaces/Srinivasa12/rag-portfolio
cd rag-portfolio
```

### 2. Install Dependencies
```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
```

### 3. Run the Gradio App
```powershell
.\.venv\Scripts\python.exe app.py
```
Open **`http://127.0.0.1:7860`** in your browser.

---

## 📜 License
MIT License. Created by [K. Srinivasa Rao](https://github.com/ksrinivasarao2012).
