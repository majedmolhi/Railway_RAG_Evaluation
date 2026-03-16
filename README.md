# Railway RAG: Retrieval-Augmented Generation for Railway Technical Documentation

This repository contains the code, data, and evaluation results for the paper:

> **Retrieval-Augmented Generation for Railway Technical Knowledge: A Comparative Evaluation of Retrieval Strategies and Hallucination Reduction**

---

## Overview

We evaluate RAG applied to railway and urban metro technical documentation. The system retrieves relevant passages from 8 official railway documents and conditions GPT-4o-mini to produce grounded, cited answers.

**Key results:**
- Hybrid RRF + Cross-Encoder Reranking achieves **Answer@5 = 1.000** and **MRR = 0.904**
- RAG improves Token F1 by **+0.442** (+165%) over LLM-only baseline
- RAG faithfulness = **0.980** (LLM-as-judge)
- List-aware chunking fix raises Answer@5 from **0.965 → 1.000**

---

## Repository Structure

```
urban_metro_rag/
├── data/
│   ├── raw_pdfs/          # 8 railway technical documents (see note below)
│   ├── sections/          # Pre-extracted section text files
│   ├── Metadata.csv       # Document metadata (title, org, year, URL)
│   └── gold_QA.csv        # Gold benchmark: 115 question-answer pairs
├── results/
│   ├── retrieval_ablation.csv        # Retrieval ablation results
│   ├── retrieval_per_document.csv    # Per-document retrieval results
│   ├── eval_summary.csv              # Generation evaluation results
│   ├── hallucination_cases.json      # Flagged hallucination cases
│   ├── retrieval_failures.json       # Retrieval failure analysis
│   └── retrieval_ablation_summary.txt
├── rag_pipeline.py        # Core RAG pipeline (chunking, indexing, retrieval)
├── retrieval_eval.py      # Retrieval ablation study (4 strategies)
├── evaluation.py          # Generation evaluation (RAG vs LLM-only)
├── test_single_query.py   # Interactive single-query demo with citations
├── analyze_failures.py    # Retrieval failure analysis
├── check_chunk.py         # Search chunks by keyword
├── requirements.txt       # Python dependencies
└── .env.example           # Environment variable template
```

---

## Dataset

### Gold Benchmark (`data/gold_QA.csv`)

115 question-answer pairs constructed through systematic review of the 8 source documents. Answers are verbatim extracts from source passages.

| Field | Description |
|---|---|
| `question_id` | Unique identifier (Q001–Q115) |
| `question` | Natural language question |
| `gold_answer` | Verbatim answer from source document |
| `document_id` | Source document identifier |
| `section` | Source section |

### Document Corpus

| ID | Title | Organisation | Year | Pages |
|---|---|---|---|---|
| RSSB_M1_2023 | Train Accident and Evacuation Procedures | RSSB | 2023 | 28 |
| FRA_PassTrainEP_1998 | Passenger Train Emergency Preparedness | FRA | 1998 | 54 |
| FRA_LERT_2021 | Locomotive Emergency Response Training | FRA | 2021 | 31 |
| NR_ESG_2025 | Emergency Services Guidance Action Cards | Network Rail | 2025 | 12 |
| ERA_ERTMSFRS_2007 | ERTMS Functional Requirements Specification | ERA | 2007 | 98 |
| ERA_ERTMS033281_2023 | Train Detection System Compatibility | ERA | 2023 | 58 |
| SCRRA_TRACKMAINT_2020 | SCRRA Track Maintenance Manual | SCRRA | 2020 | 335 |
| JICA_OM_2023 | Operations and Maintenance of Urban Railways | JICA | 2023 | 31 |

> **Note:** Raw PDFs are not included in this repository due to copyright restrictions. They are publicly available from the source organisations; URLs are provided in `data/Metadata.csv`.

---

## Installation

```bash
git clone https://github.com/majedmolhi/Railway_RAG_Evaluation.git
cd Railway_RAG_Evaluation
pip install -r requirements.txt
```

### Environment Setup

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=sk-...
```

---

## Usage

### 1. Build the Index

```bash
python rag_pipeline.py
```

This reads documents from `data/sections/`, chunks them, generates BGE-large-en-v1.5 embeddings, and stores them in ChromaDB.

### 2. Run Retrieval Ablation

```bash
python retrieval_eval.py
```

Evaluates 4 retrieval strategies (Dense, BM25, Hybrid RRF, Hybrid+Reranker) on all 115 gold questions. Results saved to `results/`.

### 3. Run Generation Evaluation

```bash
python evaluation.py
```

Compares RAG vs LLM-only across Token F1, ROUGE-L, BERTScore, and Faithfulness. Results saved to `results/eval_summary.csv`.

### 4. Ask a Single Question

```bash
python test_single_query.py --q "What is the standard track gage on SCRRA?"
```

Returns a grounded answer with full document citation.

---

## Retrieval System

| Component | Choice |
|---|---|
| Embedding model | BAAI/bge-large-en-v1.5 (1024-dim) |
| Vector store | ChromaDB + HNSW, cosine similarity |
| Sparse retrieval | BM25 (rank-bm25) |
| Fusion | Reciprocal Rank Fusion (k=60) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Generation model | GPT-4o-mini (temperature=0.0) |

---

## Results Summary

### Retrieval Ablation (115 questions)

| Strategy | Ans@1 | Ans@5 | MRR |
|---|---|---|---|
| Dense | 0.774 | 0.974 | 0.866 |
| BM25 | 0.783 | 0.965 | 0.856 |
| Hybrid RRF | 0.791 | 1.000 | 0.876 |
| **Hybrid+Reranker** | **0.835** | **1.000** | **0.904** |

### Generation (RAG vs LLM-only)

| Metric | RAG | LLM-only | Delta |
|---|---|---|---|
| Token F1 | 0.710 | 0.268 | +0.442 |
| ROUGE-L | 0.647 | 0.191 | +0.456 |
| BERTScore F1 | 0.906 | 0.775 | +0.131 |
| Faithfulness | 0.980 | — | — |

---

## Citation

If you use this code or dataset, please cite:

```bibtex
@software{molhi2026railwayrag,
  author    = {Majed Molhi},
  title     = {RailRAG: RAG Benchmark for Railway Technical Documents},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19047213},
  url       = {https://doi.org/10.5281/zenodo.19047213}
}
```

---

## License

This project is licensed under the MIT License. The gold benchmark dataset (`data/gold_QA.csv`) is released under CC BY 4.0.

---

## Acknowledgements

Source documents are official publications from RSSB, FRA, Network Rail, ERA, SCRRA, and JICA. Full URLs available in `data/Metadata.csv`.
