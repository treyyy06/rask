# AQU-MR - Adaptive Query Understanding and Modality-Aware Retrieval for Multimodal Document QA

AQU-MR is a production-quality, hardware-aware, and modality-grounded Retrieval-Augmented Generation (RAG) system designed from scratch to run efficiently on CPU environments.

---

## 1. Core Architecture Flow

```
                      ┌──────────────────────┐
                      │     User Query       │
                      └──────────┬───────────展
                                 ↓
                      ┌──────────────────────┐
                      │   AQU QUERY ENGINE   │
                      │ 1. Normalization     │
                      │ 2. Aspect Decomp.    │
                      │ 3. Modality Predict  │
                      └──────────┬───────────┘
                                 ↓
                      ┌──────────────────────┐
                      │  Modality-aware      │
                      │  Candidate Retrieval │
                      └──────────┬───────────┘
                                 ↓
                      ┌──────────────────────┐
                      │  Stage 2 Reranker    │
                      └──────────┬───────────┘
                                 ↓
                      ┌──────────────────────┐
                      │  Evidence Validator  │
                      └──────────┬───────────┘
                                 ↓
                      ┌──────────────────────┐
                      │  Modality-specific   │
                      │  Answering Strategy  │
                      └──────────┬───────────┘
                                 ↓
                      ┌──────────────────────┐
                      │  Answer Verifier     │
                      └──────────┬───────────┘
                                 ↓
                            FINAL ANSWER
```

---

## 2. Project Directory Structure

```
project/
├── data/
│   ├── raw/           # Raw PDF documents
│   ├── rendered/      # Page PNG images and status.json
│   ├── parsed/        # OCR, table details, layout zones
│   ├── chunks/        # Modality-specific chunks and crops
│   ├── embeddings/    # Cached vector embeddings
│   ├── index/         # Persistent FAISS index files
│   └── evaluation/    # 80 evaluation questions and reports
│
├── configs/
│   └── config.yaml    # CPU-optimized models and parameters
│
├── src/
│   ├── ingestion/     # PDF rendering with completeness validation
│   ├── parsing/       # OCR (PaddleOCR), Layout blocks, pdfplumber Tables
│   ├── chunking/      # Modality-specific chunking
│   ├── embeddings/    # BGE-small vector embeddings generator
│   ├── retrieval/     # FAISS search + Cross-Encoder reranking
│   ├── aqu/           # Aspect decomposition and modality prediction
│   ├── answering/     # Text, structural Table, and Visual crop VLM answerers
│   ├── validation/    # Evidence coordinates and factual answer verifiers
│   └── evaluation/    # MRR, Recall@K, Exact Match, Token F1, ANLS
│
├── scripts/
│   ├── run_pipeline.py            # CLI Pipeline controller
│   └── generate_evaluation_data.py # Synthetic 80 queries generator
│
├── tests/
│   └── test_aqu_pipeline.py       # pytest suite
│
├── app.py             # Streamlit premium dark mode dashboard
├── requirements.txt   # CPU-optimized requirements
└── README.md          # Project guide
```

---

## 3. How to Run the Pipeline

First, make sure to copy any raw PDF documents to `data/raw/` (e.g. `tests/data/financial_report.pdf` or `tests/data/story_comic.pdf`).

### Stage 1: Document Processing Ingestion
Runs PDF rendering, OCR, layout detection, table extraction, chunking, embedding generation, and FAISS indexing:
```bash
python scripts/run_pipeline.py --stage index
```
*(Supports `--force` to re-process all pages and `--resume` to skip already processed files.)*

### Stage 2: Question Querying
Ask a question directly via the command line interface:
```bash
python scripts/run_pipeline.py --stage query --question "What was the GAAP Operating Income in 2024?"
```
*(Enable `--debug` to print details about aspects, modality predictions, candidate retrieval scores, and reranked metrics.)*

### Stage 3: Running the Evaluation Benchmark
Runs evaluation over the 80 DocVQA queries:
```bash
python scripts/run_pipeline.py --stage evaluate --limit 80
```
*(Supports `--ablation dense_only` and `--ablation dense_rerank` to run comparative baseline research experiments.)*

### Stage 4: Running the Streamlit UI
Start the premium dark mode web interface:
```bash
streamlit run app.py
```

---

## 4. Running the Tests
To run the automated validation test suite:
```bash
python -m pytest
```
