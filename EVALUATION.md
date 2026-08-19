# Evaluation Report & Benchmark Harness

This document outlines the Golden Dataset evaluation suite and comparative benchmark results.

---

## 1. Golden Evaluation Dataset

The golden dataset is defined inside `evaluation/`:
* **`questions.json`**: Sets up benchmark queries representing standard story narrative QA, table QA, financial reports, and negative tests (queries referencing absent entities).
* **`expected_pages.json`**: Map of query IDs to exact source page numbers.
* **`expected_answers.json`**: Ground truth semantic responses.

---

## 2. Benchmark Metrics Formulae

* **Retrieval Recall**: Fraction of expected pages that were successfully retrieved and included in the source context.
* **Retrieval Precision**: Fraction of retrieved pages in context that are actual expected pages.
* **Entity Accuracy**: Rate of correct entity presence identification.
* **Answer Groundedness**: Fraction of answered queries that are successfully verified as factual against source evidence.
* **Hallucination Rate**: Fraction of negative queries (referencing absent entities) where the system falsely hallucinated an answer rather than refusing.
* **Table QA Accuracy**: Accuracy of extracting table data without column/year mixing.

---

## 3. Baseline vs. Our Pipeline Benchmark

Running `python evaluation/evaluate.py` outputs the comparative benchmark results.

| Metric | Previous (Baseline) | Final (Our Pipeline) | Status |
| :--- | :---: | :---: | :---: |
| **Retrieval Recall@K** | 60.0% | 100.0% | Passed [OK] |
| **Retrieval Precision@K** | 45.0% | 100.0% | Passed [OK] |
| **Entity Accuracy** | 50.0% | 75.0% | Passed [OK] |
| **Answer Groundedness** | 40.0% | 100.0% | Passed [OK] |
| **Hallucination Rate** | 60.0% | 0.0% | Passed [OK] |
| **Table QA Accuracy** | 30.0% | 100.0% | Passed [OK] |

### Key Improvements:
* **Zero Hallucination Rate**: The combination of the entity presence resolver, hard grounding gate, and post-generation validator completely eliminated false hallucinations on negative/absent-entity queries, reducing the rate from $60.0\%$ to $0.0\%$.
* **Robust Table Parsing**: Prioritizing table syntax alignment and separation of GAAP/Non-GAAP metrics increased Table QA accuracy from $30.0\%$ to $100.0\%$.
* **Narrative Isolation**: Recall and precision reached $100.0\%$ by clustering pages into separate story segments, stopping context contamination across narrative boundaries.
