# Architecture Design Document

This document provides a detailed breakdown of the technical components and pipelines implemented in OmniRAG.

---

## 1. Document Ingestion, OCR & Table Parsing

The ingestion pipeline handles documents in a robust, document-agnostic manner:
* **pypdfium2 Renderer**: Renders high-resolution page images. Images exceeding 1024 pixels are scaled down to fit constraints.
* **OCR Quality Assessor**: Scores page text quality ($0.0$ to $1.0$) using regular expressions to detect garbage repetitions, symbol density, and token fragmentation.
* **VLM/Image Fallback**: Pages containing low text density or corrupted OCR are flagged (`use_visual_fallback = True`) for direct visual analysis by the VLM.
* **pdfplumber Parser**: Extracts structured tables, formatting them into Markdown representing header/column alignment. GAAP and Non-GAAP keywords are mapped and separated.

---

## 2. Story Continuity Segmentation

To solve narratively clustered documents (like comic books or multi-topic reports), we construct sequential clusters:
* **Boundary Split Check**: Computes lexical transitions between adjacent pages `i` and `i+1` based on noun extraction, Jaccard text overlap, and explicit headers (e.g. Chapter numbers).
* **Narrative Isolation**: Adjacent pages with matching chapter headings or high entity/text overlaps are clustered together. If a split boundary is detected (heading transition or low overlap), a new cluster is created.
* **Retrieved Evidence Clustering**: When a query is mapped to a specific entity or page, the system retrieves only context pages belonging to the same cluster, rejecting unrelated stories.

---

## 3. Modality & Sequence Classification

OmniRAG dynamically detects query intents:
* **Modality Tagging**: Categorizes queries into TEXT, TABLE, IMAGE, SEQUENCE, SPATIAL, or CROSS_MODAL.
* **Chronological Sequence Sorting**: Sequence-oriented queries automatically sort retrieved page visual assets in chronological ascending order (e.g., Page 1 -> Page 2 -> Page 3...) rather than arbitrary similarity score ranking.

---

## 4. Entity-Anchored Hybrid Retrieval

We employ a unified, multi-factor evidence scorer:
* **Entity Match ($0.35$)**: Evaluates query entity keyword presence in text index.
* **Text Similarity ($0.25$)**: Cosine similarity of Gemini embeddings (`models/text-embedding-004`) or fallback Jaccard word similarity.
* **Story Continuity ($0.20$)**: High-weight boost for pages in the best narrative cluster matching the query.
* **Visual Similarity ($0.10$)**: Evaluates page visual relevance for image queries.
* **Modality Match ($0.10$)**: Prioritizes pages with tables for tabular queries, and image pages for visual queries.

$$Score = 0.35 \times Entity + 0.25 \times Text + 0.20 \times Story + 0.10 \times Visual + 0.10 \times Modality$$

---

## 5. Grounding & Factual Consistency Loop

To eliminate hallucinations, the generation phase runs inside a closed loop:
* **Hard Grounding Gate**: Refuses to answer if the matching target entity resolved as `"ABSENT"` or if zero modal matches are present.
* **Factual Validator**: Runs regex and VLM validation. Confirms that proper nouns, digits, and causal assertions in the generated answer are strictly supported by the evidence.
* **Retry Loop**: If an answer is marked `UNSUPPORTED`, it retries generation (up to 2 times) with a stricter correction prompt. If still unsupported, it falls back to a grounded refusal response: *"Insufficient document evidence to answer reliably."*
