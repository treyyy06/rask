import os
import json
from typing import Dict, Any, List
from ..config import config
from ..logging_config import get_logger
from ..ingestion.ingester import DocumentIngester
from ..parsing.parser import DocumentParser
from ..chunking.chunker import ModalityAwareChunker
from ..embeddings.embedder import DocumentEmbedder
from ..retrieval.faiss_index import FaissVectorIndex
from ..aqu.pipeline import AQUQueryEngine
from ..retrieval.retriever import ModalityAwareRetriever
from ..retrieval.reranker import CrossEncoderReranker
from ..validation.evidence_validator import EvidenceValidator
from ..validation.answer_validator import AnswerValidator

# Answering strategies
from .text_answerer import TextAnswerer
from .table_answerer import TableAnswerer
from .visual_answerer import VisualAnswerer
from .multimodal_answerer import MultimodalAnswerer

logger = get_logger("answering.qa_pipeline")

class MultimodalQAPipeline:
    def __init__(self, use_reranker: bool = True, use_aqu: bool = True):
        self.use_reranker = use_reranker
        self.use_aqu = use_aqu

        # Instantiate pipeline components
        self.ingester = DocumentIngester()
        self.parser = DocumentParser()
        self.chunker = ModalityAwareChunker()
        self.embedder = DocumentEmbedder()
        self.faiss_index = FaissVectorIndex()
        
        self.aqu_engine = AQUQueryEngine()
        self.retriever = ModalityAwareRetriever(self.faiss_index, self.embedder)
        self.reranker = CrossEncoderReranker()
        
        self.evidence_validator = EvidenceValidator()
        self.answer_validator = AnswerValidator()

        # Answerers
        self.text_answerer = TextAnswerer()
        self.table_answerer = TableAnswerer()
        self.visual_answerer = VisualAnswerer()
        self.multimodal_answerer = MultimodalAnswerer()

    def process_raw_documents(self, force: bool = False):
        """Orchestrates document rendering, parsing, chunking, embedding, and indexing."""
        logger.info("Starting raw document processing stage...")
        
        # 1. Ingestion / Rendering
        ingest_status = self.ingester.ingest_all(force=force)
        
        # 2. Parsing (OCR + Layout + Tables)
        parse_status = self.parser.parse_all(ingest_status, force=force)
        
        # 3. Chunking
        chunk_status = self.chunker.chunk_all(parse_status, force=force)
        
        # 4. Embeddings & FAISS Indexing
        if force:
            self.faiss_index.clear()

        for doc_id, c_info in chunk_status.items():
            if c_info["status"] != "COMPLETED":
                continue
                
            chunks_path = config.CHUNKS_DIR / f"{doc_id}_chunks.jsonl"
            if not chunks_path.exists():
                continue
                
            # Load chunks
            chunks = []
            with open(chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        chunks.append(json.loads(line))
                        
            # Generate embeddings (resumable)
            embeddings_map = self.embedder.embed_chunks(doc_id, chunks, force=force)
            
            # Add to index
            self.faiss_index.add_embeddings(chunks, embeddings_map)
            
        logger.info("Raw document processing completed successfully.")

    def run_qa(self, question: str, debug: bool = False) -> Dict[str, Any]:
        """
        Executes query understanding, retrieval, validation, and answer generation.
        """
        logger.info(f"--- Pipeline processing query: '{question}' ---")

        # 1. AQU Query Understanding (Ablation support: fallback to standard classification if disabled)
        if self.use_aqu:
            aqu_analysis = self.aqu_engine.analyze_query(question)
        else:
            # Baseline dense RAG query analysis mock
            aqu_analysis = {
                "question": question,
                "raw_aspects": [],
                "aspects": [],
                "predicted_modality": "Text",
                "modality_confidence": 0.50,
                "query_confidence": 0.50
            }

        # 2. Candidate Retrieval
        candidates = self.retriever.retrieve(aqu_analysis)
        if not candidates:
            return self._build_refusal_response(question, "No context chunks retrieved.")

        # 3. Cross-Encoder Reranking
        if self.use_reranker:
            selected_evidence = self.reranker.rerank(question, candidates)
        else:
            # Baseline dense candidate selection
            top_k = min(config.TOP_K, len(candidates))
            selected_evidence = candidates[:top_k]

        # 4. Evidence Validation
        is_valid, validation_reason = self.evidence_validator.validate(aqu_analysis, selected_evidence)
        if not is_valid:
            logger.warning(f"Evidence validation rejected: {validation_reason}")
            # We enforce grounding by falling back to refusal
            return self._build_refusal_response(question, f"Grounded evidence validation failed: {validation_reason}")

        # 5. Answer Generation Strategy Selection
        pred_mod = aqu_analysis["predicted_modality"]
        
        logger.info(f"Invoking answer strategy: {pred_mod}")
        if pred_mod == "Table":
            answer = self.table_answerer.answer(question, selected_evidence)
        elif pred_mod in ["Figure", "Chart"]:
            answer = self.visual_answerer.answer(question, selected_evidence)
        elif pred_mod == "Mixed":
            answer = self.multimodal_answerer.answer(question, selected_evidence)
        else:
            answer = self.text_answerer.answer(question, selected_evidence)

        # 6. Post-generation Answer Verification
        evidence_text = "\n\n".join(e["text"] for e in selected_evidence)
        validation_status, verification_reason = self.answer_validator.verify_consistency(
            question, evidence_text, answer
        )
        
        # If verification fails, default to structured refusal rather than returning hallucination
        if validation_status == "UNSUPPORTED":
            logger.error(f"Factual consistency check failed: {verification_reason}")
            return self._build_refusal_response(question, f"Factual validation failed: {verification_reason}")

        # 7. Package structured response
        source_pages = sorted(list(set(e["page"] for e in selected_evidence)))
        chunk_ids = [e["chunk_id"] for e in selected_evidence]
        modalities = list(set(e["modality"] for e in selected_evidence))

        confidence_str = "HIGH" if aqu_analysis["query_confidence"] >= 0.75 else ("MEDIUM" if aqu_analysis["query_confidence"] >= 0.50 else "LOW")

        result = {
            "answer": answer,
            "confidence": confidence_str,
            "doc_id": selected_evidence[0]["doc_id"],
            "pages": source_pages,
            "source_pages": source_pages, # duplicate key for backward compatibility
            "chunk_ids": chunk_ids,
            "modalities": modalities,
            "validation": validation_status.lower(),
            "evidence": {
                "text": "Text" in modalities,
                "table": "Table" in modalities,
                "image": any(m in modalities for m in ["Figure", "Chart"])
            }
        }

        if debug:
            result["debug_trace"] = {
                "classification": {
                    "text": pred_mod == "Text",
                    "table": pred_mod == "Table",
                    "image": pred_mod in ["Figure", "Chart"],
                    "sequence": "sequence" in question.lower(),
                    "spatial": "spatial" in question.lower(),
                    "cross_modal": pred_mod == "Mixed"
                },
                "query_analysis": aqu_analysis,
                "retrieved_results": [
                    {
                        "page": c["page"],
                        "modality": c["modality"],
                        "score": c.get("retrieval_score", 0.0),
                        "rerank_score": c.get("rerank_score", 0.0)
                    } for c in candidates[:10]
                ],
                "validation_reason": verification_reason
            }

        return result

    def _build_refusal_response(self, question: str, reason: str) -> Dict[str, Any]:
        return {
            "answer": "Insufficient document evidence to answer reliably.",
            "confidence": "LOW",
            "doc_id": "unknown",
            "pages": [],
            "source_pages": [],
            "chunk_ids": [],
            "modalities": [],
            "validation": "unsupported",
            "evidence": {"text": False, "table": False, "image": False},
            "debug_trace": {
                "validation_reason": reason
            }
        }
