import re
from typing import List, Dict, Any
from ..config import config
from ..logging_config import get_logger

logger = get_logger("retrieval.reranker")

class CrossEncoderReranker:
    def __init__(self):
        self.model_name = config.data["models"]["reranker"]["name"]
        self.model = None
        self.evidence_k = config.TOP_K
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import CrossEncoder
            # Load cross encoder on CPU
            self.model = CrossEncoder(self.model_name, device="cpu")
            logger.info(f"Loaded CrossEncoder reranker: {self.model_name} on CPU.")
        except Exception as e:
            logger.warning(f"CrossEncoder failed to load: {e}. Fallback lexical reranker will be used.")

    def rerank(self, question: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reranks candidates using Cross-Encoder or fallback lexical score.
        Returns the top `evidence_k` chunks.
        """
        if not candidates:
            return []

        logger.info(f"Reranking {len(candidates)} candidates for question: '{question}'")

        # 1. Use Cross-Encoder model if available
        if self.model:
            try:
                pairs = [[question, chunk["text"]] for chunk in candidates]
                scores = self.model.predict(pairs)
                
                # Assign reranker scores
                for idx, score in enumerate(scores):
                    candidates[idx]["rerank_score"] = float(score)
            except Exception as e:
                logger.error(f"Cross-Encoder reranking failed: {e}. Falling back to lexical scoring.")
                self._apply_lexical_rerank(question, candidates)
        else:
            self._apply_lexical_rerank(question, candidates)

        # Sort by rerank score descending
        candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        
        # Select Top K evidence chunks
        top_k = min(self.evidence_k, len(candidates))
        selected_evidence = candidates[:top_k]
        
        for idx, item in enumerate(selected_evidence):
            score_val = item.get("rerank_score", 0.0)
            logger.info(f"Reranked Rank {idx+1}: Page {item['page']} ({item['modality']}) - Score: {score_val:.3f}")
            
        return selected_evidence

    def _apply_lexical_rerank(self, question: str, candidates: List[Dict[str, Any]]):
        """Lexical Jaccard/overlap fallback scoring for reranking on constrained systems."""
        q_words = set(re.findall(r'\b\w+(?:-\w+)*\b', question.lower()))
        for chunk in candidates:
            chunk_words = set(re.findall(r'\b\w+(?:-\w+)*\b', chunk["text"].lower()))
            
            if q_words and chunk_words:
                overlap = len(q_words.intersection(chunk_words))
                union = len(q_words.union(chunk_words))
                # Normalized lexical score
                score = overlap / union
            else:
                score = 0.0
                
            # Add small retrieval score weight to break ties
            score += 0.05 * chunk.get("retrieval_score", 0.0)
            chunk["rerank_score"] = float(score)
