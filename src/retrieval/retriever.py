from typing import List, Dict, Any, Tuple
from ..config import config
from ..logging_config import get_logger
from ..embeddings.embedder import DocumentEmbedder
from .faiss_index import FaissVectorIndex

logger = get_logger("retrieval.retriever")

class ModalityAwareRetriever:
    def __init__(self, faiss_index: FaissVectorIndex, embedder: DocumentEmbedder):
        self.index = faiss_index
        self.embedder = embedder
        
        # Load modality parameters
        self.high_conf_thresh = config.data["modality"]["high_confidence"]
        self.med_conf_thresh = config.data["modality"]["medium_confidence"]
        self.candidate_k = config.CANDIDATE_K

        # Define default modality budgets
        self.budgets = {
            "Table": {"Table": 0.70, "Text": 0.15, "Figure": 0.10, "Chart": 0.05},
            "Figure": {"Figure": 0.70, "Text": 0.15, "Chart": 0.10, "Table": 0.05},
            "Chart": {"Chart": 0.70, "Figure": 0.15, "Table": 0.10, "Text": 0.05},
            "Text": {"Text": 0.80, "Table": 0.10, "Figure": 0.05, "Chart": 0.05},
            "Mixed": {"Table": 0.35, "Figure": 0.30, "Text": 0.20, "Chart": 0.15}
        }

    def retrieve(self, aqu_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Executes modality-aware candidate allocation retrieval.
        """
        question = aqu_analysis["question"]
        pred_modality = aqu_analysis["predicted_modality"]
        mod_confidence = aqu_analysis["modality_confidence"]
        aspects = aqu_analysis["aspects"]
        
        logger.info(f"Retrieving candidate chunks for '{question}' (Predicted Mode: {pred_modality})")

        # 1. Determine modality allocation weights based on confidence levels
        weights = {}
        if mod_confidence >= self.high_conf_thresh:
            # High confidence: Strict allocation budget
            weights = self.budgets.get(pred_modality, self.budgets["Text"])
        elif mod_confidence >= self.med_conf_thresh:
            # Medium confidence: Blend the predicted budget with even distribution
            pred_weights = self.budgets.get(pred_modality, self.budgets["Text"])
            for m in ["Text", "Table", "Figure", "Chart"]:
                weights[m] = 0.60 * pred_weights.get(m, 0.25) + 0.40 * 0.25
        else:
            # Low confidence: Retrieve evenly from all modalities and let the reranker decide
            weights = {"Text": 0.25, "Table": 0.25, "Figure": 0.25, "Chart": 0.25}

        logger.info(f"Modality retrieval allocation weights: {weights}")

        # 2. Get embeddings for search queries (embed target question + key aspects)
        search_queries = [question]
        for asp in aspects[:3]:  # Limit to top 3 aspects
            search_queries.append(asp["aspect"])
            
        search_vectors = []
        for sq in search_queries:
            try:
                # Retrieve normalized query vector
                sq_vec = self.embedder.embed_chunks("query", [{"chunk_id": "q", "text": sq}], force=True)["q"]
                search_vectors.append(sq_vec)
            except Exception as e:
                logger.error(f"Failed to generate search vector for query '{sq}': {e}")
        
        if not search_vectors:
            return []

        # 3. Retrieve raw candidates from FAISS
        # We query FAISS for each search query vector, retrieving K candidates
        raw_candidates = []
        seen_chunk_ids = set()
        
        for vec in search_vectors:
            # Search large pool to filter by modality allocations
            hits = self.index.search(vec, k=100)
            for chunk, score in hits:
                c_id = chunk["chunk_id"]
                if c_id not in seen_chunk_ids:
                    seen_chunk_ids.add(c_id)
                    raw_candidates.append((chunk, score))

        # 4. Filter and allocate candidates based on modality weights
        allocated_candidates = []
        modality_pools = {"Text": [], "Table": [], "Figure": [], "Chart": []}
        
        # Sort raw candidate hits by score descending
        raw_candidates.sort(key=lambda x: x[1], reverse=True)
        
        for chunk, score in raw_candidates:
            mod = chunk["modality"]
            # Treat unknown layout modes as Text
            if mod not in modality_pools:
                mod = "Text"
            modality_pools[mod].append((chunk, score))

        # Retrieve slots budget
        total_slots = self.candidate_k
        slots_limit = {m: max(1, int(w * total_slots)) for m, w in weights.items()}
        
        for mod, limit in slots_limit.items():
            pool = modality_pools.get(mod, [])
            # Take up to the limit for this modality
            selected = pool[:limit]
            for chunk, score in selected:
                # Add score directly to chunk metadata representation
                chunk_copy = chunk.copy()
                chunk_copy["retrieval_score"] = float(score)
                allocated_candidates.append(chunk_copy)

        # Sort combined results by score descending
        allocated_candidates.sort(key=lambda x: x["retrieval_score"], reverse=True)
        
        logger.info(f"Retrieved {len(allocated_candidates)} chunks total across modality allocations.")
        return allocated_candidates
