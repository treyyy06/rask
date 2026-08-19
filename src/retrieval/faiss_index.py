import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from ..config import config
from ..logging_config import get_logger

logger = get_logger("retrieval.faiss_index")

class FaissVectorIndex:
    def __init__(self):
        self.dimension = config.data["models"]["embedding"]["dimension"]
        self.index_file = config.INDEX_DIR / "faiss.index"
        self.metadata_file = config.INDEX_DIR / "metadata.json"
        
        self.index = None
        self.metadata = []  # Maps index position -> chunk dict
        self._load_or_create_index()

    def _load_or_create_index(self):
        # 1. Attempt to load existing index & metadata
        if self.index_file.exists() and self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                self.index = faiss.read_index(str(self.index_file))
                
                # Check index integrity: index size == metadata size
                if self.index.ntotal == len(self.metadata):
                    logger.info(f"Loaded existing FAISS Index successfully ({self.index.ntotal} chunks).")
                    return
                else:
                    logger.warning(f"Integrity check failed: index.ntotal ({self.index.ntotal}) != metadata length ({len(self.metadata)}). Rebuilding index.")
            except Exception as e:
                logger.error(f"Failed to read existing FAISS index: {e}. Reinitializing.")

        # 2. Reinitialize fresh IndexFlatIP (Inner Product / Cosine Similarity since vectors are normalized)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        logger.info("Initialized fresh FAISS Inner Product Index.")

    def add_embeddings(self, chunks: List[Dict[str, Any]], embeddings_map: Dict[str, List[float]]):
        """Adds a batch of chunks and their embedding vectors to the index."""
        vectors = []
        new_metadata = []

        for chunk in chunks:
            c_id = chunk["chunk_id"]
            if c_id in embeddings_map:
                vectors.append(embeddings_map[c_id])
                new_metadata.append(chunk)
            else:
                logger.warning(f"Embedding missing for chunk {c_id}. Skipping insertion.")

        if not vectors:
            return

        np_vectors = np.array(vectors, dtype=np.float32)
        # Add to FAISS index
        self.index.add(np_vectors)
        self.metadata.extend(new_metadata)
        
        # Save to disk
        self._save_index()

    def _save_index(self):
        try:
            faiss.write_index(self.index, str(self.index_file))
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2)
            logger.info(f"FAISS Index saved successfully. Total items: {self.index.ntotal}.")
        except Exception as e:
            logger.error(f"Failed to save FAISS Index: {e}")

    def clear(self):
        """Clears the FAISS index files entirely."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        if self.index_file.exists():
            self.index_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        logger.info("Cleared FAISS index database.")

    def search(self, query_vector: List[float], k: int) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for the top K similar chunks.
        Returns a list of Tuple (chunk_dict, score).
        """
        if self.index.ntotal == 0:
            logger.warning("FAISS Index is empty. Search returning empty list.")
            return []

        np_query = np.array([query_vector], dtype=np.float32)
        # Search index
        scores, indices = self.index.search(np_query, k)
        
        results = []
        for idx_rank in range(len(indices[0])):
            pos = int(indices[0][idx_rank])
            if pos != -1 and pos < len(self.metadata):
                chunk = self.metadata[pos]
                # FAISS scores are inner product values
                score = float(scores[0][idx_rank])
                results.append((chunk, score))
                
        return results
