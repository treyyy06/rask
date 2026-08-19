import hashlib
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from ..config import config
from ..logging_config import get_logger

logger = get_logger("embeddings.embedder")

class DocumentEmbedder:
    def __init__(self):
        self.model_name = config.data["models"]["embedding"]["name"]
        self.dimension = config.data["models"]["embedding"]["dimension"]
        self.model = None
        self.cache_dir = config.EMBEDDINGS_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            # Load sentence transformer on CPU
            self.model = SentenceTransformer(self.model_name, device="cpu")
            logger.info(f"Loaded embedding model: {self.model_name} on CPU.")
        except Exception as e:
            logger.error(f"Failed to load sentence-transformers model: {e}")

    def get_text_hash(self, text: str) -> str:
        """MD5 hash of a text string."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def embed_chunks(self, doc_id: str, chunks: List[Dict[str, Any]], force: bool = False) -> Dict[str, List[float]]:
        """
        Embeds a list of chunks using cached values if available and unchanged.
        Returns a mapping of chunk_id to embedding vector.
        """
        embeddings_cache_path = self.cache_dir / f"{doc_id}_embeddings.json"
        
        # Load cache if available
        cache = {}
        if not force and embeddings_cache_path.exists():
            try:
                with open(embeddings_cache_path, "r", encoding="utf-8") as f:
                    cache = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load embedding cache for {doc_id}: {e}")

        # Check hashes and identify missing embeddings
        chunk_embeddings = {}
        missing_texts = []
        missing_chunk_ids = []
        
        for chunk in chunks:
            c_id = chunk["chunk_id"]
            text = chunk["text"]
            t_hash = self.get_text_hash(text)
            
            # Check if cached matches the current hash and has correct length
            if c_id in cache and cache[c_id].get("hash") == t_hash and len(cache[c_id].get("vector", [])) == self.dimension:
                chunk_embeddings[c_id] = cache[c_id]["vector"]
            else:
                missing_texts.append(text)
                missing_chunk_ids.append(c_id)

        # Batch encode missing texts
        if missing_texts:
            if not self.model:
                logger.error("No embedding model loaded. Cannot generate new embeddings.")
                raise RuntimeError("Embedding model is unavailable.")
                
            logger.info(f"[{doc_id}] Embedding {len(missing_texts)} new/modified chunks...")
            try:
                # Normalize embeddings to enable direct Cosine/Inner Product comparison in FAISS
                embeddings = self.model.encode(missing_texts, normalize_embeddings=True, show_progress_bar=False)
                
                for idx, c_id in enumerate(missing_chunk_ids):
                    text = missing_texts[idx]
                    t_hash = self.get_text_hash(text)
                    vector = [float(val) for val in embeddings[idx]]
                    
                    chunk_embeddings[c_id] = vector
                    cache[c_id] = {
                        "hash": t_hash,
                        "vector": vector
                    }
            except Exception as e:
                logger.error(f"Batch embedding generation failed: {e}")
                raise e

            # Save updated cache file
            try:
                with open(embeddings_cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to write embedding cache: {e}")

        return chunk_embeddings
