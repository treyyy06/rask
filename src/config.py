import os
import yaml
from pathlib import Path
from typing import Dict, Any

# Root Workspace Directories
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = WORKSPACE_DIR / "configs" / "config.yaml"

# Default config fallback
DEFAULT_CONFIG = {
    "models": {
        "embedding": {"name": "BAAI/bge-small-en-v1.5", "dimension": 384},
        "reranker": {"name": "cross-encoder/ms-marco-MiniLM-L-6-v2"},
        "vlm": {"name": "gemini-1.5-flash"},
        "query_model": {"name": "gemini-1.5-flash"}
    },
    "retrieval": {
        "candidate_k": 50,
        "evidence_k": 5,
        "similarity_threshold": 0.35
    },
    "modality": {
        "high_confidence": 0.80,
        "medium_confidence": 0.55
    },
    "generation": {
        "max_new_tokens": 128,
        "temperature": 0.0
    }
}

class Config:
    def __init__(self):
        self.data = DEFAULT_CONFIG.copy()
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    user_data = yaml.safe_load(f)
                    if user_data:
                        self.deep_update(self.data, user_data)
            except Exception as e:
                print(f"Error loading configs/config.yaml: {e}. Using defaults.")

        # Resolve persistent data folders
        self.RAW_DIR = WORKSPACE_DIR / "data" / "raw"
        self.RENDERED_DIR = WORKSPACE_DIR / "data" / "rendered"
        self.PARSED_DIR = WORKSPACE_DIR / "data" / "parsed"
        self.CHUNKS_DIR = WORKSPACE_DIR / "data" / "chunks"
        self.EMBEDDINGS_DIR = WORKSPACE_DIR / "data" / "embeddings"
        self.INDEX_DIR = WORKSPACE_DIR / "data" / "index"
        self.EVAL_DIR = WORKSPACE_DIR / "data" / "evaluation"
        
        # Ensure all data subfolders exist
        for d in [self.RAW_DIR, self.RENDERED_DIR, self.PARSED_DIR, self.CHUNKS_DIR, self.EMBEDDINGS_DIR, self.INDEX_DIR, self.EVAL_DIR]:
            d.mkdir(parents=True, exist_ok=True)

        # Environment API keys
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
        self.VLM_MODEL = os.getenv("VLM_MODEL", self.data["models"]["vlm"]["name"])
        self.TOP_K = int(os.getenv("TOP_K", self.data["retrieval"]["evidence_k"]))
        self.CANDIDATE_K = self.data["retrieval"]["candidate_k"]
        self.SIMILARITY_THRESHOLD = self.data["retrieval"]["similarity_threshold"]
        
    def deep_update(self, base: Dict[str, Any], update: Dict[str, Any]):
        for k, v in update.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                self.deep_update(base[k], v)
            else:
                base[k] = v

config = Config()
