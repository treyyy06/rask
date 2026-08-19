import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SCRATCH_DIR = BASE_DIR / "scratch"
DEFAULT_SCRATCH_DIR.mkdir(exist_ok=True)

# Configuration Variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Pipeline Tuning Parameters
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "4"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))

# Model Selections
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
# We use gemini-1.5-flash as default VLM/LLM for fast and robust responses
VLM_MODEL = os.getenv("VLM_MODEL", "gemini-1.5-flash")

# Ingestion Parameters
MAX_PAGE_CHARS = int(os.getenv("MAX_PAGE_CHARS", "8000"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "1024"))

# Logging & Storage
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
SCRATCH_DIR = Path(os.getenv("SCRATCH_DIR", str(DEFAULT_SCRATCH_DIR)))
SCRATCH_DIR.mkdir(exist_ok=True)

# Schema/Version Tracking (Verify source-mismatch protection)
QA_VERSION = "1.0.0"
INDEX_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0.0"
