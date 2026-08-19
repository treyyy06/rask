import json
import re
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Optional
from ..config import config
from ..logging_config import get_logger

logger = get_logger("chunking.chunker")

class ModalityAwareChunker:
    def __init__(self):
        self.crops_dir = config.CHUNKS_DIR / "crops"
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.status_file = config.CHUNKS_DIR / "status.json"
        self.status_db = self._load_status_db()

    def _load_status_db(self) -> Dict[str, Any]:
        if self.status_file.exists():
            try:
                with open(self.status_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to read chunker status: {e}. Reinitializing.")
        return {}

    def _save_status_db(self):
        try:
            with open(self.status_file, "w", encoding="utf-8") as f:
                json.dump(self.status_db, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save chunker status: {e}")

    def chunk_all(self, parse_status: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        """Runs chunking for all parsed documents."""
        results = {}
        for doc_id, doc_info in parse_status.items():
            if doc_info["status"] != "COMPLETED":
                continue
            
            results[doc_id] = self.chunk_document(doc_id, force=force)
        return results

    def chunk_document(self, doc_id: str, force: bool = False) -> Dict[str, Any]:
        """Creates clean, modality-specific chunks for a single parsed document."""
        status_info = self.status_db.get(doc_id, {
            "doc_id": doc_id,
            "status": "PENDING",
            "chunk_count": 0
        })

        parsed_file_path = config.PARSED_DIR / f"{doc_id}.json"
        chunks_file_path = config.CHUNKS_DIR / f"{doc_id}_chunks.jsonl"

        if not force and status_info["status"] == "COMPLETED" and chunks_file_path.exists():
            # Validate integrity
            try:
                with open(chunks_file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    if len(lines) == status_info["chunk_count"]:
                        logger.info(f"Chunks for {doc_id} already exist. Skipping.")
                        return status_info
            except Exception:
                pass

        if not parsed_file_path.exists():
            logger.error(f"Parsed JSON file missing for {doc_id}.")
            return status_info

        try:
            with open(parsed_file_path, "r", encoding="utf-8") as f:
                parsed_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load parsed JSON for {doc_id}: {e}")
            return status_info

        logger.info(f"Chunking document: {doc_id}...")
        chunks = []
        chunk_idx = 0

        for page_data in parsed_data.get("pages", []):
            page_num = page_data["page"]
            page_image_path = page_data["image_path"]
            
            # --- 1. TABLE CHUNKS ---
            # Create one chunk per structural table
            for t_idx, tbl in enumerate(page_data.get("tables", [])):
                chunk_id = f"{doc_id}_p{page_num}_tbl{t_idx}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "page": page_num,
                    "modality": "Table",
                    "text": f"Table showing data: {tbl['markdown']}",
                    "image_path": page_image_path,
                    "bbox": tbl["bbox"],
                    "metadata": {
                        "headers": tbl["headers"],
                        "rows": tbl["rows"],
                        "markdown": tbl["markdown"]
                    }
                })
                chunk_idx += 1

            # --- 2. VISUAL CHUNKS (FIGURES & CHARTS) ---
            # Parse layout regions to isolate figures or charts
            fig_idx = 0
            for reg in page_data.get("layout_regions", []):
                mod = reg["modality"]
                if mod in ["Figure", "Chart"]:
                    chunk_id = f"{doc_id}_p{page_num}_{mod.lower()}{fig_idx}"
                    bbox = reg["bbox"]
                    
                    # Crop visual region and save to disk
                    crop_path = self._crop_image_region(page_image_path, bbox, doc_id, page_num, mod, fig_idx)
                    
                    # Group OCR text that falls within the cropped region
                    crop_ocr_text = self._get_ocr_overlapping_bbox(page_data.get("ocr_lines", []), bbox)
                    
                    chunks.append({
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "page": page_num,
                        "modality": mod,
                        "text": f"{mod} image region. Extracted text caption: {crop_ocr_text}",
                        "image_path": page_image_path,
                        "bbox": bbox,
                        "metadata": {
                            "crop_path": str(crop_path) if crop_path else None,
                            "crop_text": crop_ocr_text
                        }
                    })
                    fig_idx += 1
                    chunk_idx += 1

            # --- 3. TEXT CHUNKS ---
            # Group plain text lines into paragraph blocks (~150 words)
            # Make sure we don't include lines belonging to tables
            table_bboxes = [tbl["bbox"] for tbl in page_data.get("tables", [])]
            text_lines = []
            
            for line in page_data.get("ocr_lines", []):
                # Skip if the line falls inside a table bbox to prevent duplication
                if self._is_bbox_inside_any(line["bbox"], table_bboxes):
                    continue
                text_lines.append(line["text"])

            if text_lines:
                # Group text into paragraphs
                paragraphs = self._group_lines_into_paragraphs(text_lines)
                for p_idx, para in enumerate(paragraphs):
                    chunk_id = f"{doc_id}_p{page_num}_txt{p_idx}"
                    chunks.append({
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "page": page_num,
                        "modality": "Text",
                        "text": para,
                        "image_path": page_image_path,
                        "bbox": [0, 0, 0, 0],  # covers entire page text
                        "metadata": {}
                    })
                    chunk_idx += 1

        # Write to JSONL
        with open(chunks_file_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + "\n")

        status_info["status"] = "COMPLETED"
        status_info["chunk_count"] = len(chunks)
        self.status_db[doc_id] = status_info
        self._save_status_db()
        
        logger.info(f"Chunking COMPLETED for {doc_id} (created {len(chunks)} chunks).")
        return status_info

    def _crop_image_region(self, page_image_path: str, bbox: List[float], doc_id: str, page: int, modality: str, idx: int) -> Optional[Path]:
        if not page_image_path or not os.path.exists(page_image_path):
            return None
        try:
            img = Image.open(page_image_path)
            # Crop using bounding box [x_min, y_min, x_max, y_max]
            # Convert to integers and validate margins
            w, h = img.size
            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(w, int(bbox[2]))
            y2 = min(h, int(bbox[3]))
            
            if x2 <= x1 or y2 <= y1:
                return None
                
            crop_img = img.crop((x1, y1, x2, y2))
            crop_filename = f"{doc_id}_p{page}_{modality.lower()}_{idx}.png"
            crop_dir = self.crops_dir / doc_id
            crop_dir.mkdir(parents=True, exist_ok=True)
            crop_path = crop_dir / crop_filename
            crop_img.save(crop_path, "PNG")
            return crop_path
        except Exception as e:
            logger.error(f"Failed to crop image region: {e}")
            return None

    def _get_ocr_overlapping_bbox(self, ocr_lines: List[Dict[str, Any]], bbox: List[float]) -> str:
        overlapping_texts = []
        for line in ocr_lines:
            l_bbox = line["bbox"]
            # Check overlap logic (simple intersection)
            if (l_bbox[0] >= bbox[0] - 10 and l_bbox[2] <= bbox[2] + 10 and
                l_bbox[1] >= bbox[1] - 10 and l_bbox[3] <= bbox[3] + 10):
                overlapping_texts.append(line["text"])
        return " ".join(overlapping_texts).strip()

    def _is_bbox_inside_any(self, bbox: List[float], parent_bboxes: List[List[float]]) -> bool:
        for p_bbox in parent_bboxes:
            if (bbox[0] >= p_bbox[0] - 5 and bbox[2] <= p_bbox[2] + 5 and
                bbox[1] >= p_bbox[1] - 5 and bbox[3] <= p_bbox[3] + 5):
                return True
        return False

    def _group_lines_into_paragraphs(self, lines: List[str], max_words: int = 150) -> List[str]:
        paragraphs = []
        current_chunk = []
        current_word_count = 0
        
        for line in lines:
            line_words = line.split()
            if not line_words:
                continue
                
            current_chunk.append(line)
            current_word_count += len(line_words)
            
            if current_word_count >= max_words:
                paragraphs.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0
                
        if current_chunk:
            paragraphs.append(" ".join(current_chunk))
            
        return paragraphs
