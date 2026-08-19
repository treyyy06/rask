import os
import json
import hashlib
import pypdfium2 as pdfium
from PIL import Image
from pathlib import Path
from typing import Dict, Any, List, Optional
from ..config import config
from ..logging_config import get_logger

logger = get_logger("ingestion")

def get_file_hash(file_path: Path) -> str:
    """Generate MD5 hash of a file to uniquely identify it."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

class DocumentIngester:
    def __init__(self):
        self.status_file = config.RENDERED_DIR / "status.json"
        self.status_db = self._load_status_db()

    def _load_status_db(self) -> Dict[str, Any]:
        if self.status_file.exists():
            try:
                with open(self.status_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to read status database: {e}. Reinitializing.")
        return {}

    def _save_status_db(self):
        try:
            with open(self.status_file, "w", encoding="utf-8") as f:
                json.dump(self.status_db, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save status database: {e}")

    def get_doc_id(self, pdf_path: Path) -> str:
        """Standard doc_id using hash combined with filename stem."""
        file_hash = get_file_hash(pdf_path)
        return f"{pdf_path.stem}_{file_hash[:8]}"

    def ingest_all(self, force: bool = False) -> Dict[str, Any]:
        """Scan data/raw/ and render all unrendered or partially rendered PDFs."""
        raw_files = list(config.RAW_DIR.glob("*.pdf"))
        if not raw_files:
            logger.info("No raw PDF documents found in data/raw/.")
            return {}

        results = {}
        for pdf_path in raw_files:
            doc_id = self.get_doc_id(pdf_path)
            results[doc_id] = self.ingest_document(pdf_path, force=force)
        return results

    def ingest_document(self, pdf_path: Path, force: bool = False) -> Dict[str, Any]:
        """Ingest a single document, rendering its pages and validating completeness."""
        pdf_path = Path(pdf_path)
        doc_id = self.get_doc_id(pdf_path)
        file_hash = get_file_hash(pdf_path)
        doc_render_dir = config.RENDERED_DIR / doc_id
        doc_render_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Ingesting PDF: {pdf_path.name} (doc_id: {doc_id})")

        # Load existing status
        status_info = self.status_db.get(doc_id, {
            "doc_id": doc_id,
            "filename": pdf_path.name,
            "status": "PENDING",
            "file_hash": file_hash,
            "num_pages": 0,
            "rendered_pages": []
        })

        # Check if already completed and hash matches
        if not force and status_info["status"] == "COMPLETED" and status_info["file_hash"] == file_hash:
            # Double check that images actually exist and are non-empty
            actual_pages = list(doc_render_dir.glob("page_*.png"))
            if len(actual_pages) == status_info["num_pages"] and all(p.stat().st_size > 0 for p in actual_pages):
                logger.info(f"Document {doc_id} already successfully ingested (COMPLETED). Skipping.")
                return status_info
            else:
                logger.warning(f"Status marked COMPLETED for {doc_id} but files are missing or empty. Re-processing.")
                status_info["status"] = "PARTIAL"

        # Initialize or update document info
        try:
            pdf = pdfium.PdfDocument(str(pdf_path))
            num_pages = len(pdf)
            status_info["num_pages"] = num_pages
            status_info["file_hash"] = file_hash
        except Exception as e:
            logger.error(f"Failed to open PDF {pdf_path.name}: {e}")
            status_info["status"] = "FAILED"
            self.status_db[doc_id] = status_info
            self._save_status_db()
            return status_info

        status_info["status"] = "PROCESSING"
        self.status_db[doc_id] = status_info
        self._save_status_db()

        rendered_list = set(status_info.get("rendered_pages", []))
        if force:
            rendered_list = set()

        success_count = len(rendered_list)
        
        for p in range(1, num_pages + 1):
            if p in rendered_list:
                # Confirm image exists
                image_path = doc_render_dir / f"page_{p}.png"
                if image_path.exists() and image_path.stat().st_size > 0:
                    continue
                else:
                    rendered_list.remove(p)

            # Render page
            image_path = doc_render_dir / f"page_{p}.png"
            success = self._render_page(pdf, p, image_path)
            if success:
                rendered_list.add(p)
                status_info["rendered_pages"] = sorted(list(rendered_list))
                self.status_db[doc_id] = status_info
                self._save_status_db()
                logger.info(f"[{doc_id}] Rendered page {p}/{num_pages} successfully.")
            else:
                logger.error(f"[{doc_id}] Failed to render page {p}/{num_pages}.")
                status_info["status"] = "PARTIAL"
                self.status_db[doc_id] = status_info
                self._save_status_db()
                # Break to allow resumability later
                return status_info

        # Perform page count completeness validation
        actual_images = list(doc_render_dir.glob("page_*.png"))
        if len(actual_images) == num_pages and all(p.stat().st_size > 0 for p in actual_images):
            status_info["status"] = "COMPLETED"
            logger.info(f"Ingestion COMPLETED for {doc_id} (rendered all {num_pages} pages).")
        else:
            status_info["status"] = "FAILED"
            logger.error(f"Ingestion validation failed for {doc_id}: actual images ({len(actual_images)}) != expected pages ({num_pages})")

        self.status_db[doc_id] = status_info
        self._save_status_db()
        return status_info

    def _render_page(self, pdf: pdfium.PdfDocument, page_num: int, output_path: Path) -> bool:
        try:
            page = pdf[page_num - 1]
            # Render page at 150 DPI (scale=2.0)
            bitmap = page.render(scale=2.0)
            pil_img = bitmap.to_pil()
            
            # Constrain dimensions if too large
            w, h = pil_img.size
            max_side = 1024
            if max(w, h) > max_side:
                if w > h:
                    new_w = max_side
                    new_h = int(h * (max_side / w))
                else:
                    new_h = max_side
                    new_w = int(w * (max_side / h))
                pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
            pil_img.save(output_path, "PNG")
            return True
        except Exception as e:
            logger.error(f"Error rendering page {page_num}: {e}")
            return False
