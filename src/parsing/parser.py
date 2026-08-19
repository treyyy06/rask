import json
from pathlib import Path
from typing import Dict, Any, List
from ..config import config
from ..logging_config import get_logger
from .ocr import OCRProcessor
from .layout import LayoutDetector
from .tables import TableExtractor

logger = get_logger("parsing.parser")

class DocumentParser:
    def __init__(self):
        self.ocr_processor = OCRProcessor()
        self.layout_detector = LayoutDetector()
        self.table_extractor = TableExtractor()
        self.status_file = config.PARSED_DIR / "status.json"
        self.status_db = self._load_status_db()

    def _load_status_db(self) -> Dict[str, Any]:
        if self.status_file.exists():
            try:
                with open(self.status_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to read parser status database: {e}. Reinitializing.")
        return {}

    def _save_status_db(self):
        try:
            with open(self.status_file, "w", encoding="utf-8") as f:
                json.dump(self.status_db, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save parser status database: {e}")

    def parse_all(self, ingest_status: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        """Runs the parser pipeline for all ingested completed documents."""
        results = {}
        for doc_id, doc_info in ingest_status.items():
            if doc_info["status"] != "COMPLETED":
                logger.warning(f"Document {doc_id} has not finished ingestion status: {doc_info['status']}. Skipping parsing.")
                continue
            
            results[doc_id] = self.parse_document(doc_id, doc_info, force=force)
        return results

    def parse_document(self, doc_id: str, doc_info: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        """Parses a document page-by-page, combining layout blocks, tables, and OCR."""
        pdf_name = doc_info["filename"]
        pdf_path = config.RAW_DIR / pdf_name
        num_pages = doc_info["num_pages"]
        rendered_pages_dir = config.RENDERED_DIR / doc_id

        # Check existing parsed database status
        status_info = self.status_db.get(doc_id, {
            "doc_id": doc_id,
            "status": "PENDING",
            "num_pages": num_pages,
            "parsed_pages": []
        })

        parsed_file_path = config.PARSED_DIR / f"{doc_id}.json"
        
        # Check completeness
        if not force and status_info["status"] == "COMPLETED" and parsed_file_path.exists():
            # Validate JSON integrity
            try:
                with open(parsed_file_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    if len(cached_data.get("pages", [])) == num_pages:
                        logger.info(f"Document {doc_id} already fully parsed (COMPLETED). Skipping.")
                        return status_info
            except Exception:
                logger.warning(f"Parsed file for {doc_id} is corrupted. Re-parsing.")
                status_info["status"] = "PARTIAL"

        status_info["status"] = "PROCESSING"
        self.status_db[doc_id] = status_info
        self._save_status_db()

        # Load partial parse file if exists
        parsed_pages = {}
        if parsed_file_path.exists() and not force:
            try:
                with open(parsed_file_path, "r", encoding="utf-8") as f:
                    parsed_pages = {p["page"]: p for p in json.load(f).get("pages", [])}
            except Exception:
                pass

        import pypdfium2 as pdfium
        try:
            pdf = pdfium.PdfDocument(str(pdf_path))
        except Exception as e:
            logger.error(f"Failed to open PDF document {pdf_name} for native parsing: {e}")
            pdf = None

        logger.info(f"Parsing document: {pdf_name} ({num_pages} pages)")

        for p in range(1, num_pages + 1):
            if p in parsed_pages:
                continue

            page_image_path = rendered_pages_dir / f"page_{p}.png"
            if not page_image_path.exists():
                logger.error(f"Rendered image not found for Page {p} at {page_image_path}. Skipping page.")
                continue

            logger.info(f"[{doc_id}] Parsing Page {p}/{num_pages}...")
            # 1. OCR text lines
            ocr_lines = self.ocr_processor.run_ocr(page_image_path, p)
            
            # If OCR returned nothing, fallback to native PDF text
            if not ocr_lines and pdf:
                try:
                    pdf_page = pdf[p - 1]
                    native_text = pdf_page.get_textpage().get_text_bounded() or ""
                    if native_text.strip():
                        for idx, line in enumerate(native_text.split("\n")):
                            if line.strip():
                                ocr_lines.append({
                                    "page": p,
                                    "bbox": [0, 0, 0, 0],
                                    "text": line.strip(),
                                    "confidence": 1.0,
                                    "line_id": f"p{p}_native_l{idx}"
                                })
                        logger.info(f"[{doc_id}] Extracted {len(ocr_lines)} native text lines for Page {p} fallback.")
                except Exception as e:
                    logger.error(f"Native text extraction failed on Page {p}: {e}")

            # 2. Extract tables structurally
            tables = self.table_extractor.extract_tables_from_page(pdf_path, p)
            
            # 3. Layout region classification
            layout_regions = self.layout_detector.detect_regions(page_image_path, ocr_lines, p)

            # Reconstruct page text (plain string text representation)
            page_text = " ".join(line["text"] for line in ocr_lines)
            
            # Append tables markdown directly to the plain page text
            if tables:
                table_texts = [f"\n--- Extracted Table ---\n{t['markdown']}\n" for t in tables]
                page_text += "\n" + "\n".join(table_texts)

            parsed_pages[p] = {
                "page": p,
                "text": page_text,
                "ocr_lines": ocr_lines,
                "tables": tables,
                "layout_regions": layout_regions,
                "image_path": str(page_image_path)
            }
            
            # Update status incrementally
            status_info["parsed_pages"] = sorted(list(parsed_pages.keys()))
            self.status_db[doc_id] = status_info
            self._save_status_db()
            
            # Save incremental JSON output file
            output_data = {
                "doc_id": doc_id,
                "filename": pdf_name,
                "num_pages": num_pages,
                "pages": [parsed_pages[k] for k in sorted(parsed_pages.keys())]
            }
            with open(parsed_file_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2)

        # Final verification check
        if len(parsed_pages) == num_pages:
            status_info["status"] = "COMPLETED"
            logger.info(f"Parsing COMPLETED for {doc_id}.")
        else:
            status_info["status"] = "PARTIAL"
            logger.warning(f"Parsing partially completed for {doc_id}: {len(parsed_pages)}/{num_pages} parsed.")

        self.status_db[doc_id] = status_info
        self._save_status_db()
        return status_info
