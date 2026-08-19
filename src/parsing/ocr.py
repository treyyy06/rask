import os
import re
from pathlib import Path
from typing import List, Dict, Any
from ..config import config
from ..logging_config import get_logger

logger = get_logger("parsing.ocr")

class OCRProcessor:
    def __init__(self):
        self.paddle_ocr = None
        self._init_paddle_ocr()

    def _init_paddle_ocr(self):
        try:
            # Attempt to import and initialize PaddleOCR dynamically
            from paddleocr import PaddleOCR
            self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            logger.info("PaddleOCR initialized successfully.")
        except Exception as e:
            logger.warning(f"PaddleOCR is unavailable: {e}. Falling back to pytesseract/native extraction.")

    def run_ocr(self, image_path: Path, page_num: int) -> List[Dict[str, Any]]:
        """
        Runs OCR on an image and returns a list of text regions with bounding boxes.
        Each region matches:
        {
            "page": int,
            "bbox": [x_min, y_min, x_max, y_max],
            "text": str,
            "confidence": float,
            "line_id": str
        }
        """
        image_path = Path(image_path)
        if not image_path.exists():
            logger.error(f"Image not found for OCR: {image_path}")
            return []

        # 1. Try PaddleOCR first
        if self.paddle_ocr:
            try:
                result = self.paddle_ocr.ocr(str(image_path), cls=True)
                if result and result[0]:
                    ocr_results = []
                    for idx, line in enumerate(result[0]):
                        bbox_pts, (text_str, conf) = line
                        # Convert 4-point polygon to [x_min, y_min, x_max, y_max]
                        xs = [pt[0] for pt in bbox_pts]
                        ys = [pt[1] for pt in bbox_pts]
                        bbox = [min(xs), min(ys), max(xs), max(ys)]
                        
                        ocr_results.append({
                            "page": page_num,
                            "bbox": [float(b) for b in bbox],
                            "text": self._sanitize_text(text_str),
                            "confidence": float(conf),
                            "line_id": f"p{page_num}_l{idx}"
                        })
                    return ocr_results
            except Exception as e:
                logger.error(f"PaddleOCR processing failed for {image_path.name}: {e}. Trying fallback.")

        # 2. Try PyTesseract fallback
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(image_path)
            # Fetch detailed box data
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            
            ocr_results = []
            line_idx = 0
            for idx in range(len(data["text"])):
                text_str = data["text"][idx].strip()
                conf = float(data["conf"][idx]) / 100.0 if "conf" in data else 0.8
                
                # Filter out empty texts or low confidence placeholders (-1 in tesseract means no text)
                if text_str and conf >= 0.0:
                    x = float(data["left"][idx])
                    y = float(data["top"][idx])
                    w = float(data["width"][idx])
                    h = float(data["height"][idx])
                    
                    ocr_results.append({
                        "page": page_num,
                        "bbox": [x, y, x + w, y + h],
                        "text": self._sanitize_text(text_str),
                        "confidence": conf,
                        "line_id": f"p{page_num}_l{line_idx}"
                    })
                    line_idx += 1
            if ocr_results:
                logger.info(f"Tesseract OCR extracted {len(ocr_results)} words on Page {page_num}.")
                return ocr_results
        except Exception as e:
            logger.warning(f"PyTesseract OCR failed: {e}. Returning empty layout.")

        return []

    def _sanitize_text(self, text: str) -> str:
        """Sanitizes text, handling Windows-1252 / CP1252 encodings safely."""
        if not text:
            return ""
        try:
            # Check for binary encodings and decode cleanly
            if isinstance(text, bytes):
                return text.decode("utf-8", errors="replace")
            # Force cleanup of common symbols that break string formatting
            text = text.replace("\u201c", '"').replace("\u201d", '"')
            text = text.replace("\u2018", "'").replace("\u2019", "'")
            return text.strip()
        except Exception:
            return str(text).strip()
