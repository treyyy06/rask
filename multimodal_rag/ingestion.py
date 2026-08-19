import os
import re
import hashlib
import pypdfium2 as pdfium
from PIL import Image
from pathlib import Path
from typing import Dict, Any, List, Tuple
from . import config
from .logging_config import get_logger

logger = get_logger("ingestion")

def get_file_hash(file_path: Path) -> str:
    """Generate MD5 hash of a file to uniquely identify it."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def calculate_ocr_quality(text: str) -> float:
    """
    Evaluate the quality of OCR extracted text.
    Returns a score between 0.0 (garbage/empty) and 1.0 (clean text).
    """
    if not text or not text.strip():
        return 0.0
        
    text_strip = text.strip()
    total_chars = len(text_strip)
    
    # 1. Check for garbage repeated characters (e.g., 'aaaaa', '$$$$$')
    repeated_pattern = re.compile(r'(.)\1{4,}')
    repeated_chars_count = sum(len(m.group(0)) for m in repeated_pattern.finditer(text_strip))
    
    # 2. Check for high density of special non-alphanumeric characters
    alphanumeric_chars = sum(c.isalnum() or c.isspace() for c in text_strip)
    special_char_ratio = 1.0 - (alphanumeric_chars / total_chars)
    
    # 3. Check for broken words (e.g. containing letters mixed with symbols or numbers in weird spots)
    words = text_strip.split()
    if not words:
        return 0.0
        
    garbage_words = 0
    for word in words:
        # If word has letters but contains multiple punctuation marks inside it
        if len(word) > 3 and re.search(r'[a-zA-Z]+[^a-zA-Z\s]+[a-zA-Z]+', word):
            garbage_words += 1
            
    garbage_word_ratio = garbage_words / len(words)
    
    # Calculate components
    repetition_penalty = (repeated_chars_count / total_chars) * 2.0
    special_penalty = special_char_ratio * 1.5
    garbage_word_penalty = garbage_word_ratio * 2.0
    
    # Final quality score
    score = 1.0 - (repetition_penalty + special_penalty + garbage_word_penalty)
    score = max(0.0, min(1.0, score))
    
    # Extremely low text density check (if page is mostly empty but has tiny garbage)
    if total_chars < 30 and score < 0.8:
        score = min(score, 0.3)
        
    return score

def render_pdf_page(pdf_path: Path, page_num: int, output_path: Path) -> bool:
    """Renders a PDF page to a PNG image using pypdfium2."""
    try:
        pdf = pdfium.PdfDocument(str(pdf_path))
        if page_num < 1 or page_num > len(pdf):
            logger.error(f"Page number {page_num} is out of bounds for PDF with {len(pdf)} pages.")
            return False
            
        page = pdf[page_num - 1]
        # Render page at 150 DPI for high-quality text and visual retrieval
        bitmap = page.render(scale=2.0)
        pil_img = bitmap.to_pil()
        
        # Resize if image exceeds limit
        w, h = pil_img.size
        max_side = config.IMAGE_SIZE
        if max(w, h) > max_side:
            if w > h:
                new_w = max_side
                new_h = int(h * (max_side / w))
            else:
                new_h = max_side
                new_w = int(w * (max_side / h))
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pil_img.save(output_path, "PNG")
        logger.info(f"Rendered Page {page_num} to {output_path.name}")
        return True
    except Exception as e:
        logger.error(f"Failed to render Page {page_num} of PDF {pdf_path}: {e}", exc_info=True)
        return False

def extract_ocr_from_image(image_path: Path) -> str:
    """Extracts text from an image using pytesseract if available."""
    try:
        import pytesseract
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        logger.warning(f"PyTesseract OCR failed or not installed: {e}")
        return ""

class DocumentIngester:
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF document not found: {pdf_path}")
            
        self.pdf_hash = get_file_hash(self.pdf_path)
        self.output_dir = config.SCRATCH_DIR / "pages" / self.pdf_hash
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load PDF using pypdfium2 to check page count
        self.pdf = pdfium.PdfDocument(str(self.pdf_path))
        self.num_pages = len(self.pdf)
        logger.info(f"Initialized DocumentIngester for {self.pdf_path.name} (pages: {self.num_pages}, hash: {self.pdf_hash})")

    def process_document(self) -> List[Dict[str, Any]]:
        """
        Process the entire document:
        1. Render page images
        2. Perform OCR and extract native PDF text
        3. Evaluate OCR quality and flag fallback requirements
        """
        pages_metadata = []
        
        for i in range(1, self.num_pages + 1):
            image_path = self.output_dir / f"page_{i}.png"
            # Render page image if it does not already exist
            image_exists = image_path.exists()
            if not image_exists:
                image_exists = render_pdf_page(self.pdf_path, i, image_path)
                
            # Attempt to extract text from PDF natively first
            pdf_page = self.pdf[i - 1]
            text = pdf_page.get_textpage().get_text_bounded() or ""
            
            # If native text is empty or very short, try OCR
            is_ocr_used = False
            if len(text.strip()) < 50 and image_exists:
                ocr_text = extract_ocr_from_image(image_path)
                if len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text
                    is_ocr_used = True
            
            # Calculate OCR quality
            quality_score = calculate_ocr_quality(text)
            
            # Flag visual fallback if quality is poor (< 0.4) or empty
            use_visual_fallback = quality_score < 0.4
            
            metadata = {
                "page": i,
                "text": text,
                "image_path": str(image_path) if image_exists else None,
                "ocr_quality": quality_score,
                "is_ocr_used": is_ocr_used,
                "use_visual_fallback": use_visual_fallback
            }
            pages_metadata.append(metadata)
            logger.info(f"Processed Page {i} - Length: {len(text)}, Quality: {quality_score:.2f}, Fallback: {use_visual_fallback}")
            
        return pages_metadata

    def find_page_image(self, page_num: int) -> str:
        """Find the image path of a specific page. Never crashes, returns None if unavailable."""
        image_path = self.output_dir / f"page_{page_num}.png"
        if image_path.exists():
            return str(image_path)
            
        # Try rendering on demand
        if 1 <= page_num <= self.num_pages:
            success = render_pdf_page(self.pdf_path, page_num, image_path)
            if success:
                return str(image_path)
                
        logger.warning(f"Page image for page {page_num} is unavailable.")
        return None
