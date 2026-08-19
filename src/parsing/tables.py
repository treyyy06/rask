import pdfplumber
from typing import List, Dict, Any, Optional
from pathlib import Path
from ..logging_config import get_logger

logger = get_logger("parsing.tables")

class TableExtractor:
    def __init__(self):
        pass

    def extract_tables_from_page(self, pdf_path: Path, page_num: int) -> List[Dict[str, Any]]:
        """
        Extract tables structurally from a PDF page using pdfplumber.
        Returns:
        {
            "page": int,
            "bbox": [x_min, y_min, x_max, y_max],
            "headers": List[str],
            "rows": List[List[str]],
            "markdown": str
        }
        """
        pdf_path = Path(pdf_path)
        tables_extracted = []
        if not pdf_path.exists():
            return []

        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                if page_num < 1 or page_num > len(pdf.pages):
                    return []
                
                page = pdf.pages[page_num - 1]
                tables = page.find_tables()
                
                for idx, t in enumerate(tables):
                    bbox = [float(v) for v in t.bbox]
                    data = t.extract()
                    
                    if not data or len(data) < 1:
                        continue
                    
                    headers = [str(h).strip() if h else "" for h in data[0]]
                    rows = []
                    for row in data[1:]:
                        rows.append([str(c).strip() if c else "" for c in row])
                        
                    # Generate normalized Markdown representation
                    markdown_lines = []
                    # Header row
                    markdown_lines.append("| " + " | ".join(headers) + " |")
                    # Separator
                    markdown_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                    # Data rows
                    for r in rows:
                        markdown_lines.append("| " + " | ".join(r) + " |")
                    markdown_str = "\n".join(markdown_lines)
                    
                    tables_extracted.append({
                        "page": page_num,
                        "bbox": bbox,
                        "headers": headers,
                        "rows": rows,
                        "markdown": markdown_str
                    })
            if tables_extracted:
                logger.info(f"Extracted {len(tables_extracted)} tables on Page {page_num} of {pdf_path.name}.")
            return tables_extracted
        except Exception as e:
            logger.error(f"Failed to extract tables on Page {page_num} of PDF {pdf_path.name}: {e}")
            return []
