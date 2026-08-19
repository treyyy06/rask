import pdfplumber
from pathlib import Path
from typing import List, Dict, Any, Optional
from .logging_config import get_logger

logger = get_logger("table_qa")

def format_table_as_markdown(table: List[List[Optional[str]]]) -> str:
    """Converts a raw list-of-lists table into a clean markdown table string."""
    if not table or not table[0]:
        return ""
        
    # Standardize row lengths to the header length
    num_cols = len(table[0])
    cleaned_table = []
    for r in table:
        cleaned_row = []
        for i in range(num_cols):
            val = r[i] if i < len(r) else ""
            cleaned_row.append(str(val).strip().replace("\n", " ") if val is not None else "")
        # Only keep row if it has at least one non-empty value
        if any(cleaned_row):
            cleaned_table.append(cleaned_row)
            
    if not cleaned_table:
        return ""
        
    headers = cleaned_table[0]
    separator = ["---"] * num_cols
    
    rows = []
    # Header row
    rows.append("| " + " | ".join(headers) + " |")
    # Separator row
    rows.append("| " + " | ".join(separator) + " |")
    # Content rows
    for r in cleaned_table[1:]:
        rows.append("| " + " | ".join(r) + " |")
        
    return "\n".join(rows)

def detect_gaap_status(table_text: str) -> Dict[str, Any]:
    """
    Scans a table structure for financial terms to determine if it contains
    GAAP metrics, Non-GAAP metrics, or both, listing metrics present.
    """
    text_lower = table_text.lower()
    
    gaap_terms = ["gaap", "operating income", "net income", "eps", "revenue", "gross margin"]
    nongaap_terms = ["non-gaap", "adjusted operating", "adjusted net", "adjusted eps", "adjusted EBITDA", "excluding", "pro forma"]
    
    found_gaap = [t for t in gaap_terms if t in text_lower]
    found_nongaap = [t for t in nongaap_terms if t in text_lower]
    
    # Exclude normal terms if they are only present due to Non-GAAP qualifiers
    # e.g., if it says "Non-GAAP Operating Income", "operating income" matches but is qualified as Non-GAAP
    has_gaap = len(found_gaap) > 0
    has_nongaap = len(found_nongaap) > 0
    
    return {
        "has_gaap": has_gaap,
        "has_non_gaap": has_nongaap,
        "gaap_terms_detected": found_gaap,
        "non_gaap_terms_detected": found_nongaap
    }

class TableExtractor:
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        
    def extract_tables_from_page(self, page_num: int) -> List[Dict[str, Any]]:
        """
        Extract tables on a specific page.
        Returns a list of dictionaries with raw tables, markdown representation, and financial categorization.
        """
        tables_metadata = []
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                if page_num < 1 or page_num > len(pdf.pages):
                    logger.error(f"Page number {page_num} is out of bounds for table extraction.")
                    return []
                    
                page = pdf.pages[page_num - 1]
                tables = page.extract_tables()
                
                for idx, table in enumerate(tables):
                    if not table:
                        continue
                        
                    markdown_str = format_table_as_markdown(table)
                    if not markdown_str:
                        continue
                        
                    financial_status = detect_gaap_status(markdown_str)
                    
                    tables_metadata.append({
                        "table_index": idx,
                        "raw_table": table,
                        "markdown": markdown_str,
                        "has_financials": financial_status["has_gaap"] or financial_status["has_non_gaap"],
                        "financial_status": financial_status
                    })
                    
            logger.info(f"Page {page_num} - Extracted {len(tables_metadata)} tables.")
        except Exception as e:
            logger.error(f"Failed to extract tables from page {page_num}: {e}", exc_info=True)
            
        return tables_metadata
