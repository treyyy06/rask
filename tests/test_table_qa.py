import pytest
from pathlib import Path
from multimodal_rag.table_qa import TableExtractor, detect_gaap_status

def test_financial_table_extraction():
    pdf_path = Path(__file__).resolve().parent / "data" / "financial_report.pdf"
    assert pdf_path.exists()
    
    extractor = TableExtractor(str(pdf_path))
    tables = extractor.extract_tables_from_page(1)
    
    assert len(tables) == 1, "Financial table should be extracted."
    table_meta = tables[0]
    
    # Check table structure
    assert table_meta["has_financials"] is True
    assert "| Revenue |" in table_meta["markdown"]
    assert "| 2024 |" in table_meta["markdown"]
    
    # GAAP/Non-GAAP detection
    status = detect_gaap_status(table_meta["markdown"])
    assert status["has_gaap"] is True
    assert status["has_non_gaap"] is True
    assert "revenue" in status["gaap_terms_detected"]
    assert "non-gaap" in status["non_gaap_terms_detected"]
