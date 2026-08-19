import pytest
from pathlib import Path
from src.config import config
from src.ingestion.ingester import DocumentIngester
from src.parsing.ocr import OCRProcessor
from src.parsing.layout import LayoutDetector
from src.parsing.tables import TableExtractor
from src.aqu.pipeline import AQUQueryEngine
from src.answering.qa_pipeline import MultimodalQAPipeline

def test_config_dirs():
    """Verify that configuration folders are correctly resolved."""
    assert config.RAW_DIR.exists()
    assert config.RENDERED_DIR.exists()
    assert config.PARSED_DIR.exists()

def test_aqu_modality_predictor():
    """Verify AQU modality prediction cues."""
    engine = AQUQueryEngine()
    
    # 1. Table query classification
    analysis_t = engine.analyze_query("What was the GAAP Operating Income in 2024?")
    assert analysis_t["predicted_modality"] == "Table"
    
    # 2. Figure query classification
    analysis_f = engine.analyze_query("What color is the woman's dress?")
    assert analysis_f["predicted_modality"] == "Figure"
    
    # 3. Text query classification
    analysis_p = engine.analyze_query("According to the paragraph on page 2, what was stated?")
    assert analysis_p["predicted_modality"] == "Text"

def test_document_ingestion_and_parse():
    """Verify document ingestion and table parsing logic."""
    pdf_path = Path(__file__).resolve().parent / "data" / "financial_report.pdf"
    assert pdf_path.exists()
    
    ingester = DocumentIngester()
    doc_id = ingester.get_doc_id(pdf_path)
    
    # Run rendering
    status = ingester.ingest_document(pdf_path, force=True)
    assert status["status"] == "COMPLETED"
    assert status["num_pages"] == 1
    
    # Run table extractor directly
    extractor = TableExtractor()
    tables = extractor.extract_tables_from_page(pdf_path, 1)
    assert len(tables) == 1
    assert "GAAP Operating Income" in tables[0]["headers"]
    assert "2024" in tables[0]["rows"][0][0]

def test_full_pipeline_table_qa():
    """Verify end-to-end table QA retrieval and answering."""
    pipeline = MultimodalQAPipeline()
    
    # Run document indexing
    pipeline.process_raw_documents(force=True)
    
    # Ask table query
    result = pipeline.run_qa("What was the GAAP Operating Income in 2024?", debug=True)
    assert result["validation"] == "supported"
    assert "200M" in result["answer"]
    assert 1 in result["pages"]
    assert "Table" in result["modalities"]

def test_full_pipeline_negation_checks():
    """Verify that negation queries correctly trigger grounding gate refusals."""
    pipeline = MultimodalQAPipeline()
    
    # Ask question about absent entity (lion)
    result = pipeline.run_qa("What does the wild lion eat in the savanna?", debug=True)
    assert result["answer"] == "Insufficient document evidence to answer reliably."
    assert result["validation"] == "unsupported"
    assert len(result["pages"]) == 0
