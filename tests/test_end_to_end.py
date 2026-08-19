import pytest
from pathlib import Path
from multimodal_rag.qa_pipeline import MultimodalQAPipeline

def test_pipeline_story_comic_end_to_end():
    pdf_path = Path(__file__).resolve().parent / "data" / "story_comic.pdf"
    assert pdf_path.exists()
    
    pipeline = MultimodalQAPipeline(str(pdf_path))
    
    # 1. Test query for bear
    result_bear = pipeline.run_qa("What happens to the bear after finding honey?", debug=True)
    assert "bear" in result_bear["entities"]
    assert 2 in result_bear["source_pages"]
    assert 3 not in result_bear["source_pages"], "Bear query should not retrieve Elephant pages (page 3)."
    
    # 2. Test query for elephant
    result_elephant = pipeline.run_qa("What is the elephant looking for?", debug=True)
    assert 3 in result_elephant["source_pages"]
    assert 1 not in result_elephant["source_pages"], "Elephant query should not retrieve Bear pages (page 1)."
    
    # 3. Test negative query (hard grounding refusal)
    result_neg = pipeline.run_qa("What does the lion eat in the savanna?", debug=True)
    assert result_neg["answer"] == "Insufficient document evidence to answer reliably."
    assert result_neg["confidence"] == "INSUFFICIENT"

def test_pipeline_financial_table_end_to_end():
    pdf_path = Path(__file__).resolve().parent / "data" / "financial_report.pdf"
    assert pdf_path.exists()
    
    pipeline = MultimodalQAPipeline(str(pdf_path))
    
    # Run financial metric query
    result = pipeline.run_qa("What was the Non-GAAP EPS in 2024?", debug=True)
    assert result["modality"] == "table"
    assert 1 in result["source_pages"]
    # Check that it did not fail grounding or validation
    assert result["validation"] != "failed"
