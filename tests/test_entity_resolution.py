import pytest
from multimodal_rag.entity_resolution import run_vlm_entity_resolution

def test_entity_resolution_fallback():
    # Test local text-based fallback when GEMINI_API_KEY is not set
    result_present = run_vlm_entity_resolution("bear", ["dummy_path.png"], [1, 2])
    assert result_present["reference"] == "bear"
    assert result_present["status"] == "PRESENT"
    assert result_present["confidence"] > 0.5
    
    result_absent = run_vlm_entity_resolution("lion", [], [])
    assert result_absent["status"] == "ABSENT"
    assert result_absent["confidence"] == 0.0
