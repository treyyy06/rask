import pytest
from multimodal_rag.grounding import GroundingGate

def test_grounding_gate_scenarios():
    gate = GroundingGate()
    
    # Scenario A: No evidence retrieved
    is_valid, reason = gate.validate_grounding("What happens to the bear?", [], [], {"text": True, "table": False, "image": False})
    assert is_valid is False
    assert "No evidence" in reason
    
    # Scenario B: Target entity absent
    retrieved = [{"page": 1, "text_content": "Gardening tips for roses.", "image_path": "page_1.png"}]
    entities = [{"reference": "bear", "status": "ABSENT", "confidence": 0.0, "pages": []}]
    flags = {"text": True, "table": False, "image": True}
    
    is_valid, reason = gate.validate_grounding("What happens to the bear?", retrieved, entities, flags)
    assert is_valid is False
    assert "marked as ABSENT" in reason
    
    # Scenario C: Visual query with no image evidence
    retrieved_no_img = [{"page": 1, "text_content": "The bear walks.", "image_path": None}]
    entities_present = [{"reference": "bear", "status": "PRESENT", "confidence": 0.9, "pages": [1]}]
    flags_img = {"text": True, "table": False, "image": True}
    
    is_valid, reason = gate.validate_grounding("Describe the bear's color.", retrieved_no_img, entities_present, flags_img)
    assert is_valid is False
    assert "no images were retrieved" in reason
