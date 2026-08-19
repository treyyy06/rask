import pytest
from multimodal_rag.retrieval import classify_query

def test_visual_query_intents():
    queries = [
        "What is next to the tree in panel 2?",
        "What color dress is the girl wearing?",
        "Where is the bear positioned relative to the river?",
        "Which landmark is closest to the bridge on the map?"
    ]
    
    for q in queries:
        flags = classify_query(q)
        assert flags["image"] is True, f"Visual intent not detected for query: {q}"
