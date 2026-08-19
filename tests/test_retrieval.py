import pytest
from pathlib import Path
from multimodal_rag.ingestion import DocumentIngester
from multimodal_rag.segmentation import StorySegmenter
from multimodal_rag.retrieval import RetrievalEngine, classify_query, extract_query_entities

def test_query_classification():
    q_table = "What was the Non-GAAP EPS in FY 2024?"
    flags_table = classify_query(q_table)
    assert flags_table["table"] is True
    assert flags_table["image"] is False
    
    q_seq = "What happens to the character from beginning to end?"
    flags_seq = classify_query(q_seq)
    assert flags_seq["sequence"] is True
    assert flags_seq["image"] is True

def test_entity_extraction():
    q = "What happens to the bear in the story?"
    entities = extract_query_entities(q)
    assert "bear" in entities
    assert "story" not in entities

def test_hybrid_retrieval_ranking():
    pdf_path = Path(__file__).resolve().parent / "data" / "story_comic.pdf"
    ingester = DocumentIngester(str(pdf_path))
    pages_metadata = ingester.process_document()
    
    segmenter = StorySegmenter()
    clusters = segmenter.segment_document(pages_metadata)
    
    engine = RetrievalEngine(pages_metadata, clusters)
    
    # Query about the Bear
    results = engine.retrieve_evidence("What does the bear find in the oak tree?")
    
    # Page 2 has exact oak tree beehive text, Page 1 has bear text, Page 3 has elephant text.
    # Page 2 should be ranked #1, Page 1 should be ranked higher than Page 3 because of Bear story alignment.
    assert len(results) > 0
    assert results[0]["page"] == 2, "Page 2 should be top ranked for bear oak tree question."
    
    # Find page rankings
    rank_page_1 = next(idx for idx, r in enumerate(results) if r["page"] == 1)
    rank_page_3 = next(idx for idx, r in enumerate(results) if r["page"] == 3)
    assert rank_page_1 < rank_page_3, "Page 1 (Bear) should rank higher than Page 3 (Elephant) for a Bear query."
