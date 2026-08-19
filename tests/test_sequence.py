import pytest
from pathlib import Path
from multimodal_rag.ingestion import DocumentIngester
from multimodal_rag.segmentation import StorySegmenter
from multimodal_rag.retrieval import RetrievalEngine

def test_chronological_sequence_ordering():
    pdf_path = Path(__file__).resolve().parent / "data" / "story_comic.pdf"
    ingester = DocumentIngester(str(pdf_path))
    pages_metadata = ingester.process_document()
    
    segmenter = StorySegmenter()
    clusters = segmenter.segment_document(pages_metadata)
    
    engine = RetrievalEngine(pages_metadata, clusters)
    
    # Query with chronological sequence keywords (beginning, end)
    results = engine.retrieve_evidence("Describe the bear's story from beginning to end.")
    
    # The returned page numbers should be in strictly ascending order
    pages = [r["page"] for r in results]
    sorted_pages = sorted(pages)
    
    assert pages == sorted_pages, f"Sequence results were not sorted chronologically: {pages} vs {sorted_pages}"
