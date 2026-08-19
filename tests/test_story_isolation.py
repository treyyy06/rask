import pytest
from pathlib import Path
from multimodal_rag.ingestion import DocumentIngester
from multimodal_rag.segmentation import StorySegmenter

def test_story_isolation_comic():
    pdf_path = Path(__file__).resolve().parent / "data" / "story_comic.pdf"
    assert pdf_path.exists(), "Test PDF does not exist."
    
    ingester = DocumentIngester(str(pdf_path))
    pages_metadata = ingester.process_document()
    
    segmenter = StorySegmenter()
    clusters = segmenter.segment_document(pages_metadata)
    
    # We expect 2 separate story clusters:
    # Cluster 1 should contain Page 1 & 2 (Bear Story)
    # Cluster 2 should contain Page 3 (Elephant Story)
    assert len(clusters) >= 2, "Document did not segment into multiple stories."
    
    # Check that page 1 and page 2 are in the same cluster
    bear_cluster = None
    for cluster in clusters:
        if 1 in cluster:
            bear_cluster = cluster
            break
            
    assert bear_cluster is not None
    assert 2 in bear_cluster, "Page 2 should be clustered with Page 1 (same Bear story)."
    assert 3 not in bear_cluster, "Page 3 (Elephant story) should not contaminate the Bear story cluster."
