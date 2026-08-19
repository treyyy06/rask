import re
from typing import List, Dict, Any, Set
from .logging_config import get_logger

logger = get_logger("segmentation")

def extract_candidate_entities(text: str) -> Set[str]:
    """Extract capitalization and noun phrases as candidate entity names."""
    if not text:
        return set()
    # Simple regex to extract capitalized names / nouns
    words = re.findall(r'\b[A-Z][a-z]+\b', text)
    # Also extract common repeating nouns (simple filters)
    common_stops = {"The", "A", "An", "He", "She", "It", "They", "We", "Then", "But", "And", "In", "On", "At", "To"}
    entities = {w.lower() for w in words if w not in common_stops}
    return entities

def calculate_jaccard_similarity(set1: Set[Any], set2: Set[Any]) -> float:
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

class StorySegmenter:
    def __init__(self):
        pass
        
    def segment_document(self, pages_metadata: List[Dict[str, Any]]) -> List[Set[int]]:
        """
        Segment pages into story/continuity clusters.
        Returns a list of sets, where each set contains the page numbers (1-indexed) in that cluster.
        """
        if not pages_metadata:
            return []
            
        num_pages = len(pages_metadata)
        if num_pages == 1:
            return [{pages_metadata[0]["page"]}]
            
        clusters = []
        current_cluster = {pages_metadata[0]["page"]}
        
        # Analyze boundary transitions sequentially
        for idx in range(num_pages - 1):
            page_current = pages_metadata[idx]
            page_next = pages_metadata[idx + 1]
            
            p_curr_num = page_current["page"]
            p_next_num = page_next["page"]
            
            text_curr = page_current.get("text", "")
            text_next = page_next.get("text", "")
            
            # Extract nouns
            ents_curr = extract_candidate_entities(text_curr)
            ents_next = extract_candidate_entities(text_next)
            
            entity_overlap = calculate_jaccard_similarity(ents_curr, ents_next)
            
            # Words Jaccard similarity (ignoring stop words)
            words_curr = set(re.findall(r'\b\w{3,}\b', text_curr.lower()))
            words_next = set(re.findall(r'\b\w{3,}\b', text_next.lower()))
            text_overlap = calculate_jaccard_similarity(words_curr, words_next)
            
            # Check for header transitions (e.g. "Chapter", "Story", or bold titles)
            is_new_story_indicator = False
            
            # Helper to extract chapter/story number from a text block
            def get_chapter_num(t: str):
                m = re.search(r'\b(?:chapter|story|part|section|episode|act)\s+(\d+)\b', t, re.IGNORECASE)
                if m:
                    return int(m.group(1))
                rm = re.search(r'\b(?:chapter|story|part|section|episode|act)\s+([IVXLCDM]+)\b', t, re.IGNORECASE)
                if rm:
                    r_map = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}
                    return r_map.get(rm.group(1).upper())
                return None

            curr_chap = get_chapter_num(text_curr)
            next_chap = get_chapter_num(text_next)
            
            if next_chap is not None:
                if curr_chap is not None and curr_chap != next_chap:
                    is_new_story_indicator = True
                elif curr_chap is None:
                    # Next page starts a chapter but current page was not in one (e.g. introduction page -> Chapter 1)
                    is_new_story_indicator = True
            else:
                # If next page has a bold uppercase word that looks like a title transition
                first_lines = [line.strip() for line in text_next.split('\n') if line.strip()][:2]
                for line in first_lines:
                    if re.match(r'^[A-Z\s]{5,15}$', line):
                        # Simple capitalized title transition check, but verify low overlap to avoid false alarms
                        if text_overlap < 0.15:
                            is_new_story_indicator = True
                            break
            
            # Make a clustering split decision
            # Boundary split conditions:
            # - Explicit new story indicator
            # - Extremely low text overlap (< 0.05) AND low entity overlap (< 0.05) when both pages actually have text
            has_substantial_text = len(words_curr) > 5 and len(words_next) > 5
            
            should_split = False
            if is_new_story_indicator:
                should_split = True
                logger.info(f"Page transition {p_curr_num} -> {p_next_num}: Splitting on header indicator.")
            elif has_substantial_text and text_overlap < 0.06 and entity_overlap < 0.06:
                should_split = True
                logger.info(f"Page transition {p_curr_num} -> {p_next_num}: Splitting on low similarity (text: {text_overlap:.2f}, entities: {entity_overlap:.2f}).")
            
            if should_split:
                clusters.append(current_cluster)
                current_cluster = {p_next_num}
            else:
                current_cluster.add(p_next_num)
                
        clusters.append(current_cluster)
        
        # Log final clustering overview
        for i, cluster in enumerate(clusters):
            logger.info(f"Story Cluster {i}: Pages {sorted(list(cluster))}")
            
        return clusters

    def find_story_pages_for_page(self, page_num: int, clusters: List[Set[int]]) -> Set[int]:
        """Find all pages in the same story cluster as the target page."""
        for cluster in clusters:
            if page_num in cluster:
                return cluster
        return {page_num}
