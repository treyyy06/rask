from typing import List, Dict, Any, Tuple
from .logging_config import get_logger

logger = get_logger("grounding")

class GroundingGate:
    def __init__(self):
        pass
        
    def validate_grounding(
        self, 
        query: str, 
        retrieved_evidence: List[Dict[str, Any]], 
        entities_resolution: List[Dict[str, Any]], 
        query_classification: Dict[str, bool]
    ) -> Tuple[bool, str]:
        """
        Validates if the retrieved evidence is sufficient and relevant to answer.
        Returns: (is_valid, validation_reason)
        """
        # 1. Hard Check: Does any evidence exist?
        if not retrieved_evidence:
            return False, "No evidence pages were retrieved."
            
        # Extract text presence
        has_text = any(len(ev.get("text_content", "")) > 10 for ev in retrieved_evidence)
        has_table = any("[---]" in ev.get("text_content", "") or "| " in ev.get("text_content", "") for ev in retrieved_evidence)
        has_image = any(ev.get("image_path") is not None for ev in retrieved_evidence)
        
        # Determine if query requires a specific modality
        requires_image = query_classification.get("image", False)
        requires_table = query_classification.get("table", False)
        
        # 2. Match entities presence
        # If query has entities, at least one target entity must be PRESENT with confidence
        if entities_resolution:
            present_entities = [ent for ent in entities_resolution if ent["status"] == "PRESENT"]
            
            # If all entities are ABSENT, refuse to answer
            if not present_entities:
                return False, f"All query entities ({[e['reference'] for e in entities_resolution]}) were marked as ABSENT in the document."
                
            # If the VLM is highly confident that the entity is absent
            highest_conf_present = max([e["confidence"] for e in present_entities]) if present_entities else 0.0
            if highest_conf_present < 0.4:
                return False, "Evidence lacks the required entities with sufficient confidence."
                
        # 3. Check modality sufficiency
        if requires_table and not has_table and not has_text:
            return False, "Query requires table evidence, but no structured tables or text blocks were retrieved."
            
        if requires_image and not has_image:
            return False, "Query requires visual page images, but no images were retrieved."
            
        # 4. Check image relevance
        # If an image exists but the entities resolution indicates the target entity is not present in the retrieved page set,
        # we reject it as irrelevant visual evidence.
        if requires_image and entities_resolution:
            relevant_pages = set()
            for ent in entities_resolution:
                if ent["status"] == "PRESENT":
                    relevant_pages.update(ent.get("pages", []))
                    
            if relevant_pages:
                retrieved_pages = {ev["page"] for ev in retrieved_evidence if ev.get("image_path")}
                # If there's zero overlap between relevant entity pages and retrieved image pages
                if not retrieved_pages.intersection(relevant_pages):
                    return False, f"Retrieved image pages {list(retrieved_pages)} do not contain the target entities (found on pages {list(relevant_pages)})."
                    
        logger.info("Grounding Gate: PASSED. Evidence is relevant and sufficient.")
        return True, "Evidence is sufficient and grounded."
