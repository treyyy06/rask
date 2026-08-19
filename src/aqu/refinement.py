import re
from typing import List, Dict, Any
from ..logging_config import get_logger

logger = get_logger("aqu.refinement")

class AspectRefiner:
    def __init__(self):
        self.stop_words = {
            "the", "and", "for", "but", "are", "was", "were", "this", "that", 
            "these", "those", "have", "has", "had", "been", "will", "would", 
            "should", "could", "who", "whom", "whose", "which", "how", "why", 
            "can", "may", "might", "must", "look", "looking", "find", "found", 
            "what", "where", "about", "above", "below", "under", "over"
        }

    def refine(self, aspects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Refines aspects, filtering out noisy terms and resolving low-confidence items."""
        refined_aspects = []
        for item in aspects:
            aspect_text = item["aspect"].strip()
            conf = item["confidence"]
            
            # 1. Strip punctuation and double spaces
            aspect_text = re.sub(r'[^\w\s-]', '', aspect_text)
            aspect_text = re.sub(r'\s+', ' ', aspect_text).strip()
            
            # 2. Check if aspect is empty or consists purely of stop words
            words = [w.lower() for w in aspect_text.split()]
            if not words:
                continue
                
            non_stop_words = [w for w in words if w not in self.stop_words]
            if not non_stop_words:
                logger.info(f"Filtered out noise aspect: '{item['aspect']}'")
                continue

            # 3. Handle low confidence refinement:
            # If confidence is low, and text is very long, try split or truncate
            if conf < 0.55 and len(words) > 4:
                # Truncate to key nouns or first 3 words
                refined_text = " ".join(words[:3])
                refined_aspects.append({
                    "aspect": refined_text,
                    "confidence": 0.60  # Boost refined confidence slightly
                })
                logger.info(f"Refined long low-confidence aspect: '{item['aspect']}' -> '{refined_text}'")
            else:
                refined_aspects.append({
                    "aspect": aspect_text,
                    "confidence": conf
                })
                
        return refined_aspects
