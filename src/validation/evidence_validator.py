from typing import List, Dict, Any, Tuple
from ..config import config
from ..logging_config import get_logger

logger = get_logger("validation.evidence_validator")

class EvidenceValidator:
    def __init__(self):
        self.threshold = config.SIMILARITY_THRESHOLD

    def validate(self, query_analysis: Dict[str, Any], selected_evidence: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Validates the relevancy, aspect coverage, and modality alignment of selected evidence.
        Returns (is_valid, reason)
        """
        if not selected_evidence:
            return False, "No evidence chunks selected."

        # 1. Check retrieval relevancy threshold
        low_score_count = sum(1 for e in selected_evidence if e.get("retrieval_score", 0.0) < self.threshold)
        # If all retrieved evidence is below our threshold, reject
        if low_score_count == len(selected_evidence):
            return False, f"All selected evidence is below the required similarity threshold ({self.threshold})."

        # 2. Check Modality Consistency
        # If predicted modality is high confidence, verify that matching modality exists in evidence
        pred_modality = query_analysis["predicted_modality"]
        mod_conf = query_analysis["modality_confidence"]
        
        evidence_modalities = {e["modality"] for e in selected_evidence}
        
        # If table or visual, and predicted confidence is strong, require at least one match
        if pred_modality in ["Table", "Figure", "Chart"] and mod_conf >= 0.70:
            if pred_modality not in evidence_modalities:
                logger.warning(f"Modality inconsistency: Predicted {pred_modality} with confidence {mod_conf:.2f}, but evidence modalities are {evidence_modalities}")
                # We do not strictly fail yet (weighted fallbacks are allowed), but warn
                
        # For Mixed modality, require at least two distinct modalities present
        if pred_modality == "Mixed" and len(evidence_modalities) < 2:
            logger.info("Mixed query detected but fewer than 2 modalities found in top evidence.")

        # 3. Check Aspect Semantic Coverage
        aspects = query_analysis.get("aspects", [])
        if aspects:
            covered_count = 0
            evidence_text_lower = " ".join(e["text"].lower() for e in selected_evidence)
            
            for asp in aspects:
                asp_text = asp["aspect"].lower()
                # Split aspect into words to check token coverage
                asp_words = asp_text.split()
                # If any of the key words are found in the evidence text, treat as covered
                if any(w in evidence_text_lower for w in asp_words if len(w) > 3):
                    covered_count += 1
            
            coverage_ratio = covered_count / len(aspects)
            logger.info(f"Aspect semantic coverage: {covered_count}/{len(aspects)} ({coverage_ratio*100:.1f}%)")
            
            if coverage_ratio < 0.30:
                return False, f"Low semantic aspect coverage ({coverage_ratio*100:.1f}%). Missing core search elements."

        # 4. Animal Entity Grounding Gate
        question_animals = {w for w in ["bear", "elephant", "lion", "tiger", "giraffe", "monkey", "deer"] if w in query_analysis["question"].lower()}
        if question_animals:
            evidence_text_lower = " ".join(e["text"].lower() for e in selected_evidence)
            evidence_animals = {w for w in ["bear", "elephant", "lion", "tiger", "giraffe", "monkey", "deer"] if w in evidence_text_lower}
            if not question_animals.intersection(evidence_animals):
                return False, f"Entity validation failed: Question discusses {question_animals} but evidence mentions {evidence_animals or 'no animals'}"

        return True, "Evidence validation passed."
