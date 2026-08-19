from typing import Dict, Any, List
from .aspect_decomposition import AspectDecomposer
from .refinement import AspectRefiner
from .modality_prediction import ModalityPredictor
from ..logging_config import get_logger

logger = get_logger("aqu.pipeline")

class AQUQueryEngine:
    def __init__(self):
        self.decomposer = AspectDecomposer()
        self.refiner = AspectRefiner()
        self.modality_predictor = ModalityPredictor()

    def analyze_query(self, question: str) -> Dict[str, Any]:
        """
        Executes query understanding: aspect decomposition, refinement, and modality prediction.
        """
        logger.info(f"AQU Analyzing Query: '{question}'")
        
        # 1. Aspect Decomposition
        raw_aspects = self.decomposer.decompose(question)
        
        # 2. Aspect Refinement
        refined_aspects = self.refiner.refine(raw_aspects)
        
        # 3. Modality Prediction
        predicted_modality, modality_confidence = self.modality_predictor.predict(question)
        
        # Calculate overall query understanding confidence (mean of aspects + modality confidence)
        aspect_confidences = [a["confidence"] for a in refined_aspects]
        avg_aspect_conf = sum(aspect_confidences) / len(aspect_confidences) if aspect_confidences else 0.5
        
        # Combined uncertainty score
        query_confidence = 0.40 * avg_aspect_conf + 0.60 * modality_confidence
        
        analysis = {
            "question": question,
            "raw_aspects": raw_aspects,
            "aspects": refined_aspects,
            "predicted_modality": predicted_modality,
            "modality_confidence": modality_confidence,
            "query_confidence": float(query_confidence)
        }
        
        logger.info(f"Query Understanding Completed. Predicted Mode: {predicted_modality} (Query Confidence: {query_confidence:.2f})")
        return analysis
