import re
import json
import google.generativeai as genai
from typing import Dict, Any, Tuple
from ..config import config
from ..logging_config import get_logger

logger = get_logger("aqu.modality_prediction")

class ModalityPredictor:
    def __init__(self):
        # Specific lexical anchors mapping
        self.modalities_keywords = {
            "Table": {"table", "revenue", "income", "margin", "eps", "percentage", "rate", "year", "quarter", "fy", "metrics", "financials", "gaap", "non-gaap", "profit", "balance sheet", "adjust", "adjusted"},
            "Figure": {"looks like", "color", "appearance", "visible", "shown", "depicted", "wearing", "panels", "panel", "comic", "storyboard", "dress", "character", "man", "woman", "person", "dog", "bear", "elephant", "animal", "draw"},
            "Chart": {"chart", "graph", "plot", "bar chart", "pie chart", "line chart", "axis", "trend", "bars", "legend"},
            "Text": {"paragraph", "statement", "according to text", "report states", "document says", "text says", "author", "write", "described in paragraph"}
        }

    def predict(self, question: str) -> Tuple[str, float]:
        """
        Predicts the query modality (Text, Table, Figure, Chart, Mixed) and a calibrated confidence.
        Returns: (modality, confidence)
        """
        question_clean = question.lower().strip()
        
        # 1. Try LLM Predictor if Gemini API key is available
        if config.GEMINI_API_KEY:
            try:
                genai.configure(api_key=config.GEMINI_API_KEY)
                model = genai.GenerativeModel(config.VLM_MODEL)
                
                prompt = f"""Classify the target evidence modality needed to answer this question.
Modality categories: Text, Table, Figure, Chart, Mixed.
Output MUST be a valid JSON object with keys "modality" (string) and "confidence" (float, between 0.0 and 1.0).
Do not output markdown code blocks or additional text.

Question: "{question}"
"""
                response = model.generate_content(prompt)
                resp_text = response.text.strip()
                
                if resp_text.startswith("```"):
                    lines = resp_text.split("\n")
                    if lines[0].startswith("```json"):
                        resp_text = "\n".join(lines[1:-1])
                    else:
                        resp_text = "\n".join(lines[1:-1])
                        
                data = json.loads(resp_text)
                modality = data["modality"].strip().capitalize()
                confidence = float(data["confidence"])
                
                # Check formatting validation
                if modality in ["Text", "Table", "Figure", "Chart", "Mixed"]:
                    logger.info(f"LLM predicted modality: {modality} (Confidence: {confidence:.2f})")
                    return modality, confidence
            except Exception as e:
                logger.error(f"LLM modality prediction failed: {e}. Running fallback rule classifier.")

        # 2. Rule-based lexical fallback
        table_matches = sum(1 for w in self.modalities_keywords["Table"] if re.search(r'\b' + re.escape(w) + r'\b', question_clean))
        fig_matches = sum(1 for w in self.modalities_keywords["Figure"] if re.search(r'\b' + re.escape(w) + r'\b', question_clean))
        chart_matches = sum(1 for w in self.modalities_keywords["Chart"] if re.search(r'\b' + re.escape(w) + r'\b', question_clean))
        text_matches = sum(1 for w in self.modalities_keywords["Text"] if re.search(r'\b' + re.escape(w) + r'\b', question_clean))
        
        scores = {
            "Table": table_matches,
            "Figure": fig_matches,
            "Chart": chart_matches,
            "Text": text_matches
        }
        
        # Mixed check: if multiple distinct modes contain strong match indicators
        non_zero_modes = [k for k, v in scores.items() if v > 0]
        
        if len(non_zero_modes) >= 2:
            logger.info(f"Mixed modality detected via lexical matches on: {non_zero_modes}")
            return "Mixed", 0.75
            
        best_mode = max(scores, key=scores.get)
        best_score = scores[best_mode]
        
        if best_score > 0:
            # Calibrate confidence based on match density
            confidence = min(0.95, 0.60 + 0.15 * best_score)
            logger.info(f"Rules-based modality prediction: {best_mode} (Confidence: {confidence:.2f})")
            return best_mode, confidence
            
        # Default to Text if no keywords match, but assign a lower uncertainty score (0.45)
        logger.info("Defaulting to Text modality due to lack of lexical match indicators.")
        return "Text", 0.45
