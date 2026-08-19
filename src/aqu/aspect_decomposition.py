import re
import json
import google.generativeai as genai
from typing import List, Dict, Any
from ..config import config
from ..logging_config import get_logger

logger = get_logger("aqu.aspect_decomposition")

class AspectDecomposer:
    def __init__(self):
        pass

    def decompose(self, question: str) -> List[Dict[str, Any]]:
        """
        Decomposes a query statement into its constituent semantic aspects.
        Each aspect dict matches:
        {
            "aspect": str,
            "confidence": float
        }
        """
        question = question.strip()
        if not question:
            return []

        # 1. Try LLM Aspect Decomposition if Gemini API is available
        if config.GEMINI_API_KEY:
            try:
                genai.configure(api_key=config.GEMINI_API_KEY)
                model = genai.GenerativeModel(config.VLM_MODEL)
                
                prompt = f"""Decompose the following question into its key semantic search aspects/topics/filters.
Output MUST be valid JSON list containing objects with keys "aspect" (string) and "confidence" (float).
Do not output any Markdown wrapping or prefix, just raw JSON.

Question: "{question}"
"""
                response = model.generate_content(prompt)
                resp_text = response.text.strip()
                
                # Strip json code block formatting if present
                if resp_text.startswith("```"):
                    lines = resp_text.split("\n")
                    if lines[0].startswith("```json"):
                        resp_text = "\n".join(lines[1:-1])
                    else:
                        resp_text = "\n".join(lines[1:-1])
                
                aspects = json.loads(resp_text)
                if isinstance(aspects, list):
                    logger.info(f"LLM extracted aspects: {[a['aspect'] for a in aspects]}")
                    return aspects
            except Exception as e:
                logger.error(f"LLM aspect decomposition failed: {e}. Falling back to lexical extractor.")

        # 2. Rules-based lexical fallback
        # Split search terms by prepositions, articles, and question indicators
        clean_q = re.sub(r'^(what|who|where|how|why|when|is|are|the|a|an|describe|explain|show)\b', '', question, flags=re.IGNORECASE)
        # Split by typical relational prepositions
        connectors = r'\b(in|on|at|of|for|with|by|from|to|between|and|shown|reported)\b'
        parts = [p.strip() for p in re.split(connectors, clean_q, flags=re.IGNORECASE) if p and p.strip()]
        
        aspects = []
        for p in parts:
            # Clean up words and punctuation
            p_clean = re.sub(r'[^\w\s-]', '', p).strip()
            # Retain aspects that contain real search tokens
            if len(p_clean) > 3 and not re.match(r'^(in|on|at|of|for|with|by|from|to|between|and)$', p_clean, re.IGNORECASE):
                aspects.append({
                    "aspect": p_clean,
                    "confidence": 0.70  # Standard rules-based confidence
                })
                
        # If still empty, use clean query as a single aspect
        if not aspects:
            aspects.append({
                "aspect": re.sub(r'[^\w\s-]', '', question).strip(),
                "confidence": 0.50
            })

        logger.info(f"Rules-based aspects extracted: {[a['aspect'] for a in aspects]}")
        return aspects
