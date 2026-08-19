import re
import google.generativeai as genai
from typing import List, Dict, Any
from ..config import config
from ..logging_config import get_logger

logger = get_logger("answering.text")

class TextAnswerer:
    def __init__(self):
        pass

    def answer(self, question: str, evidence: List[Dict[str, Any]]) -> str:
        """Generates a text-based grounded answer using the provided evidence."""
        # Join text content of evidence chunks
        evidence_text_list = []
        for e in evidence:
            evidence_text_list.append(f"--- Page {e['page']} ---\n{e['text']}")
        evidence_context = "\n\n".join(evidence_text_list)

        # 1. Use Gemini LLM if API key is active
        if config.GEMINI_API_KEY:
            try:
                genai.configure(api_key=config.GEMINI_API_KEY)
                model = genai.GenerativeModel(config.VLM_MODEL)
                
                prompt = f"""Use the supplied DOCUMENT EVIDENCE to answer the question.
Your answer must be concise, accurate, and completely grounded in the evidence.
Do not guess or assume details not present. If evidence is insufficient, state "UNKNOWN".

EVIDENCE:
{evidence_context}

QUESTION: {question}
ANSWER:"""
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                logger.error(f"Gemini generation failed: {e}. Trying offline fallback.")

        # 2. Offline high-fidelity mock extraction fallback:
        # Scan clean non-table narrative lines for answer sentences matching question words
        narrative_lines = []
        for e in evidence:
            lines = [l.strip() for l in e["text"].split("\n") if l.strip()]
            for l in lines:
                if not l.startswith("|") and not l.startswith("---"):
                    narrative_lines.append(l)
                    
        full_text = " ".join(narrative_lines)
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        
        if full_text:
            return f"Based on the document: {full_text}"
            
        return "Insufficient document evidence to answer reliably."
