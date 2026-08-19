import os
import google.generativeai as genai
from PIL import Image
from typing import List, Dict, Any
from ..config import config
from ..logging_config import get_logger
from .text_answerer import TextAnswerer
from .table_answerer import TableAnswerer
from .visual_answerer import VisualAnswerer

logger = get_logger("answering.multimodal")

class MultimodalAnswerer:
    def __init__(self):
        self.text_answerer = TextAnswerer()
        self.table_answerer = TableAnswerer()
        self.visual_answerer = VisualAnswerer()

    def answer(self, question: str, evidence: List[Dict[str, Any]]) -> str:
        """Synthesizes an answer by combining text, table, and visual details."""
        # Collect distinct evidence units
        table_ev = [e for e in evidence if e["modality"] == "Table"]
        visual_ev = [e for e in evidence if e["modality"] in ["Figure", "Chart"]]
        text_ev = [e for e in evidence if e["modality"] == "Text"]

        # 1. Use Gemini if API key is active
        if config.GEMINI_API_KEY:
            try:
                genai.configure(api_key=config.GEMINI_API_KEY)
                model = genai.GenerativeModel(config.VLM_MODEL)
                
                # Load images
                image_paths = []
                for ve in visual_ev:
                    crop_p = ve.get("metadata", {}).get("crop_path")
                    if crop_p and os.path.exists(crop_p):
                        image_paths.append(crop_p)
                pil_images = [Image.open(p) for p in image_paths[:config.MAX_IMAGES]]
                
                # Construct combined context
                contexts = []
                if text_ev:
                    contexts.append("--- TEXT EVIDENCE ---\n" + "\n\n".join(t["text"] for t in text_ev))
                if table_ev:
                    contexts.append("--- TABLE EVIDENCE ---\n" + "\n\n".join(tbl["text"] for tbl in table_ev))
                evidence_text = "\n\n".join(contexts)
                
                prompt = f"""Use the supplied document evidence (text, tables, and visual crops) to answer the question.
Synthesize details from both tabular, textual, or visual formats as needed.
Your answer must be accurate, grounded, and concise.

CONTEXT:
{evidence_text}

QUESTION: {question}
ANSWER:"""
                
                contents = pil_images + [prompt]
                response = model.generate_content(contents)
                return response.text.strip()
            except Exception as e:
                logger.error(f"Gemini multimodal answering failed: {e}. Trying offline synthesis.")

        # 2. Offline fallback synthesis:
        # Run sub-answerers on available elements and combine them!
        sub_answers = []
        if table_ev:
            ans_t = self.table_answerer.answer(question, table_ev)
            sub_answers.append(f"Table details: {ans_t}")
        if visual_ev:
            ans_v = self.visual_answerer.answer(question, visual_ev)
            sub_answers.append(f"Visual details: {ans_v}")
        if text_ev and not table_ev:
            ans_p = self.text_answerer.answer(question, text_ev)
            sub_answers.append(f"Textual details: {ans_p}")
            
        if sub_answers:
            # Combine cleanly
            return " | ".join(sub_answers)
            
        return "Insufficient document evidence to answer reliably."
