import os
import re
import json
import google.generativeai as genai
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Tuple
from . import config
from .logging_config import get_logger, setup_logging
from .ingestion import DocumentIngester
from .table_qa import TableExtractor
from .segmentation import StorySegmenter
from .retrieval import RetrievalEngine, classify_query
from .entity_resolution import run_vlm_entity_resolution
from .grounding import GroundingGate
from .validator import AnswerValidator

logger = get_logger("qa_pipeline")

def get_gemini_model_response(prompt: str, image_paths: List[str]) -> str:
    """Helper to query the Gemini model with a prompt and list of image paths."""
    if not config.GEMINI_API_KEY:
        # Mock generator response if API key is missing:
        # Extract evidence text from the prompt and return a snippet from it
        evidence_marker = "EVIDENCE:"
        if evidence_marker in prompt:
            parts = prompt.split(evidence_marker)
            if len(parts) > 1:
                evidence_content = parts[1].split("QUESTION:")[0].strip()
                lines = [l.strip() for l in evidence_content.split("\n") if l.strip()]
                table_lines = [l for l in lines if l.startswith("|")]
                is_table_query = any(k in prompt.lower() for k in ["table", "gaap", "eps", "revenue", "income"])
                
                if is_table_query and table_lines:
                    # Return table headers and the first data row containing digits
                    data_row = ""
                    for l in table_lines[1:]:
                        if any(c.isdigit() for c in l):
                            data_row = l
                            break
                    if data_row:
                        ans = f"Table indicates: {table_lines[0]} with data {data_row}"
                    else:
                        ans = f"Table indicates: {table_lines[0]}"
                else:
                    # Reconstruct paragraph text from non-table, non-header lines to keep wrapped sentences together
                    non_table_lines = [l for l in lines if not l.startswith("|") and not l.startswith("---") and not l.startswith("[---]")]
                    # Filter out chapter/story headings cleanly by line prefix
                    narrative_lines = [l for l in non_table_lines if not l.lower().startswith("chapter") and not l.lower().startswith("story")]
                    
                    full_text = " ".join(narrative_lines)
                    full_text = re.sub(r'\s+', ' ', full_text).strip()
                    if full_text:
                        ans = f"Based on the document: {full_text}"
                    else:
                        ans = "Insufficient document evidence to answer reliably."
                
                # Ensure the mock response ends with a period
                if not ans.endswith((".", "!", "?")):
                    ans += "."
                return ans
        return "Insufficient document evidence to answer reliably."
        
    try:
        genai.configure(api_key=config.GEMINI_API_KEY)
        model = genai.GenerativeModel(config.VLM_MODEL)
        
        # Load images
        loaded_images = []
        for path in image_paths:
            if path and Path(path).exists():
                try:
                    loaded_images.append(Image.open(path))
                except Exception as e:
                    logger.error(f"Failed to load image for answer generation: {e}")
                    
        contents = [prompt] + loaded_images
        response = model.generate_content(contents)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error calling Gemini in generation: {e}", exc_info=True)
        return f"Error occurred during generation: {e}"

class MultimodalQAPipeline:
    def __init__(self, pdf_path: str):
        # Configure logging
        setup_logging()
        
        self.pdf_path = pdf_path
        self.ingester = DocumentIngester(pdf_path)
        self.table_extractor = TableExtractor(pdf_path)
        self.segmenter = StorySegmenter()
        self.gate = GroundingGate()
        self.validator = AnswerValidator()
        
        # Ingestion step
        logger.info("Starting document ingestion...")
        self.pages_metadata = self.ingester.process_document()
        
        # Extract tables and merge into page texts
        logger.info("Extracting tables...")
        for page_data in self.pages_metadata:
            p_num = page_data["page"]
            tables = self.table_extractor.extract_tables_from_page(p_num)
            page_data["tables"] = tables
            
            # Append tables markdown directly to page text representation for indexing
            if tables:
                table_texts = []
                for t in tables:
                    table_texts.append(f"\n--- Extracted Table ---\n{t['markdown']}\n")
                page_data["text"] += "\n" + "\n".join(table_texts)
                
        # Perform story continuity segmentation
        logger.info("Clustering pages into story segments...")
        self.story_clusters = self.segmenter.segment_document(self.pages_metadata)
        
        # Build retrieval engine
        logger.info("Initializing retrieval engine index...")
        self.retriever = RetrievalEngine(self.pages_metadata, self.story_clusters)
        logger.info("Pipeline initialization completed successfully.")

    def run_qa(self, question: str, debug: bool = False) -> Dict[str, Any]:
        """
        Executes the QA pipeline on a user query.
        Returns a structured dictionary (JSON output).
        """
        logger.info(f"--- Processing Query: '{question}' ---")
        
        # 1. Classification
        modality_flags = classify_query(question)
        
        # 2. Retrieval
        retrieved_evidences = self.retriever.retrieve_evidence(question)
        if not retrieved_evidences:
            return self._build_refusal_response(question, "No evidence retrieved.", debug)
            
        # 3. Entity Resolution Pass
        # Extract entities and run presence checks
        from .retrieval import extract_query_entities
        entities = extract_query_entities(question)
        entities_resolution = []
        
        image_paths_for_vlm = [ev["image_path"] for ev in retrieved_evidences[:4] if ev.get("image_path")]
        page_nums_for_vlm = [ev["page"] for ev in retrieved_evidences[:4] if ev.get("image_path")]
        
        is_visual_query = modality_flags["image"] or modality_flags["spatial"] or modality_flags["sequence"]
        
        if entities:
            for ent in entities:
                if is_visual_query and image_paths_for_vlm:
                    res = run_vlm_entity_resolution(ent, image_paths_for_vlm, page_nums_for_vlm)
                else:
                    # Text-based presence check
                    text_matches = []
                    for ev in retrieved_evidences:
                        p_num = ev["page"]
                        meta = next(p for p in self.pages_metadata if p["page"] == p_num)
                        if re.search(r'\b' + re.escape(ent) + r'\b', meta.get("text", "").lower()):
                            text_matches.append(p_num)
                    
                    status = "PRESENT" if text_matches else "ABSENT"
                    res = {
                        "reference": ent,
                        "category": "textual_entity",
                        "status": status,
                        "pages": text_matches,
                        "confidence": 1.0 if text_matches else 0.0,
                        "evidence": f"Text match found on pages {text_matches}." if text_matches else "No text match found in retrieved pages."
                    }
                entities_resolution.append(res)
                
        # 4. Hard Grounding Gate
        is_grounded, grounding_reason = self.gate.validate_grounding(
            question, retrieved_evidences, entities_resolution, modality_flags
        )
        
        if not is_grounded:
            logger.warning(f"Grounding Gate Rejected: {grounding_reason}")
            return self._build_refusal_response(question, f"Grounding failed: {grounding_reason}", debug)
            
        # 5. Evidence Assembly
        # Assemble text and table contents from retrieved pages
        evidence_text_list = []
        source_pages = []
        image_paths_for_gen = []
        
        has_text_evidence = False
        has_table_evidence = False
        has_image_evidence = False
        
        # Filter retrieved pages by similarity threshold and sort chronologically (ascending page order)
        valid_evidences = [ev for ev in retrieved_evidences if ev["final_score"] >= config.SIMILARITY_THRESHOLD]
        valid_evidences.sort(key=lambda x: x["page"])
        
        for ev in valid_evidences:
            p_num = ev["page"]
            # Locate original metadata
            meta = next(p for p in self.pages_metadata if p["page"] == p_num)
            
            text_snippet = meta.get("text", "")
            evidence_text_list.append(f"--- Page {p_num} (Score: {ev['final_score']:.2f}) ---\n{text_snippet}")
            source_pages.append(p_num)
            
            if len(meta.get("text", "").strip()) > 20:
                has_text_evidence = True
            if meta.get("tables"):
                has_table_evidence = True
                
            # Collect images if query has visual intent
            if is_visual_query and meta.get("image_path") and len(image_paths_for_gen) < config.MAX_IMAGES:
                image_paths_for_gen.append(meta["image_path"])
                has_image_evidence = True
                
        evidence_context = "\n\n".join(evidence_text_list)
        
        # 6. Grounded Answer Generation Loop (with retry on consistency checks)
        generation_prompt = self._build_generation_prompt(question, evidence_context, modality_flags)
        
        logger.info("Generating grounded answer...")
        answer = get_gemini_model_response(generation_prompt, image_paths_for_gen)
        
        # Consistency checks & Retries
        max_retries = 2
        retry_count = 0
        validation_status, validation_reason = self.validator.verify_consistency(
            question, evidence_context, answer
        )
        
        while validation_status == "UNSUPPORTED" and retry_count < max_retries:
            retry_count += 1
            logger.warning(f"Validation failed (Attempt {retry_count}/{max_retries}): {validation_reason}. Retrying generation...")
            stricter_prompt = self._build_generation_prompt(
                question, evidence_context, modality_flags, stricter=True, last_reason=validation_reason
            )
            answer = get_gemini_model_response(stricter_prompt, image_paths_for_gen)
            validation_status, validation_reason = self.validator.verify_consistency(
                question, evidence_context, answer
            )
            
        # If still unsupported after retries, trigger hard refusal
        if validation_status == "UNSUPPORTED":
            logger.error(f"Factual consistency check failed after {max_retries} retries. Reason: {validation_reason}")
            return self._build_refusal_response(question, f"Factual validation failed: {validation_reason}", debug)
            
        # 7. Confidence Assessment
        confidence, conf_reason = self._calculate_confidence(
            retrieved_evidences, entities_resolution, validation_status, modality_flags
        )
        
        # Format provenance
        provenance = {
            "text": has_text_evidence,
            "table": has_table_evidence,
            "image": has_image_evidence
        }
        
        # Output result
        result = {
            "question": question,
            "answer": answer,
            "modality": "cross_modal" if modality_flags["cross_modal"] else ("table" if modality_flags["table"] else ("visual" if is_visual_query else "text")),
            "entities": [ent["reference"] for ent in entities_resolution],
            "source_pages": sorted(list(set(source_pages))),
            "evidence": provenance,
            "confidence": confidence,
            "confidence_reason": conf_reason,
            "validation": validation_status.lower()
        }
        
        if debug:
            result["debug_trace"] = {
                "classification": modality_flags,
                "entities_resolution": entities_resolution,
                "retrieved_results": [
                    {
                        "page": r["page"],
                        "final_score": r["final_score"],
                        "entity_score": r["entity_score"],
                        "text_score": r["text_score"],
                        "story_score": r["story_score"],
                        "reason": r["reason"]
                    } for r in retrieved_evidences
                ],
                "story_clusters": [list(c) for c in self.story_clusters],
                "validation_reason": validation_reason
            }
            
        return result

    def _build_generation_prompt(self, question: str, context: str, flags: Dict[str, bool], stricter: bool = False, last_reason: str = "") -> str:
        prompt = f"""
        You are a highly precise, document-grounded Multimodal RAG generator.
        Your task is to answer the question using ONLY the provided page evidence (representing pages in the document).
        
        INSTRUCTIONS:
        1. Base your answer strictly on the provided evidence text and images. Do NOT assume, extrapolate, or use general world knowledge.
        2. If the answer cannot be found or is not supported by the evidence, state exactly "Insufficient document evidence to answer reliably." and nothing else.
        3. Never mix values between columns or years.
        4. Keep GAAP and Non-GAAP operating metrics separate.
        5. Answer clearly and concisely.
        """
        
        if stricter:
            prompt += f"""
            CRITICAL WARNING: Your previous attempt was flagged as UNSUPPORTED or containing a hallucination: "{last_reason}".
            You must correct this. Ensure all proper nouns, causal linkages, and numbers are explicitly matched with the text.
            Do not make any unsupported causal assertions.
            """
            
        prompt += f"""
        EVIDENCE:
        {context}
        
        QUESTION: "{question}"
        
        ANSWER:
        """
        return prompt

    def _build_refusal_response(self, question: str, reason: str, debug: bool) -> Dict[str, Any]:
        result = {
            "question": question,
            "answer": "Insufficient document evidence to answer reliably.",
            "modality": "unknown",
            "entities": [],
            "source_pages": [],
            "evidence": {"text": False, "table": False, "image": False},
            "confidence": "INSUFFICIENT",
            "confidence_reason": f"Grounding gate refused: {reason}",
            "validation": "failed"
        }
        if debug:
            result["debug_trace"] = {
                "refusal_reason": reason,
                "retrieved_results": []
            }
        return result

    def _calculate_confidence(
        self, 
        evidences: List[Dict[str, Any]], 
        entities_resolution: List[Dict[str, Any]], 
        validation_status: str,
        flags: Dict[str, bool]
    ) -> Tuple[str, str]:
        """Calculates confidence categories: HIGH, MEDIUM, LOW, INSUFFICIENT."""
        if not evidences:
            return "INSUFFICIENT", "No evidence retrieved."
            
        # Top page score
        top_score = evidences[0]["final_score"]
        
        # Check validation
        if validation_status == "UNSUPPORTED":
            return "INSUFFICIENT", "Answer was flagged as factual mismatch against evidence."
            
        # Check entities
        all_entities_present = True
        has_entities = len(entities_resolution) > 0
        if has_entities:
            all_entities_present = all(e["status"] == "PRESENT" for e in entities_resolution)
            
        # Compute scoring metrics
        if top_score > 0.8 and validation_status == "SUPPORTED" and (not has_entities or all_entities_present):
            return "HIGH", "Strong matching text/visual evidence fully validated."
        elif top_score > 0.5 and validation_status in ["SUPPORTED", "PARTIALLY_SUPPORTED"]:
            return "MEDIUM", "Relevant evidence retrieved with partial validation."
        else:
            return "LOW", "Weak retrieval match or validation discrepancies."
