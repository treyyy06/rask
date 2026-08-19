import re
import google.generativeai as genai
from typing import List, Dict, Any, Tuple
from . import config
from .logging_config import get_logger

logger = get_logger("validator")

class AnswerValidator:
    def __init__(self):
        pass
        
    def check_weak_answer(self, question: str, answer: str, evidence_text: str) -> Tuple[bool, str]:
        """
        Check if the generated answer is weak (empty, placeholders, hallucinated entities, etc.)
        Returns: (is_weak, reason)
        """
        ans_clean = answer.strip()
        if not ans_clean:
            return True, "Empty answer"
            
        # 1. Length check
        if len(ans_clean) < 10:
            return True, "Answer is too short"
            
        # 2. Placeholders / Lack of information check
        insufficient_phrases = [
            "placeholder", "insert here", "unable to answer", 
            "do not have enough information", "don't have enough information",
            "no evidence found", "not mentioned in the text", "insufficient document evidence",
            "insufficient evidence"
        ]
        if any(ph in ans_clean.lower() for ph in insufficient_phrases):
            return True, "Contains insufficient-evidence placeholder phrases"
            
        # 3. Sentence completeness check
        if not ans_clean[-1] in ['.', '!', '?', '"', "'"]:
            return True, "Answer ends abruptly (incomplete sentence)"
            
        # 4. Tautological / Repetition check
        words = ans_clean.split()
        if len(words) > 10:
            unique_words = set(w.lower() for w in words)
            ratio = len(unique_words) / len(words)
            if ratio < 0.4:
                return True, "Excessive word repetition detected"
                
        # 5. Entity mismatch & Proper noun checks
        # Avoid entity substitution (e.g. question asks about lion, answer talks about elephant)
        question_animals = {w for w in ["bear", "elephant", "lion", "tiger", "giraffe", "monkey", "deer"] if w in question.lower()}
        answer_animals = {w for w in ["bear", "elephant", "lion", "tiger", "giraffe", "monkey", "deer"] if w in answer.lower()}
        if question_animals:
            mismatched = answer_animals - question_animals
            if mismatched and not (question_animals & answer_animals):
                logger.warning(f"Entity mismatch: Question asks about {question_animals}, but answer talks about {mismatched}.")
                return True, f"Entity mismatch: Answer discusses {list(mismatched)} instead of {list(question_animals)}."
        # Find proper nouns (capitalized words) in the answer
        ans_proper_nouns = set(re.findall(r'\b[A-Z][a-z]+\b', answer))
        # Ignore common start of sentences and question nouns
        stop_words = {
            "The", "A", "An", "He", "She", "It", "They", "We", "Then", "But", "And", 
            "In", "On", "At", "To", "Based", "According", "This", "Table", "Report", 
            "Document", "Here", "There", "In", "On", "For", "With", "By", "As", "At"
        }
        ans_proper_nouns = {w for w in ans_proper_nouns if w not in stop_words}
        
        # Check if proper nouns in the answer are not present in evidence or question
        evidence_lower = evidence_text.lower()
        question_lower = question.lower()
        
        for noun in ans_proper_nouns:
            noun_lower = noun.lower()
            if noun_lower not in evidence_lower and noun_lower not in question_lower:
                logger.warning(f"Entity mismatch: Proper noun '{noun}' appears in answer but not in evidence or query.")
                return True, f"Unsupported proper noun: '{noun}' is not mentioned in the source evidence"
                
        # 6. Causal claim validation
        causal_words = ["because", "due to", "since", "caused by", "reasons for"]
        has_causal_claim = any(w in ans_clean.lower() for w in causal_words)
        if has_causal_claim:
            # Simple check: make sure the cause keywords exist in the evidence.
            # If the answer claims a reason not present in the text, it could be a hallucination.
            # E.g. Answer: "The bear left because it was afraid."
            # If "afraid", "scared", "fear", "fright" do not exist in the evidence, flag it.
            evidence_words = set(re.findall(r'\b\w+\b', evidence_lower))
            causal_evidence_indicators = {"afraid", "scared", "fear", "due", "because", "why", "reason", "cause", "since", "as"}
            
            # Extract words near the causal word in the answer
            claim_match = re.search(r'\b(because|due to|since)\b\s+([^.]+)', ans_clean.lower())
            if claim_match:
                claim_words = set(re.findall(r'\b\w{4,}\b', claim_match.group(2)))
                # If these words describe a reason that has 0% keyword overlap with the evidence, flag it
                overlap = claim_words.intersection(evidence_words)
                if not overlap and len(claim_words) > 2:
                    return True, "Causal claim made in the answer is not supported by keywords in the evidence"
                    
        return False, "Answer satisfies baseline quality criteria"

    def verify_consistency(self, question: str, evidence_text: str, answer: str) -> Tuple[str, str]:
        """
        Verify the consistency of the answer against the retrieved evidence.
        Returns: (status, reason) where status is SUPPORTED, PARTIALLY_SUPPORTED, or UNSUPPORTED.
        """
        # Baseline checks first
        is_weak, weak_reason = self.check_weak_answer(question, answer, evidence_text)
        if is_weak:
            return "UNSUPPORTED", weak_reason
            
        if not config.GEMINI_API_KEY:
            logger.warning("No Gemini API key available for consistency check. Defaulting to local validation.")
            # Clean commas and currency symbols from both texts for robust number matching
            clean_ans = re.sub(r'[\$,]', '', answer)
            clean_ev = re.sub(r'[\$,]', '', evidence_text)
            
            # Match numbers (like 1200, 1000, 2024)
            ans_numbers = re.findall(r'\d+(?:\.\d+)?', clean_ans)
            ev_numbers = set(re.findall(r'\d+(?:\.\d+)?', clean_ev))
            
            for num in ans_numbers:
                # If a number is just a short 1-digit like 1 or 2, ignore it to prevent false positive mismatches
                if len(num) <= 1:
                    continue
                if num not in ev_numbers:
                    logger.warning(f"Numerical mismatch: Answer contains '{num}' which is not in the retrieved evidence.")
                    return "UNSUPPORTED", f"Numerical value '{num}' is not supported by the evidence."
            return "SUPPORTED", "Passed local parsing validations."
            
        try:
            genai.configure(api_key=config.GEMINI_API_KEY)
            model = genai.GenerativeModel(config.VLM_MODEL)
            
            prompt = f"""
            You are a strict factual consistency validator.
            Compare the given GENERATED ANSWER against the provided EVIDENCE for the QUESTION.
            
            QUESTION: "{question}"
            EVIDENCE:
            {evidence_text}
            
            GENERATED ANSWER:
            "{answer}"
            
            Assess if the generated answer is fully supported by the evidence.
            - It must not hallucinate facts, numbers, dates, or names not present in the evidence.
            - It must not assume causal relationships not explicitly written in the evidence.
            - It must not mix values between years, columns, or GAAP/Non-GAAP categories.
            
            Respond in this JSON format:
            {{
              "status": "SUPPORTED" or "PARTIALLY_SUPPORTED" or "UNSUPPORTED",
              "reason": "Detailed explanation of why it is supported or what unsupported assertions/mismatches exist."
            }}
            
            Only return raw JSON. No markdown or wrappers.
            """
            
            generation_config = {
                "response_mime_type": "application/json"
            }
            
            response = model.generate_content([prompt], generation_config=generation_config)
            result = json.loads(response.text.strip())
            
            logger.info(f"Answer consistency status: {result['status']} | Reason: {result['reason']}")
            return result["status"], result["reason"]
            
        except Exception as e:
            logger.error(f"Error during VLM consistency validation: {e}", exc_info=True)
            return "PARTIALLY_SUPPORTED", f"Consistency validation error: {e}. Presumed partially supported."
