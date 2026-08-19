import re
from typing import Tuple
from ..logging_config import get_logger

logger = get_logger("validation.answer_validator")

class AnswerValidator:
    def __init__(self):
        pass

    def verify_consistency(self, question: str, evidence: str, answer: str) -> Tuple[str, str]:
        """
        Runs factual consistency validation checks on the generated answer.
        Returns: (status, reason) where status is "SUPPORTED", "PARTIALLY_SUPPORTED", or "UNSUPPORTED"
        """
        ans_clean = answer.strip()
        if not ans_clean:
            return "UNSUPPORTED", "Empty answer"
            
        # 1. Length check
        if len(ans_clean) < 5:
            return "UNSUPPORTED", "Answer is too short"
            
        # 2. Placeholders / Refusal check
        refusal_phrases = [
            "placeholder", "insert here", "unable to answer", 
            "do not have enough information", "don't have enough information",
            "no evidence found", "not mentioned in the text", "insufficient document evidence",
            "insufficient evidence"
        ]
        if any(ph in ans_clean.lower() for ph in refusal_phrases):
            return "PARTIALLY_SUPPORTED", "Contains insufficient-evidence placeholder/refusal phrases"
            
        # 3. Numeric values verification
        # Extract digits, currency values, percentages from the answer
        # e.g., "$1,200M", "69.6", "5%"
        ans_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', ans_clean.replace(",", ""))
        
        evidence_clean = evidence.replace(",", "").lower()
        
        # Verify that each number extracted in the answer actually exists in the retrieved evidence
        # (Ignore small common integers like 0, 1, 2)
        for num in ans_numbers:
            if float(num) > 3.0:
                # Search as literal substring token
                if num not in evidence_clean:
                    # Check if the number is calculated (if the question contains calculation keywords, allow it)
                    if any(w in question.lower() for w in ["increase", "decrease", "change", "difference", "total", "sum", "math"]):
                        continue
                    logger.warning(f"Factual validation failed: Number '{num}' appears in answer but not in source evidence.")
                    return "UNSUPPORTED", f"Hallucinated numerical value: '{num}' is not mentioned in the source evidence"

        # 4. Animal/Character entity substitution check (from our previous verification lesson!)
        question_animals = {w for w in ["bear", "elephant", "lion", "tiger", "giraffe", "monkey", "deer"] if w in question.lower()}
        answer_animals = {w for w in ["bear", "elephant", "lion", "tiger", "giraffe", "monkey", "deer"] if w in answer.lower()}
        
        if question_animals:
            mismatched = answer_animals - question_animals
            if mismatched and not (question_animals & answer_animals):
                logger.warning(f"Factual validation failed: Entity mismatch. Question discusses {question_animals}, but answer discusses {mismatched}.")
                return "UNSUPPORTED", f"Entity mismatch: Answer discusses {list(mismatched)} instead of {list(question_animals)}"

        return "SUPPORTED", "Answer passed all factual consistency checks."
