import pytest
from multimodal_rag.validator import AnswerValidator

def test_weak_answer_detection():
    validator = AnswerValidator()
    
    # 1. Short answer
    assert validator.check_weak_answer("What is GAAP Revenue?", "Short", "GAAP Revenue was $1,000M in 2024.")[0] is True
    
    # 2. Placeholders
    assert validator.check_weak_answer("What happens to bear?", "I don't have enough information to answer this.", "The bear eats honey.")[0] is True
    
    # 3. Proper noun hallucination
    # Answer mentions John, which is not in the evidence
    assert validator.check_weak_answer("What happens to bear?", "John then feeds the Bear.", "The bear eats honey.")[0] is True
    
    # 4. Correct answer
    is_weak, reason = validator.check_weak_answer("What does the bear eat?", "The bear eats honey.", "The bear climbs oak tree and eats honey.")
    assert is_weak is False

def test_numerical_mismatch_fallback():
    validator = AnswerValidator()
    
    # If the answer claims $1,200M but evidence only has $1,000M
    status, reason = validator.verify_consistency(
        "What is 2024 Revenue?",
        "Revenue for 2024 was $1,000M.",
        "The revenue for 2024 was $1,200M."
    )
    assert status == "UNSUPPORTED"
    assert "Numerical value" in reason
