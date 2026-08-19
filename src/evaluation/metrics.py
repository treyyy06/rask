import re
from typing import List, Set

def normalize_text(text: str) -> str:
    """Lowercases text, strips punctuation, articles, and duplicate whitespace."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    # Remove articles
    text = re.sub(r'\b(the|a|an)\b', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def calculate_exact_match(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0

def calculate_token_f1(prediction: str, ground_truth: str) -> float:
    pred_words = normalize_text(prediction).split()
    gt_words = normalize_text(ground_truth).split()
    
    if not pred_words or not gt_words:
        return 1.0 if pred_words == gt_words else 0.0
        
    common = set(pred_words).intersection(set(gt_words))
    if not common:
        return 0.0
        
    precision = len(common) / len(pred_words)
    recall = len(common) / len(gt_words)
    
    return float(2 * (precision * recall) / (precision + recall))

def calculate_recall_at_k(retrieved_pages: List[int], expected_pages: List[int], k: int) -> float:
    if not expected_pages:
        return 1.0 if not retrieved_pages else 0.0
        
    top_k_retrieved = set(retrieved_pages[:k])
    intersection = top_k_retrieved.intersection(set(expected_pages))
    
    return len(intersection) / len(expected_pages)

def calculate_mrr(retrieved_pages: List[int], expected_pages: List[int]) -> float:
    if not expected_pages:
        return 1.0
        
    expected_set = set(expected_pages)
    for idx, p in enumerate(retrieved_pages):
        if p in expected_set:
            return float(1.0 / (idx + 1))
            
    return 0.0

def Levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return Levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
        
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    return previous_row[-1]

def calculate_anls(prediction: str, ground_truth: str, threshold: float = 0.5) -> float:
    """
    Average Normalized Levenshtein Similarity. Used for document QA benchmarks.
    If Levenshtein distance is above threshold, score is 0.0, else 1 - normalized distance.
    """
    p_norm = normalize_text(prediction)
    gt_norm = normalize_text(ground_truth)
    
    if not p_norm and not gt_norm:
        return 1.0
        
    max_len = max(len(p_norm), len(gt_norm))
    if max_len == 0:
        return 0.0
        
    dist = Levenshtein_distance(p_norm, gt_norm)
    normalized_dist = dist / max_len
    
    if normalized_dist > threshold:
        return 0.0
        
    return float(1.0 - normalized_dist)
