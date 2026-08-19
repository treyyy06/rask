import time
import json
from pathlib import Path
from typing import List, Dict, Any
from ..config import config
from ..logging_config import get_logger
from .metrics import (
    calculate_exact_match,
    calculate_token_f1,
    calculate_anls,
    calculate_recall_at_k,
    calculate_mrr
)

logger = get_logger("evaluation.evaluator")

class AQU_Evaluator:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.questions_file = config.EVAL_DIR / "questions.json"
        self.expected_pages_file = config.EVAL_DIR / "expected_pages.json"
        self.expected_answers_file = config.EVAL_DIR / "expected_answers.json"
        self.failures_file = config.EVAL_DIR / "failures.json"

    def run_evaluation(self, limit: int = 80, ablation_mode: str = "full") -> Dict[str, Any]:
        """
        Runs evaluation on the dataset and prints a structured summary.
        Supports ablation modes: 'dense_only', 'dense_rerank', 'full'
        """
        # Load dataset files
        if not self.questions_file.exists():
            logger.error("questions.json file not found in data/evaluation/.")
            return {}

        with open(self.questions_file, "r", encoding="utf-8") as f:
            questions = json.load(f)
        
        expected_pages = {}
        if self.expected_pages_file.exists():
            with open(self.expected_pages_file, "r", encoding="utf-8") as f:
                expected_pages = json.load(f)
                
        expected_answers = {}
        if self.expected_answers_file.exists():
            with open(self.expected_answers_file, "r", encoding="utf-8") as f:
                expected_answers = json.load(f)

        # Apply evaluation limit
        eval_list = questions[:limit]
        total_queries = len(eval_list)
        logger.info(f"Running evaluation on {total_queries} queries (Ablation: {ablation_mode}).")

        failures = []
        metrics_sum = {
            "recall_1": 0.0, "recall_5": 0.0, "recall_10": 0.0, "mrr": 0.0,
            "exact_match": 0.0, "token_f1": 0.0, "anls": 0.0,
            "latency": 0.0
        }
        
        modality_labeled_count = 0
        correct_modality_predictions = 0

        # Temporarily adjust pipeline settings for ablation tests
        prev_rerank_val = self.pipeline.use_reranker
        prev_aqu_val = self.pipeline.use_aqu
        
        if ablation_mode == "dense_only":
            self.pipeline.use_reranker = False
            self.pipeline.use_aqu = False
        elif ablation_mode == "dense_rerank":
            self.pipeline.use_reranker = True
            self.pipeline.use_aqu = False
        else:
            self.pipeline.use_reranker = True
            self.pipeline.use_aqu = True

        for idx, q_data in enumerate(eval_list):
            q_id = q_data.get("id", f"q{idx}")
            question = q_data["question"]
            expected_p = expected_pages.get(q_id, [])
            expected_a = expected_answers.get(q_id, "")
            gold_modality = q_data.get("modality")  # Might be missing

            logger.info(f"[{idx+1}/{total_queries}] Query: '{question}'")
            
            start_time = time.time()
            try:
                # Run pipeline
                result = self.pipeline.run_qa(question, debug=True)
                elapsed = time.time() - start_time
                
                # Retrieve validation outputs
                pred_ans = result["answer"]
                retrieved_pages = result.get("source_pages", [])
                pred_mod = result.get("modality", "")

                # 1. Calculate Retrieval Metrics
                rec_1 = calculate_recall_at_k(retrieved_pages, expected_p, 1)
                rec_5 = calculate_recall_at_k(retrieved_pages, expected_p, 5)
                rec_10 = calculate_recall_at_k(retrieved_pages, expected_p, 10)
                mrr_val = calculate_mrr(retrieved_pages, expected_p)

                # 2. Calculate Answer Correctness Metrics
                em_val = calculate_exact_match(pred_ans, expected_a)
                f1_val = calculate_token_f1(pred_ans, expected_a)
                anls_val = calculate_anls(pred_ans, expected_a)

                # Add to metrics accumulator
                metrics_sum["recall_1"] += rec_1
                metrics_sum["recall_5"] += rec_5
                metrics_sum["recall_10"] += rec_10
                metrics_sum["mrr"] += mrr_val
                metrics_sum["exact_match"] += em_val
                metrics_sum["token_f1"] += f1_val
                metrics_sum["anls"] += anls_val
                metrics_sum["latency"] += elapsed

                # 3. Modality Prediction check
                if gold_modality:
                    modality_labeled_count += 1
                    if pred_mod.lower() == gold_modality.lower():
                        correct_modality_predictions += 1

                # 4. Failure Analysis Classification
                is_failed = em_val == 0.0 and f1_val < 0.50
                if is_failed:
                    # Classify stage failures
                    failure_stage = "answer extraction failure"
                    if not retrieved_pages:
                        failure_stage = "retrieval failure"
                    elif expected_p and not set(retrieved_pages).intersection(set(expected_p)):
                        failure_stage = "retrieval failure"
                    elif gold_modality and pred_mod.lower() != gold_modality.lower():
                        failure_stage = "modality prediction failure"
                    elif result.get("validation") == "unsupported":
                        failure_stage = "evidence selection failure"

                    failures.append({
                        "qid": q_id,
                        "question": question,
                        "ground_truth": expected_a,
                        "prediction": pred_ans,
                        "failure_stage": failure_stage,
                        "predicted_modality": pred_mod,
                        "expected_modality": gold_modality or "unknown",
                        "evidence_pages": retrieved_pages,
                        "latency": elapsed
                    })

            except Exception as e:
                logger.error(f"Execution crashed for query '{question}': {e}", exc_info=True)
                failures.append({
                    "qid": q_id,
                    "question": question,
                    "ground_truth": expected_a,
                    "prediction": "",
                    "failure_stage": "parsing failure",
                    "predicted_modality": "unknown",
                    "expected_modality": gold_modality or "unknown",
                    "evidence_pages": [],
                    "latency": 0.0
                })

        # Revert pipeline configurations
        self.pipeline.use_reranker = prev_rerank_val
        self.pipeline.use_aqu = prev_aqu_val

        # Compute averages
        report = {
            "total_queries": total_queries,
            "ablation_mode": ablation_mode,
            "avg_recall_1": metrics_sum["recall_1"] / total_queries,
            "avg_recall_5": metrics_sum["recall_5"] / total_queries,
            "avg_recall_10": metrics_sum["recall_10"] / total_queries,
            "mrr": metrics_sum["mrr"] / total_queries,
            "exact_match": metrics_sum["exact_match"] / total_queries,
            "token_f1": metrics_sum["token_f1"] / total_queries,
            "anls": metrics_sum["anls"] / total_queries,
            "avg_latency": metrics_sum["latency"] / total_queries,
            "modality_labeled_questions": modality_labeled_count,
            "modality_accuracy": (correct_modality_predictions / modality_labeled_count) if modality_labeled_count > 0 else "N/A"
        }

        # Write failure logs
        with open(self.failures_file, "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2)

        logger.info(f"Evaluation completed. Summary: EM: {report['exact_match']*100:.1f}%, F1: {report['token_f1']*100:.1f}%, Latency: {report['avg_latency']:.2f}s")
        return report
