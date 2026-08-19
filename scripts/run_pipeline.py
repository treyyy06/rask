import os
import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add workspace directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Load local environment variables (.env file)
load_dotenv()

from src.config import config
from src.logging_config import setup_logging, get_logger
from src.answering.qa_pipeline import MultimodalQAPipeline
from src.evaluation.evaluator import AQU_Evaluator

# Initialize logger
setup_logging()
logger = get_logger("scripts.run_pipeline")

def main():
    parser = argparse.ArgumentParser(description="AQU-MR Pipeline Runner")
    parser.add_argument("--stage", required=True, choices=["ingest", "parse", "chunk", "embed", "index", "query", "evaluate"],
                        help="The stage of the pipeline to run.")
    parser.add_argument("--question", type=str, help="Question to query (only for query stage)")
    parser.add_argument("--doc-id", type=str, help="Specific document ID to filter execution")
    parser.add_argument("--limit", type=int, default=80, help="Query limit for evaluation stage (e.g. 5, 10, 80)")
    parser.add_argument("--resume", action="store_true", help="Resume from last completed step")
    parser.add_argument("--force", action="store_true", help="Force rebuild/overwrite of files")
    parser.add_argument("--debug", action="store_true", help="Expose full pipeline logs and aspect analysis trace")
    parser.add_argument("--ablation", choices=["dense_only", "dense_rerank", "full"], default="full",
                        help="Ablation experiment mode for evaluation")

    args = parser.parse_args()

    # Apply debug logs if requested
    if args.debug:
        import logging
        setup_logging(level=logging.DEBUG)

    # Initialize Pipeline
    pipeline = MultimodalQAPipeline()

    # Stage Ingestion / Processing
    if args.stage in ["ingest", "parse", "chunk", "embed", "index"]:
        pipeline.process_raw_documents(force=args.force)
        print(f"\nStage '{args.stage}' completed successfully.")
        return

    # Stage Query
    if args.stage == "query":
        if not args.question:
            logger.error("A query question is required: --question '...'")
            sys.exit(1)
        
        # Verify that index exists
        if not pipeline.faiss_index.index_file.exists():
            logger.warning("FAISS vector index file not found. Re-processing raw documents first.")
            pipeline.process_raw_documents(force=False)
            
        result = pipeline.run_qa(args.question, debug=args.debug or True)
        
        print("\n" + "="*50)
        print(f"QUESTION: {args.question}")
        print(f"ANSWER: {result['answer']}")
        print(f"CONFIDENCE: {result['confidence']}")
        print(f"SOURCE PAGES: {result['pages']}")
        print(f"MODALITIES: {result['modalities']}")
        print(f"VALIDATION: {result['validation']}")
        print("="*50)
        return

    # Stage Evaluate
    if args.stage == "evaluate":
        # Check if evaluation files exist in evaluation folder
        # If not, let's copy them from workspace evaluation/
        src_questions = Path("evaluation/questions.json")
        dest_questions = config.EVAL_DIR / "questions.json"
        
        if src_questions.exists() and not dest_questions.exists():
            import shutil
            shutil.copy(src_questions, dest_questions)
            shutil.copy(Path("evaluation/expected_pages.json"), config.EVAL_DIR / "expected_pages.json")
            shutil.copy(Path("evaluation/expected_answers.json"), config.EVAL_DIR / "expected_answers.json")

        evaluator = AQU_Evaluator(pipeline)
        report = evaluator.run_evaluation(limit=args.limit, ablation_mode=args.ablation)
        
        print("\n" + "="*60)
        print(f"AQU-MR EVALUATION REPORT (Ablation: {args.ablation.upper()})")
        print("="*60)
        print(f"Total Queries Evaluated:    {report['total_queries']}")
        print(f"Recall@1:                   {report['avg_recall_1']*100:.1f}%")
        print(f"Recall@5:                   {report['avg_recall_5']*100:.1f}%")
        print(f"Recall@10:                  {report['avg_recall_10']*100:.1f}%")
        print(f"MRR:                        {report['mrr']:.3f}")
        print(f"Exact Match:                {report['exact_match']*100:.1f}%")
        print(f"Token F1:                   {report['token_f1']*100:.1f}%")
        print(f"ANLS:                       {report['anls']:.3f}")
        print(f"Average Latency:            {report['avg_latency']:.2f}s")
        print(f"Modality Accuracy:          {report['modality_accuracy']}")
        print("="*60)
        return

if __name__ == "__main__":
    main()
