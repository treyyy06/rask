import argparse
import sys
import json
from pathlib import Path
from multimodal_rag.qa_pipeline import MultimodalQAPipeline
from multimodal_rag.logging_config import get_logger

logger = get_logger("cli")

def main():
    parser = argparse.ArgumentParser(description="Multimodal RAG QA pipeline CLI.")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF document.")
    parser.add_argument("question", type=str, help="The query question to ask.")
    parser.add_argument("--debug", action="store_true", help="Print debug/tracing logs.")
    parser.add_argument("--json", action="store_true", help="Print structured JSON output only.")
    
    args = parser.parse_args()
    
    # Verify file
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        if args.json:
            print(json.dumps({"error": f"PDF path does not exist: {args.pdf_path}"}))
        else:
            print(f"Error: PDF path does not exist: {args.pdf_path}", file=sys.stderr)
        sys.exit(1)
        
    try:
        # Run pipeline
        pipeline = MultimodalQAPipeline(str(pdf_path))
        result = pipeline.run_qa(args.question, debug=args.debug)
        
        if args.json:
            # Output formatted JSON
            print(json.dumps(result, indent=2))
        else:
            print("\n" + "="*50)
            print(f"QUESTION: {result['question']}")
            print(f"ANSWER: {result['answer']}")
            print(f"CONFIDENCE: {result['confidence']} ({result['confidence_reason']})")
            print(f"SOURCE PAGES: {result['source_pages']}")
            print(f"EVIDENCE: Text: {result['evidence']['text']}, Tables: {result['evidence']['table']}, Images: {result['evidence']['image']}")
            print(f"VALIDATION: {result['validation']}")
            print("="*50)
            
            if args.debug and "debug_trace" in result:
                print("\n--- DEBUG TRACE ---")
                trace = result["debug_trace"]
                print(f"Modality Classification: {trace['classification']}")
                print(f"Story Clusters: {trace['story_clusters']}")
                print("Retrieved Page Scores:")
                for r in trace["retrieved_results"]:
                    print(f"  Page {r['page']}: Final={r['final_score']:.3f} | Entity={r['entity_score']:.2f} | Text={r['text_score']:.2f} | Story={r['story_score']:.2f} | Reason: {r['reason']}")
                if trace.get("entities_resolution"):
                    print(f"VLM Entities Resolved: {trace['entities_resolution']}")
                print(f"Validation Reason: {trace['validation_reason']}")
                print("="*50)
                
    except Exception as e:
        logger.error(f"Critical execution failure: {e}", exc_info=True)
        err_msg = "Insufficient document evidence to answer reliably due to system execution exception."
        if args.json:
            print(json.dumps({"error": err_msg, "technical_details": str(e)}))
        else:
            print(f"\nSystem Error: {err_msg}", file=sys.stderr)
            print(f"Technical Details: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
