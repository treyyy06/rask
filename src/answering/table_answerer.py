import re
import google.generativeai as genai
from typing import List, Dict, Any
from ..config import config
from ..logging_config import get_logger

logger = get_logger("answering.table")

class TableAnswerer:
    def __init__(self):
        pass

    def answer(self, question: str, evidence: List[Dict[str, Any]]) -> str:
        """Answers table QA queries structurally by parsing Markdown table coordinates."""
        table_chunks = [e for e in evidence if e["modality"] == "Table"]
        if not table_chunks:
            # Fall back to first available text chunk if no structural table is found
            table_chunks = evidence

        # Assemble table metadata for LLM prompt
        table_mds = []
        for tc in table_chunks:
            table_mds.append(f"--- Page {tc['page']} Table ---\n{tc['text']}")
        table_context = "\n\n".join(table_mds)

        # 1. Use Gemini LLM if API key is active (instructing structural column/row extraction)
        if config.GEMINI_API_KEY:
            try:
                genai.configure(api_key=config.GEMINI_API_KEY)
                model = genai.GenerativeModel(config.VLM_MODEL)
                
                prompt = f"""Use the supplied TABLE EVIDENCE to answer the question.
Locate the correct row and column intersection. Extract the exact value, verify units, and perform any requested calculations (e.g. increases, totals).
Your answer must be concise, accurate, and completely grounded.

TABLES:
{table_context}

QUESTION: {question}
ANSWER:"""
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                logger.error(f"Gemini table answering failed: {e}. Trying structural offline parser.")

        # 2. Offline high-fidelity structural lookup:
        # Search for year indicators (like 2024, 2025) and metric terms (like EPS, Revenue) in the question
        year_match = re.search(r'\b(20\d{2})\b', question)
        target_year = year_match.group(1) if year_match else None
        
        # Look for headers in table chunks
        for tc in table_chunks:
            meta = tc.get("metadata", {})
            headers = meta.get("headers", [])
            rows = meta.get("rows", [])
            
            if not headers or not rows:
                continue
                
            # Attempt to locate target column
            col_idx = -1
            for idx, h in enumerate(headers):
                # Check for match in question text (e.g., EPS, Revenue, Income)
                # Clean up word boundary matches
                h_clean = re.sub(r'[^\w\s]', '', h.lower())
                if any(w in question.lower() for w in h_clean.split() if len(w) > 2):
                    col_idx = idx
                    # Prioritize exact match if possible
                    if h.lower() in question.lower():
                        break
            
            if col_idx == -1:
                # If no matching header, default to first numeric column after Year
                col_idx = 1 if len(headers) > 1 else 0
                
            # Attempt to locate target row matching the year
            target_row = None
            if target_year:
                for r in rows:
                    if r and target_year in r[0]:  # Year is usually first column
                        target_row = r
                        break
            
            # If no target year found, use the first data row
            if not target_row and rows:
                target_row = rows[0]
                
            if target_row and col_idx < len(target_row):
                header_name = headers[col_idx]
                val = target_row[col_idx]
                row_key = target_row[0]
                
                # Check if calculation is requested (e.g., "increase", "difference", "change")
                if any(w in question.lower() for w in ["increase", "decrease", "change", "difference", "compare"]) and len(rows) > 1:
                    try:
                        # Extract first two row values and subtract
                        val1_str = re.sub(r'[^\d\.]', '', rows[0][col_idx])
                        val2_str = re.sub(r'[^\d\.]', '', rows[1][col_idx])
                        val1 = float(val1_str) if val1_str else 0.0
                        val2 = float(val2_str) if val2_str else 0.0
                        diff = abs(val1 - val2)
                        
                        # Preserve units/currency symbols if present
                        unit = "$" if "$" in rows[0][col_idx] else ""
                        suffix = "M" if "M" in rows[0][col_idx] else ("%" if "%" in rows[0][col_idx] else "")
                        
                        return f"The change in {header_name} between {rows[1][0]} and {rows[0][0]} is {unit}{diff:.2f}{suffix} (based on structural table lookup: {rows[0][0]}={rows[0][col_idx]} and {rows[1][0]}={rows[1][col_idx]})."
                    except Exception as e:
                        logger.warning(f"Failed to calculate table difference: {e}")
                
                return f"Table indicates: {row_key} {header_name} is {val}."
                
        # Fallback if structural lookup fails
        for tc in table_chunks:
            lines = [l.strip() for l in tc["text"].split("\n") if l.strip()]
            for l in lines:
                if target_year and target_year in l:
                    return f"Table indicates: {l}"
                    
        return "Insufficient document evidence to answer reliably."
