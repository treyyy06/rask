import re
import numpy as np
import google.generativeai as genai
from typing import List, Dict, Any, Set, Tuple
from . import config
from .logging_config import get_logger
from .segmentation import extract_candidate_entities

logger = get_logger("retrieval")

def classify_query(query: str) -> Dict[str, bool]:
    """
    Classify the query into different modalities and semantic intents.
    Attributes: text, table, image, sequence, spatial, cross_modal
    """
    q_lower = query.lower()
    
    # Keyword sets
    sequence_keywords = {
        "sequence", "events", "beginning", "middle", "ending", "first", "next", 
        "finally", "before", "after", "develop", "timeline", "progression", 
        "then", "happen next", "start", "end", "panels"
    }
    
    spatial_keywords = {
        "near", "beside", "between", "above", "below", "left of", "right of", 
        "adjacent", "closest", "located", "positioned", "surrounding", "map", 
        "coordinate", "next to", "behind", "in front of"
    }
    
    table_keywords = {
        "table", "revenue", "income", "margin", "eps", "percentage", "rate", 
        "year", "quarter", "fy", "metrics", "financials", "gaap", "non-gaap",
        "net worth", "profit", "cash flow", "balance sheet", "report", "growth",
        "value", "numbers", "statistics", "data"
    }
    
    image_keywords = {
        "looks like", "appearance", "color", "shape", "size", "visible", 
        "shown", "depicted", "wearing", "panels", "panel", "comic", "storyboard",
        "draw", "illustration", "image", "picture", "photo", "dress", "character"
    }
    
    cross_modal_keywords = {
        "text and image", "according to the text and image", "both text and",
        "where in the image", "read the text", "caption"
    }
    
    # Detect flags
    is_sequence = any(w in q_lower for w in sequence_keywords)
    is_spatial = any(w in q_lower for w in spatial_keywords)
    is_table = any(w in q_lower for w in table_keywords)
    is_image = any(w in q_lower for w in image_keywords) or is_spatial or is_sequence
    is_cross_modal = any(w in q_lower for w in cross_modal_keywords) or (is_image and (is_table or "text" in q_lower))
    
    # Default to text
    is_text = True
    
    # Refine visual intent: if query is strictly asking for values/textual details, don't force image
    if is_table and not any(w in q_lower for w in ["visible", "color", "panels", "draw"]):
        is_image = False
        
    logger.info(f"Query Classification: TEXT={is_text}, TABLE={is_table}, IMAGE={is_image}, SEQUENCE={is_sequence}, SPATIAL={is_spatial}, CROSS_MODAL={is_cross_modal}")
    
    return {
        "text": is_text,
        "table": is_table,
        "image": is_image,
        "sequence": is_sequence,
        "spatial": is_spatial,
        "cross_modal": is_cross_modal
    }

def get_text_embedding(text: str) -> List[float]:
    """Retrieves embedding for text from Gemini. Falls back to mock array if unavailable."""
    if not config.GEMINI_API_KEY:
        # Fallback representation (mock vector)
        return [0.0] * 768
        
    try:
        genai.configure(api_key=config.GEMINI_API_KEY)
        # Use text-embedding-004
        result = genai.embed_content(
            model=config.EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        logger.warning(f"Failed to fetch Gemini embedding: {e}. Using fallback mock embedding.")
        return [0.0] * 768

def calculate_cosine_similarity(v1: List[float], v2: List[float]) -> float:
    a = np.array(v1)
    b = np.array(v2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def extract_query_entities(query: str) -> Set[str]:
    """Extract key entity/noun references from the query."""
    # Strip common question words
    q_clean = re.sub(r'^(what|who|where|how|why|when|is|are|the|a|an|describe|explain|show)\b', '', query, flags=re.IGNORECASE)
    words = re.findall(r'\b\w+(?:-\w+)*\b', q_clean.lower())
    stop_words = {
        "what", "where", "happens", "story", "comic", "image", "table", "document", 
        "first", "beginning", "next", "after", "before", "sequence", "events", 
        "between", "about", "there", "their", "from", "with", "than", "near", 
        "color", "panel", "panels", "looks", "shown", "the", "and", "for", "but",
        "are", "was", "were", "this", "that", "these", "those", "have", "has", "had",
        "been", "will", "would", "should", "could", "who", "whom", "whose", "which",
        "how", "why", "can", "could", "may", "might", "must", "look", "looking",
        "find", "finds", "found", "describe", "explain", "show", "shows", "shown",
        "character", "characters", "people", "person", "about", "above", "below",
        "under", "over", "into", "onto", "out", "off", "down", "up", "through",
        "with", "without", "during", "since", "until", "while", "against", "among",
        "again", "further", "then", "once", "here", "there", "when", "all", "any",
        "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "too", "very", "just", "now", "did",
        "does", "doing", "done", "does", "doesnt", "didnt", "isnt", "arent", "wasnt"
    }
    return {w for w in words if w not in stop_words}

class RetrievalEngine:
    def __init__(self, pages_metadata: List[Dict[str, Any]], story_clusters: List[Set[int]]):
        self.pages = pages_metadata
        self.clusters = story_clusters
        
        # Precompute embeddings if API key is present
        self.page_embeddings = {}
        self.precompute_embeddings()

    def precompute_embeddings(self):
        """Precompute text embeddings for all pages."""
        if not config.GEMINI_API_KEY:
            logger.info("No Gemini API key found. Skipping embedding precomputation. Fallback Jaccard will be used.")
            return
            
        for page_data in self.pages:
            p_num = page_data["page"]
            text = page_data.get("text", "")
            if text.strip():
                self.page_embeddings[p_num] = get_text_embedding(text)
                
    def retrieve_evidence(self, query: str) -> List[Dict[str, Any]]:
        """
        Executes hybrid, entity-anchored evidence retrieval.
        Returns pages ranked by unified score.
        """
        # 1. Classify query modality and extract query entities
        modality_flags = classify_query(query)
        query_entities = extract_query_entities(query)
        logger.info(f"Extracted Query Entities: {query_entities}")
        
        # Calculate Query Embedding (if key available)
        query_emb = None
        if config.GEMINI_API_KEY and self.page_embeddings:
            try:
                genai.configure(api_key=config.GEMINI_API_KEY)
                res = genai.embed_content(
                    model=config.EMBEDDING_MODEL,
                    content=query,
                    task_type="retrieval_query"
                )
                query_emb = res['embedding']
            except Exception as e:
                logger.warning(f"Failed to fetch query embedding: {e}")
                
        # 2. Document-Level Entity Retrieval: Identify pages matching entities
        entity_matches = {}
        matched_any_entity = False
        for page_data in self.pages:
            p_num = page_data["page"]
            text_lower = page_data.get("text", "").lower()
            
            matches = 0
            for ent in query_entities:
                # Direct word match (using boundary word regex for exact matching)
                if re.search(r'\b' + re.escape(ent) + r'\b', text_lower):
                    matches += 1
            
            entity_matches[p_num] = matches / len(query_entities) if query_entities else 0.0
            if matches > 0:
                matched_any_entity = True
                
        # 3. Story Continuity Alignment:
        # Find which story clusters contain the highest entity matching pages
        cluster_scores = []
        for cluster in self.clusters:
            cluster_match_sum = sum(entity_matches.get(p, 0.0) for p in cluster)
            # Normalize by cluster size to prevent large clusters from dominating unfairly
            cluster_scores.append(cluster_match_sum / len(cluster) if cluster else 0.0)
            
        best_cluster_idx = int(np.argmax(cluster_scores)) if cluster_scores else -1
        best_cluster_pages = self.clusters[best_cluster_idx] if best_cluster_idx != -1 else set()
        
        # 4. Text/Visual/Modality Scoring
        evidence_list = []
        for page_data in self.pages:
            p_num = page_data["page"]
            text = page_data.get("text", "")
            
            # Entity Score
            ent_score = entity_matches.get(p_num, 0.0)
            
            # Text Similarity (Cosine Similarity or Jaccard Fallback)
            text_sim = 0.0
            if query_emb and p_num in self.page_embeddings:
                text_sim = calculate_cosine_similarity(query_emb, self.page_embeddings[p_num])
            else:
                # Fallback: Jaccard word similarity
                q_words = set(re.findall(r'\b\w+(?:-\w+)*\b', query.lower()))
                p_words = set(re.findall(r'\b\w+(?:-\w+)*\b', text.lower()))
                if q_words and p_words:
                    text_sim = len(q_words.intersection(p_words)) / len(q_words.union(p_words))
                    
            # Story Continuity Score
            in_best_story = 1.0 if p_num in best_cluster_pages else 0.0
            
            # Visual Similarity Fallback (VLM queries might weight this, but we prefer text rules first)
            visual_sim = 0.0
            if modality_flags["image"]:
                # If page is flagged for visual fallback, boost its visual match score
                if page_data.get("use_visual_fallback", False):
                    visual_sim = 0.8
                else:
                    # Normal page, visual overlap is estimated based on text representation of panels
                    visual_sim = text_sim
                    
            # Modality Match Score:
            # Check if query asks for a table, and this page contains tables
            has_table = "[---]" in text or "| " in text or ("table_index" in page_data)
            modality_score = 0.0
            if modality_flags["table"] and has_table:
                modality_score = 1.0
            elif modality_flags["image"] and page_data.get("image_path"):
                modality_score = 0.5
            elif not modality_flags["image"] and not modality_flags["table"]:
                modality_score = 0.8  # Text match
                
            # Unified Score Calculation:
            # Score = 0.35 * entity + 0.25 * text_sim + 0.20 * story + 0.10 * visual + 0.10 * modality
            final_score = (
                0.35 * ent_score +
                0.25 * text_sim +
                0.20 * in_best_story +
                0.10 * visual_sim +
                0.10 * modality_score
            )
            
            # Enforce strict entity matching hierarchy:
            # Boost score if we have exact entity match or same-story overlap
            if matched_any_entity:
                if ent_score > 0:
                    final_score += 0.2  # Exact entity boost
                elif in_best_story > 0:
                    final_score += 0.1  # Same story boost
                else:
                    final_score -= 0.1  # Deduct if completely unrelated story
                    
            # Cap final score
            final_score = max(0.0, min(1.0, final_score))
            
            # Generate selection reason
            reason_parts = []
            if ent_score > 0:
                reason_parts.append("entity match")
            if in_best_story > 0:
                reason_parts.append("same story cluster")
            if modality_score > 0.5:
                reason_parts.append("modality match")
                
            reason = " + ".join(reason_parts) if reason_parts else "semantic text similarity"
            
            evidence_list.append({
                "page": p_num,
                "text_content": text,
                "text_score": float(text_sim),
                "visual_score": float(visual_sim),
                "entity_score": float(ent_score),
                "story_score": float(in_best_story),
                "modality_score": float(modality_score),
                "final_score": float(final_score),
                "reason": reason,
                "image_path": page_data.get("image_path")
            })
            
        # Sort by final score descending
        evidence_list.sort(key=lambda x: x["final_score"], reverse=True)
        
        # 5. Correct Ordering for Multi-Page Sequence Questions:
        # If sequence query, retrieve the top K candidate pages, but sort them sequentially by page number
        top_k = min(config.TOP_K, len(evidence_list))
        retrieved_pages = evidence_list[:top_k]
        
        if modality_flags["sequence"]:
            # Sort by page number ascending
            retrieved_pages.sort(key=lambda x: x["page"])
            logger.info("Sequence query detected. Output sorted chronologically by page number.")
            
        for r in retrieved_pages:
            logger.info(f"Retrieved Page {r['page']} - Score: {r['final_score']:.3f} | Reason: {r['reason']}")
            
        return retrieved_pages
