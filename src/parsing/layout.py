import os
from pathlib import Path
from typing import List, Dict, Any
from ..config import config
from ..logging_config import get_logger

logger = get_logger("parsing.layout")

class LayoutDetector:
    def __init__(self):
        self.yolo_model = None
        self._init_yolo_model()

    def _init_yolo_model(self):
        # Hardware-aware loading of YOLO models
        try:
            from ultralytics import YOLO
            # Load smallest DocLayout YOLO variant or standard nano yolo model
            yolo_path = config.INDEX_DIR.parent / "models" / "yolov8n.pt"
            if not yolo_path.exists():
                logger.info("YOLO weights not found locally. Fallback rules-based layout parser will run.")
                return
            self.yolo_model = YOLO(str(yolo_path))
            logger.info("YOLO Layout Detector initialized successfully.")
        except Exception as e:
            logger.debug(f"YOLO Layout library is not active: {e}. Running fallback rule-based segmenter.")

    def detect_regions(self, image_path: Path, ocr_results: List[Dict[str, Any]], page_num: int) -> List[Dict[str, Any]]:
        """
        Classifies image regions into Text, Table, Figure, Chart.
        Returns:
        {
            "page": int,
            "bbox": [x_min, y_min, x_max, y_max],
            "modality": "Text" | "Table" | "Figure" | "Chart",
            "confidence": float
        }
        """
        regions = []

        # 1. Try YOLO Model if available
        if self.yolo_model:
            try:
                results = self.yolo_model(str(image_path), verbose=False)
                if results and len(results) > 0:
                    for box in results[0].boxes:
                        bbox = [float(val) for val in box.xyxy[0].tolist()]
                        conf = float(box.conf[0])
                        cls_idx = int(box.cls[0])
                        class_name = self.yolo_model.names[cls_idx].lower()
                        
                        # Map classes to standard AQU modalities
                        modality = "Text"
                        if "table" in class_name:
                            modality = "Table"
                        elif "chart" in class_name or "graph" in class_name:
                            modality = "Chart"
                        elif "figure" in class_name or "picture" in class_name or "image" in class_name or "diagram" in class_name:
                            modality = "Figure"
                            
                        regions.append({
                            "page": page_num,
                            "bbox": bbox,
                            "modality": modality,
                            "confidence": conf
                        })
                    if regions:
                        return regions
            except Exception as e:
                logger.error(f"YOLO detection failed: {e}. Trying rules-based fallback.")

        # 2. Rules-based fallback layout parser:
        # Group OCR word boxes by vertical line heights and search for layout gaps.
        if not ocr_results:
            return []

        # Simple grouping heuristic:
        # Tables can be detected if lines contain pipe characters or alignment patterns.
        # Figures are large empty margins.
        # Paragraphs are standard blocks.
        
        # Sort words vertically
        sorted_words = sorted(ocr_results, key=lambda w: (w["bbox"][1], w["bbox"][0]))
        
        # Group into lines
        lines = []
        current_line = []
        curr_y_mid = -1
        
        for w in sorted_words:
            w_y_mid = (w["bbox"][1] + w["bbox"][3]) / 2.0
            if curr_y_mid == -1:
                curr_y_mid = w_y_mid
                current_line.append(w)
            elif abs(w_y_mid - curr_y_mid) < 10:  # Same line threshold
                current_line.append(w)
            else:
                lines.append(current_line)
                current_line = [w]
                curr_y_mid = w_y_mid
        if current_line:
            lines.append(current_line)

        # Detect table indicators in lines
        has_table_indicators = False
        table_line_count = 0
        for l in lines:
            txt_line = " ".join(w["text"] for w in l)
            # Look for table formatting or multiple numeric sequences
            if "|" in txt_line or txt_line.count("  ") > 3 or sum(1 for w in l if w["text"].replace(".", "", 1).isdigit()) > 3:
                table_line_count += 1
                
        # If multiple lines are table-formatted, flag the page as Table
        if table_line_count >= 2:
            has_table_indicators = True

        # Compute page boundaries
        x_mins = [w["bbox"][0] for w in ocr_results]
        y_mins = [w["bbox"][1] for w in ocr_results]
        x_maxs = [w["bbox"][2] for w in ocr_results]
        y_maxs = [w["bbox"][3] for w in ocr_results]
        page_bbox = [min(x_mins), min(y_mins), max(x_maxs), max(y_maxs)]

        if has_table_indicators:
            regions.append({
                "page": page_num,
                "bbox": page_bbox,
                "modality": "Table",
                "confidence": 0.85
            })
        else:
            # Default to Text
            regions.append({
                "page": page_num,
                "bbox": page_bbox,
                "modality": "Text",
                "confidence": 0.90
            })

        return regions
