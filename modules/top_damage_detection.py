"""
TOP CAMERA ANGLE: Damage Detection Module (Stage 4)

Simplified damage detection for top view wagons.
"""

from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np
from collections import defaultdict
import json

from ultralytics import YOLO


class TopDamageDetection:
    """
    STAGE 4 for TOP Camera: Damage Detection from top view
    
    Uses top_damage_best.pt model to detect damage from above.
    """
    
    def __init__(
        self,
        model_path: str = "models/weights/top_damage_best.pt",
        device: str = "cpu",
        conf_threshold: float = 0.25,
        img_size: int = 1280,
        output_base_dir: str = "outputs",
    ):
        """Initialize top damage detection."""
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        self.output_base_dir = Path(output_base_dir)
        
        print(f"[Top Damage] Loading YOLO model from: {model_path}")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = YOLO(model_path)
        
    def process_folder(self, input_dir: Path, on_damage_processed: Optional[callable] = None) -> Dict:
        """
        Process enhanced wagon images and detect damage.
        
        Args:
            input_dir: Directory containing enhanced wagon images (from Stage 3)
            on_damage_processed: Optional callback for progress updates
            
        Returns:
            Dictionary with damage results per wagon
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise ValueError(f"Input directory not found: {input_dir}")
            
        # Create output directory
        output_dir = self.output_base_dir / "stage4_top_damage"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Group images by wagon
        wagon_groups = defaultdict(list)
        for img_path in sorted(input_dir.glob("*.jpg")):
            # Extract wagon identifier from filename
            # Format: wagon1_id_2_enhanced.jpg or wagon1_id_2_frame_3.jpg
            parts = img_path.stem.split("_")
            if len(parts) >= 2:
                wagon_key = f"{parts[0]}_{parts[1]}_{parts[2]}"  # wagon1_id_2
                wagon_groups[wagon_key].append(img_path)
                
        results = {}
        total_wagons = len(wagon_groups)
        
        print(f"\n[Top Damage] Processing {total_wagons} wagons...")
        
        # Counter for callback (cumulative defects)
        running_total_defects = 0
        
        for idx, (wagon_key, image_paths) in enumerate(sorted(wagon_groups.items()), 1):
            print(f"[Top Damage] {idx}/{total_wagons}: {wagon_key}")
            
            # Process all images for this wagon
            all_detections = []
            best_image = None
            best_count = 0
            
            for img_path in image_paths:
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                    
                # Run detection (verbose=False to reduce log spam)
                detection_results = self.model(
                    image,
                    conf=self.conf_threshold,
                    imgsz=self.img_size,
                    device=self.device,
                    classes=[0],  # Only class 0
                    verbose=False,  # Reduce log output
                )
                
                if detection_results and len(detection_results[0].boxes) > 0:
                    boxes = detection_results[0].boxes.xyxy.cpu().numpy()
                    confidences = detection_results[0].boxes.conf.cpu().numpy()
                    
                    for box, conf in zip(boxes, confidences):
                        all_detections.append({
                            "bbox": box.tolist(),
                            "conf": float(conf),
                        })
                        
                    # Track best image
                    if len(boxes) > best_count:
                        best_count = len(boxes)
                        best_image = img_path
                        
            # Create annotated image from best frame
            annotated_path = None
            if best_image and all_detections:
                image = cv2.imread(str(best_image))
                
                # Draw all detections
                for det in all_detections:
                    x1, y1, x2, y2 = map(int, det["bbox"])
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(image, f"{det['conf']:.2f}", (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                              
                annotated_path = output_dir / f"{wagon_key}_damage_annotated.jpg"
                cv2.imwrite(str(annotated_path), image)
                
            # Aggregate results
            results[wagon_key] = {
                "wagon_name": wagon_key,
                "total_detections": len(all_detections),
                "has_damage": len(all_detections) > 0,
                "avg_confidence": float(np.mean([d["conf"] for d in all_detections])) if all_detections else 0.0,
                "max_confidence": float(np.max([d["conf"] for d in all_detections])) if all_detections else 0.0,
                "annotated_image_path": str(annotated_path) if annotated_path else None,
                "detections": all_detections,
            }

            # Update count and callback
            running_total_defects += len(all_detections)
            if on_damage_processed:
                try:
                    on_damage_processed(running_total_defects)
                except Exception as e:
                    pass
            
        # Save summary
        summary = {
            "total_wagons": total_wagons,
            "wagons_with_damage": sum(1 for r in results.values() if r["has_damage"]),
            "total_damage_detections": sum(r["total_detections"] for r in results.values()),
            "results": results,
        }
        
        summary_path = self.output_base_dir / "stage4_top_damage_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
            
        print(f"[Top Damage] Complete: {summary['wagons_with_damage']}/{total_wagons} wagons with damage")
        print(f"[Top Damage] Total damage detections: {summary['total_damage_detections']}")
        
        return results
