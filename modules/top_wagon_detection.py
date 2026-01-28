"""
TOP CAMERA ANGLE: Wagon Detection Module (Stage 1)

Based on wagon_detection_counting.py but simplified for top view:
- Uses top_detection_best.pt model
- Single ROI box for top camera
- Same tracking & counting logic
- No door detection needed for stage 1
"""

from pathlib import Path
from typing import Optional, Dict, List, Tuple
import cv2
import numpy as np
from collections import defaultdict
import json

from ultralytics import YOLO


class TopWagonDetectionCounter:
    """
    STAGE 1 for TOP Camera Angle: Wagon Detection, Tracking and Counting
    
    Simplified version focusing on top-view wagon detection.
    """
    
    # ROI box for top camera
    ROI_BOX = (300, 100, 900, 500)  # (x1, y1, x2, y2)
    
    def __init__(
        self,
        model_path: str = "models/weights/top_detection_best.pt",
        device: str = "cpu",
        img_size: int = 960,
        conf_threshold: float = 0.15,
        min_frames_in_roi: int = 3,
        output_base_dir: str = "outputs",
        min_confidence_for_storage: float = 0.25,
        min_bbox_area: int = 15000,
        max_frames_per_wagon: int = 10,
        roi_pad_top: int = 5,
        roi_pad_bottom: int = 5,
    ):
        """Initialize top wagon detection system."""
        self.model_path = model_path
        self.device = device
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.min_frames_in_roi = min_frames_in_roi
        
        self.output_base_dir = Path(output_base_dir)
        self.min_confidence_for_storage = min_confidence_for_storage
        self.min_bbox_area = min_bbox_area
        self.max_frames_per_wagon = max_frames_per_wagon
        self.roi_pad_top = roi_pad_top
        self.roi_pad_bottom = roi_pad_bottom
        
        self.roi_box = self.ROI_BOX
        
        print(f"[Top Detection] Loading YOLO model from: {model_path}")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = YOLO(model_path)
        
        # Tracking data
        self.tracks = {}  # track_id -> data
        self.wagon_counter = 0
        self.counted_tracks = set()
        
    def _is_in_roi(self, bbox: Tuple[float, float, float, float]) -> bool:
        """Check if bbox center is inside ROI."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        roi_x1, roi_y1, roi_x2, roi_y2 = self.roi_box
        return roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2
        
    def _crop_wagon(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """Crop wagon from frame with padding."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        
        # Add vertical padding
        pad_h = int((y2 - y1) * self.roi_pad_top / 100)
        pad_b = int((y2 - y1) * self.roi_pad_bottom / 100)
        
        y1 = max(0, y1 - pad_h)
        y2 = min(h, y2 + pad_b)
        x1 = max(0, x1)
        x2 = min(w, x2)
        
        return frame[y1:y2, x1:x2]
        
    def process_video(
        self,
        video_path: str,
        output_video_path: Optional[str] = None,
        on_wagon_counted: Optional[callable] = None,  # Callback for real-time count updates
    ) -> Dict:
        """
        Process video and detect/count wagons from top view.
        
        Returns:
            Dictionary with wagon data and counts
        """
        # Store callback for use later
        self._on_wagon_counted = on_wagon_counted
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
            
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Dynamic ROI: bottom 65% of frame (horizontal line at 35% from top, counting region below)
        # ROI format: (x1, y1, x2, y2) - covers full width, bottom 65% of height
        line_y = int(height * 0.35)  # Line at 35% from top
        self.roi_box = (0, line_y, width, height)
        print(f"[Top Detection] Dynamic ROI (bottom 65%): x1=0, y1={line_y}, x2={width}, y2={height}")
        print(f"[Top Detection] Horizontal line at 35% from top (y={line_y})")
        
        # Output writer
        writer = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
        # Output directories
        stage1_dir = self.output_base_dir / "stage1_top_detections"
        stage1_dir.mkdir(parents=True, exist_ok=True)
        
        frame_idx = 0
        
        print(f"\n[Top Detection] Processing video: {total_frames} frames")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # IMPORTANT: Save original frame BEFORE drawing any annotations
            # This ensures wagon crops don't include ROI box overlays
            original_frame = frame.copy()
                
            frame_idx += 1
            
            # Run detection with tracking (verbose=False to reduce log spam)
            results = self.model.track(
                frame,
                conf=self.conf_threshold,
                imgsz=self.img_size,
                device=self.device,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],  # Only class 0
                verbose=False,  # Reduce log spam
            )
            
            # Draw horizontal line at 35% (green, thicker)
            roi_x1, roi_y1, roi_x2, roi_y2 = self.roi_box
            cv2.line(frame, (0, roi_y1), (width, roi_y1), (0, 255, 0), 3)
            
            # Draw ROI box (yellow)
            cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 2)
            
            # Label the count zone
            cv2.putText(frame, "COUNT ZONE: BOTTOM 65% (TOP)", (roi_x1 + 10, roi_y1 + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if results and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                
                for bbox, track_id, conf in zip(boxes, track_ids, confidences):
                    # Initialize track
                    if track_id not in self.tracks:
                        self.tracks[track_id] = {
                            "frames_in_roi": 0,
                            "best_frames": [],
                            "wagon_id": None,
                        }
                        
                    # Check if in ROI
                    if self._is_in_roi(bbox):
                        self.tracks[track_id]["frames_in_roi"] += 1
                        
                        # Count wagon if meets threshold
                        if (track_id not in self.counted_tracks and 
                            self.tracks[track_id]["frames_in_roi"] >= self.min_frames_in_roi):
                            self.wagon_counter += 1
                            self.counted_tracks.add(track_id)
                            self.tracks[track_id]["wagon_id"] = self.wagon_counter
                            print(f"[Top Detection] Wagon #{self.wagon_counter} detected (track {track_id})")
                            # Call the callback for real-time updates
                            if hasattr(self, '_on_wagon_counted') and self._on_wagon_counted:
                                try:
                                    self._on_wagon_counted(self.wagon_counter)
                                except Exception as e:
                                    print(f"Callback error: {e}")
                            
                    # Store quality frames
                    wagon_id = self.tracks[track_id]["wagon_id"]
                    if wagon_id:
                        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        if conf >= self.min_confidence_for_storage and bbox_area >= self.min_bbox_area:
                            self.tracks[track_id]["best_frames"].append({
                                "frame_idx": frame_idx,
                                "bbox": bbox,
                                "conf": float(conf),
                                "area": float(bbox_area),
                                "frame": original_frame.copy(),  # Use original frame without annotations
                            })
                            
                    # Draw detection
                    x1, y1, x2, y2 = map(int, bbox)
                    color = (0, 255, 0) if track_id in self.counted_tracks else (255, 0, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"Wagon #{self.tracks[track_id]['wagon_id']}" if self.tracks[track_id]['wagon_id'] else f"Track {track_id}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
            # Draw count
            cv2.putText(frame, f"Total Count: {self.wagon_counter}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                       
            if writer:
                writer.write(frame)
                
            if frame_idx % 30 == 0:
                print(f"[Top Detection] Progress: {frame_idx}/{total_frames} | Count: {self.wagon_counter}")
                
        cap.release()
        if writer:
            writer.release()
            
        # Save best frames for each wagon in SAME STRUCTURE as main pipeline
        # Structure: stage1_top_detections/wagon{N}_id_{track_id}/best_frame.jpg
        wagon_data = {}
        for track_id, data in self.tracks.items():
            wagon_id = data["wagon_id"]
            if not wagon_id:
                continue
                
            # Sort by confidence + area to get best frames
            frames = sorted(data["best_frames"], 
                          key=lambda x: x["conf"] * (x["area"] ** 0.5), 
                          reverse=True)[:self.max_frames_per_wagon]
            
            if not frames:
                continue
                
            # Create wagon subdirectory (SAME as main pipeline)
            wagon_dir_name = f"wagon{wagon_id}_id_{track_id}"
            wagon_dir = stage1_dir / wagon_dir_name
            wagon_dir.mkdir(exist_ok=True)
            
            # Save best frame as best_frame.jpg (REQUIRED by Stage 2)
            best_frame_data = frames[0]
            best_crop = self._crop_wagon(best_frame_data["frame"], best_frame_data["bbox"])
            best_frame_path = wagon_dir / "best_frame.jpg"
            cv2.imwrite(str(best_frame_path), best_crop)
            
            # Save additional quality frames
            quality_frames_dir = wagon_dir / "quality_frames"
            quality_frames_dir.mkdir(exist_ok=True)
            
            for idx, frame_data in enumerate(frames, 1):
                crop = self._crop_wagon(frame_data["frame"], frame_data["bbox"])
                frame_path = quality_frames_dir / f"frame{idx:02d}_conf{frame_data['conf']:.2f}.jpg"
                cv2.imwrite(str(frame_path), crop)
                
            wagon_data[wagon_dir_name] = {
                "wagon_id": wagon_id,
                "track_id": int(track_id),
                "num_frames": len(frames),
                "best_frame_path": str(best_frame_path),
                "avg_conf": float(np.mean([f["conf"] for f in frames])),
            }
            
        # Save summary
        summary = {
            "total_wagons": self.wagon_counter,
            "wagon_data": wagon_data,
        }
        
        summary_path = self.output_base_dir / "top_wagon_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
            
        print(f"\n[Top Detection] Complete: {self.wagon_counter} wagons detected")
        return wagon_data
