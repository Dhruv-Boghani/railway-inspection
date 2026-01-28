"""
STAGE 1: Wagon Detection, Tracking & Counting Module

Outputs: High-quality Wagon ROIs (only from counting zone) for downstream processing
"""

from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime


@dataclass
class WagonFrame:
    """Single frame data for a wagon"""
    track_id: int
    frame_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[int, int]          # (cx, cy)
    confidence: float
    roi_image: np.ndarray            # Cropped wagon ROI
    bbox_area: int                   # Area of bounding box
    distance_score: float            # How close to camera (larger bbox = closer)
    is_counted_frame: bool           # True if this is the frame where wagon was counted
    quality_score: float             # Overall quality score for this frame


@dataclass
class WagonData:
    """Complete data for one wagon across all frames"""
    wagon_number: int               # Sequential number (1, 2, 3...)
    track_id: int                   # ByteTrack ID (1, 5, 10, 23...)
    frames: List[WagonFrame]        # Selected high-quality frames from ROI zone
    counted_at_frame: int           # Frame number where counted
    total_frames: int               # Total frames this wagon appeared
    best_frame_idx: int             # Index of best quality frame in frames list
    best_frame: WagonFrame          # The best quality frame for this wagon


class DoorChecker:
    """
    Lightweight door detector used ONLY on the 5-10 candidate ROIs per wagon.
    It returns (num_doors, num_good_doors, score) for each ROI.
    """

    def __init__(
        self,
        model_path: str,
        door_class_ids: Optional[list] = None,
        device: str = "cpu",
        conf_threshold: float = 0.25,
        edge_margin_ratio: float = 0.10,  # 10% margin on each side
    ):
        """
        model_path: weights for door detection model
        door_class_ids: list of class indices that correspond to doors
        edge_margin_ratio: fraction of width/height treated as edge band
        """
        print(f"Loading door detection model from: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(device)

        self.door_class_ids = door_class_ids
        self.conf_threshold = conf_threshold
        self.device = device
        self.edge_margin_ratio = edge_margin_ratio

    def count_and_score_doors(
        self,
        roi_image: np.ndarray,
    ) -> Tuple[int, int, float]:
        """
        Returns (num_doors, num_good_doors, score) for this ROI.

        - num_doors: number of door detections (after class filter).
        - num_good_doors: doors whose CENTER lies away from edges.
        - score: heuristic combining good/total doors.
        """
        if roi_image is None or roi_image.size == 0:
            return 0, 0, 0.0

        h, w = roi_image.shape[:2]
        if h == 0 or w == 0:
            return 0, 0, 0.0

        results = self.model.predict(
            roi_image,
            conf=self.conf_threshold,
            imgsz=max(h, w),
            device=self.device,
            verbose=False,
        )
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return 0, 0, 0.0

        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        num_doors = 0
        num_good_doors = 0

        # define inner “safe” region (doors here = well inside ROI)
        x_min = w * self.edge_margin_ratio
        x_max = w * (1 - self.edge_margin_ratio)
        y_min = h * self.edge_margin_ratio
        y_max = h * (1 - self.edge_margin_ratio)

        for box, cls_id in zip(boxes, classes):
            if self.door_class_ids is not None and int(cls_id) not in self.door_class_ids:
                continue

            x1, y1, x2, y2 = box
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)

            num_doors += 1

            # “good door” = center well away from ROI border
            if x_min <= cx <= x_max and y_min <= cy <= y_max:
                num_good_doors += 1

        # heuristic: reward good doors most, total doors a bit
        score = num_good_doors * 1.0 + num_doors * 0.25
        return num_doors, num_good_doors, float(score)


class WagonDetectionCounter:
    """
    STAGE 1: Wagon Detection, Tracking and Counting System

    Pipeline Output:
    - Only wagon ROI crops (not full frames)
    - Only frames from counting zone (ROI box)
    - Only high-quality frames (5-10 per wagon)
    - Clear view + whole wagon visible + near to camera
    """

    # ROI will be calculated dynamically based on frame dimensions
    # LR: Right half (center vertical line to right edge)
    # RL: Left half (left edge to center vertical line)
    ROI_BOXES = {
        "LR": None,  # Will be calculated as right half of frame
        "RL": None,  # Will be calculated as left half of frame
    }

    def __init__(
        self,
        model_path: str,
        direction: str = "LR",
        device: str = "cpu",
        img_size: int = 960,
        conf_threshold: float = 0.15,
        min_frames_in_roi: int = 3,
        roi_padding: int = 15,  # for config summary / backward compatibility
        output_base_dir: str = "outputs",
        # Quality filtering parameters
        min_confidence_for_storage: float = 0.25,  # Only store frames with conf > this
        min_bbox_area: int = 15000,                # Minimum bbox area for quality frames
        max_frames_per_wagon: int = 10,            # Store max 10 best frames per wagon
        # Padding for crop (no left/right padding as per requirement)
        roi_pad_left_right: int = 0,               # keep 0: no horizontal padding
        roi_pad_top: int = 5,                      # % extra above bbox
        roi_pad_bottom: int = 5,                   # % extra below bbox
        # Door detection for best-frame refinement
        door_model_path: Optional[str] = None,
        door_class_ids: Optional[list] = None,
        door_conf_threshold: float = 0.25,
    ):
        """
        Initialize wagon detection and counting system
        """
        self.model_path = model_path
        self.direction = direction.upper()
        self.device = device
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.min_frames_in_roi = min_frames_in_roi
        self.roi_padding = roi_padding

        self.roi_pad_left_right = roi_pad_left_right
        self.roi_pad_top = roi_pad_top
        self.roi_pad_bottom = roi_pad_bottom

        self.output_base_dir = Path(output_base_dir)
        self.min_confidence_for_storage = min_confidence_for_storage
        self.min_bbox_area = min_bbox_area
        self.max_frames_per_wagon = max_frames_per_wagon

        if self.direction not in self.ROI_BOXES:
            raise ValueError(f"Direction must be 'LR' or 'RL', got '{direction}'")
        # ROI will be set dynamically when processing video based on frame dimensions
        self.roi_box = None

        print(f"Loading YOLOv12s model from: {model_path}")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = YOLO(model_path)
        print("Wagon model loaded successfully")

        # Door checker
        self.door_checker: Optional[DoorChecker] = None
        if door_model_path is not None:
            if door_class_ids is None:
                door_class_ids = [0]
            self.door_checker = DoorChecker(
                model_path=door_model_path,
                door_class_ids=door_class_ids,
                device=device,
                conf_threshold=door_conf_threshold,
            )

        self.counted_ids = set()
        self.inside_roi_counter = defaultdict(int)
        self.last_centers: Dict[int, Tuple[int, int]] = {}
        self.wagon_log = []
        self.wagon_frames_by_id: Dict[int, List[WagonFrame]] = defaultdict(list)
        self.counted_frames: Dict[int, int] = {}
        self.wagon_counter = 0
        self.track_id_to_wagon_number: Dict[int, int] = {}
        self.frame_idx = 0

        self.create_output_dirs()

    def create_output_dirs(self):
        self.output_dirs = {
            "wagon_rois": self.output_base_dir / "stage1_wagon_rois",
            "annotated_frames": self.output_base_dir / "stage1_annotated_frames",
            "results": self.output_base_dir / "stage1_results",
            "videos": self.output_base_dir / "videos",
        }
        for dir_path in self.output_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Output directories created at: {self.output_base_dir}")

    def reset(self):
        self.counted_ids = set()
        self.inside_roi_counter = defaultdict(int)
        self.last_centers = {}
        self.wagon_log = []
        self.wagon_frames_by_id = defaultdict(list)
        self.counted_frames = {}
        self.wagon_counter = 0
        self.track_id_to_wagon_number = {}
        self.frame_idx = 0
        print("Tracking state reset")

    def _check_direction(self, tid: int, cx: int) -> bool:
        if tid not in self.last_centers:
            return True
        prev_cx = self.last_centers[tid][0]
        if self.direction == "LR":
            return cx > prev_cx
        else:
            return cx < prev_cx

    def _is_inside_roi(self, cx: int, cy: int) -> bool:
        if self.roi_box is None:
            return False
        x1, y1, x2, y2 = self.roi_box
        return (x1 <= cx <= x2 and y1 <= cy <= y2)

    def _calculate_roi_from_frame(self, frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
        """
        Calculate ROI based on direction:
        - LR: Vertical line at 35% from left, counting zone = RIGHT 65%
        - RL: Vertical line at 65% from left, counting zone = LEFT 65%
        """
        line_position_lr = int(frame_width * 0.35)  # 35% from left for LR
        line_position_rl = int(frame_width * 0.65)  # 65% from left for RL
        y_top = 0
        y_bottom = frame_height
        
        if self.direction == "LR":
            # Right 65%: from 35% line to right edge
            return (line_position_lr, y_top, frame_width, y_bottom)
        else:  # RL
            # Left 65%: from left edge to 65% line
            return (0, y_top, line_position_rl, y_bottom)

    def _extract_wagon_roi(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract wagon ROI exactly around the detection box with only vertical padding.
        No left/right padding (as requested).
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1

        # no horizontal padding
        pad_lr = int(w * self.roi_pad_left_right / 100)
        pad_top = int(h * self.roi_pad_top / 100)
        pad_bottom = int(h * self.roi_pad_bottom / 100)

        x1_crop = max(0, x1 - pad_lr)
        x2_crop = min(frame.shape[1], x2 + pad_lr)
        y1_crop = max(0, y1 - pad_top)
        y2_crop = min(frame.shape[0], y2 + pad_bottom)

        if x2_crop <= x1_crop or y2_crop <= y1_crop:
            x1_crop, y1_crop, x2_crop, y2_crop = x1, y1, x2, y2

        roi = frame[y1_crop:y2_crop, x1_crop:x2_crop].copy()
        return roi

    def _calculate_distance_score(self, bbox: Tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        max_area = 200000
        score = min(1.0, area / max_area)
        return score

    def _calculate_quality_score(
        self,
        confidence: float,
        bbox_area: int,
        distance_score: float,
        is_counted: bool,
    ) -> float:
        area_score = min(1.0, bbox_area / 200000)
        quality = (
            confidence * 0.3
            + distance_score * 0.4
            + area_score * 0.2
            + (1.0 if is_counted else 0.0) * 0.1
        )
        return quality

    def _should_store_frame(self, confidence: float, bbox_area: int, inside_roi: bool) -> bool:
        if not inside_roi:
            return False
        if confidence < self.min_confidence_for_storage:
            return False
        if bbox_area < self.min_bbox_area:
            return False
        return True

    def _center_alignment_score(self, bbox: Tuple[int, int, int, int]) -> float:
        """
        How centered the wagon bbox is inside the counting ROI.
        1.0 = perfectly centered; 0 = far from center.
        """
        bx1, by1, bx2, by2 = bbox
        cx = 0.5 * (bx1 + bx2)
        cy = 0.5 * (by1 + by2)

        rx1, ry1, rx2, ry2 = self.roi_box
        rcx = 0.5 * (rx1 + rx2)
        rcy = 0.5 * (ry1 + ry2)

        rw = rx2 - rx1
        rh = ry2 - ry1
        if rw <= 0 or rh <= 0:
            return 0.0

        dx = abs(cx - rcx) / rw
        dy = abs(cy - rcy) / rh

        # normalize: closer to center -> higher score
        score = 1.0 - min(1.0, np.sqrt(dx * dx + dy * dy) * 2.0)
        return float(max(0.0, score))

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        # IMPORTANT: Save original frame BEFORE drawing any annotations
        # This ensures wagon ROI crops don't include counting box overlays
        original_frame = frame.copy()
        
        results = self.model.track(
            source=frame,
            conf=self.conf_threshold,
            imgsz=self.img_size,
            tracker="bytetrack.yaml",
            persist=True,
            device=self.device,
            verbose=False,
        )

        r = results[0]

        # Draw ROI box and vertical line at ROI boundary (on annotated frame only, NOT original)
        if self.roi_box is not None:
            x1, y1, x2, y2 = self.roi_box
            h = frame.shape[0]
            w = frame.shape[1]
            
            # Draw vertical line at ROI boundary (65%/35% position)
            # For LR: line at 65% from left (x1 of ROI)
            # For RL: line at 35% from left (x2 of ROI)
            line_x = x1 if self.direction == "LR" else x2
            cv2.line(frame, (line_x, 0), (line_x, h), (0, 255, 0), 3)
            
            # Draw ROI box (yellow)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            # Label the count zone with percentage
            zone_side = "RIGHT 65%" if self.direction == "LR" else "LEFT 65%"
            cv2.putText(
                frame,
                f"COUNT ZONE: {zone_side} ({self.direction})",
                (x1 + 10, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        if r.boxes.id is not None:
            for box, tid, conf in zip(r.boxes.xyxy, r.boxes.id, r.boxes.conf):
                tid = int(tid)
                bx1, by1, bx2, by2 = map(int, box)
                confidence = float(conf)

                cx = int((bx1 + bx2) / 2)
                cy = int((by1 + by2) / 2)

                if not self._check_direction(tid, cx):
                    continue

                self.last_centers[tid] = (cx, cy)
                inside_roi = self._is_inside_roi(cx, cy)

                if inside_roi:
                    self.inside_roi_counter[tid] += 1
                else:
                    self.inside_roi_counter[tid] = 0

                newly_counted = False
                if (
                    self.inside_roi_counter[tid] == self.min_frames_in_roi
                    and tid not in self.counted_ids
                ):
                    self.counted_ids.add(tid)
                    self.wagon_counter += 1
                    self.track_id_to_wagon_number[tid] = self.wagon_counter
                    self.wagon_log.append((tid, self.frame_idx))
                    self.counted_frames[tid] = self.frame_idx
                    newly_counted = True
                    print(
                        f"COUNTED wagon#{self.wagon_counter} "
                        f"(track_id={tid}) at frame {self.frame_idx}"
                    )
                    # Call the callback for real-time updates
                    if hasattr(self, '_on_wagon_counted') and self._on_wagon_counted:
                        try:
                            self._on_wagon_counted(self.wagon_counter)
                        except Exception as e:
                            print(f"Callback error: {e}")

                bbox = (bx1, by1, bx2, by2)
                bbox_area = (bx2 - bx1) * (by2 - by1)
                distance_score = self._calculate_distance_score(bbox)

                if self._should_store_frame(confidence, bbox_area, inside_roi):
                    # Extract ROI from ORIGINAL frame (without annotations)
                    wagon_roi = self._extract_wagon_roi(original_frame, bbox)
                    quality_score = self._calculate_quality_score(
                        confidence, bbox_area, distance_score, newly_counted
                    )

                    wagon_frame = WagonFrame(
                        track_id=tid,
                        frame_id=self.frame_idx,
                        bbox=bbox,
                        center=(cx, cy),
                        confidence=confidence,
                        roi_image=wagon_roi,
                        bbox_area=bbox_area,
                        distance_score=distance_score,
                        is_counted_frame=newly_counted,
                        quality_score=quality_score,
                    )
                    self.wagon_frames_by_id[tid].append(wagon_frame)

                color = (0, 255, 0) if newly_counted else (255, 0, 0)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                label = f"ID:{tid} ({confidence:.2f})"
                if newly_counted:
                    wagon_num = self.track_id_to_wagon_number[tid]
                    label = f"WAGON#{wagon_num} (ID:{tid})"
                cv2.putText(
                    frame,
                    label,
                    (bx1, by1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

        count_text = f"Total Count: {len(self.counted_ids)}"
        cv2.putText(
            frame,
            count_text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
        )

        self.frame_idx += 1
        return frame, len(self.counted_ids)

    def _select_best_frames(self, frames: List[WagonFrame], max_frames: int) -> List[WagonFrame]:
        if not frames:
            return []
        sorted_frames = sorted(frames, key=lambda f: f.quality_score, reverse=True)
        best_frames = sorted_frames[:max_frames]
        if not best_frames:
            return []
        best_frame = max(best_frames, key=lambda f: f.quality_score)
        other_frames = [f for f in best_frames if f is not best_frame]
        other_frames = sorted(other_frames, key=lambda f: f.frame_id)
        return [best_frame] + other_frames

    def _refine_best_frame_with_doors(
        self,
        wagon_frames: List[WagonFrame],
    ) -> Tuple[int, WagonFrame]:
        """
        Final best-frame selection combining:

        1) Wagon detection box centered in counting ROI and maximal width.
        2) Doors as centered as possible with at least 2 detected doors preferred.

        Strategy:
        - Compute for each frame:
          - door_info: (num_doors, num_good_doors, door_score)
          - wagon_center_score: how centered bbox is in ROI
          - wagon_width: width of bbox
        - First, only consider frames with num_doors >= 2.
        - Among them, pick by key:
            (num_good_doors, num_doors, wagon_center_score, wagon_width, quality_score)
        - If no frame has >= 2 doors:
            fall back to best by:
            (wagon_center_score, wagon_width, quality_score)
        """
        if not wagon_frames:
            return 0, wagon_frames[0]

        # Precompute geometric scores
        center_scores = [self._center_alignment_score(wf.bbox) for wf in wagon_frames]
        widths = [(wf.bbox[2] - wf.bbox[0]) for wf in wagon_frames]

        # If no door checker, just use geometry-based fallback
        if self.door_checker is None:
            best_idx = max(
                range(len(wagon_frames)),
                key=lambda i: (center_scores[i], widths[i], wagon_frames[i].quality_score),
            )
            return best_idx, wagon_frames[best_idx]

        # With door checker: compute door info per frame
        door_infos = []
        for wf in wagon_frames:
            num_doors, num_good_doors, score = self.door_checker.count_and_score_doors(
                wf.roi_image
            )
            door_infos.append((num_doors, num_good_doors, score))

        # 1) Prefer frames with at least 2 doors
        candidates = []
        for idx, wf in enumerate(wagon_frames):
            n, g, s = door_infos[idx]
            if n >= 2:
                key = (
                    g,                          # more good doors
                    n,                          # more total doors
                    center_scores[idx],         # more centered wagon
                    widths[idx],                # wider wagon
                    wf.quality_score,           # higher frame quality
                )
                candidates.append((key, idx))

        if candidates:
            candidates.sort(reverse=True, key=lambda x: x[0])
            best_idx = candidates[0][1]
            return best_idx, wagon_frames[best_idx]

        # 2) No frame with >= 2 doors -> fall back to pure geometry + quality
        best_idx = max(
            range(len(wagon_frames)),
            key=lambda i: (center_scores[i], widths[i], wagon_frames[i].quality_score),
        )
        return best_idx, wagon_frames[best_idx]

    def save_pipeline_outputs(self, video_name: str = "video") -> Dict[int, WagonData]:
        print(f"\nSaving pipeline outputs for {len(self.counted_ids)} wagons...")

        wagon_data_dict: Dict[int, WagonData] = {}

        for track_id in sorted(self.counted_ids):
            frames = self.wagon_frames_by_id[track_id]
            if not frames:
                continue

            wagon_number = self.track_id_to_wagon_number[track_id]
            best_frames = self._select_best_frames(frames, self.max_frames_per_wagon)
            if not best_frames:
                continue

            # Stage-1 best list based on quality
            # Stage-2: refine using door + geometry logic
            refined_idx, refined_frame = self._refine_best_frame_with_doors(best_frames)
            best_frame_idx = refined_idx
            best_frame = refined_frame

            wagon_dir_name = f"wagon{wagon_number}_id_{track_id}"
            wagon_dir = self.output_dirs["wagon_rois"] / wagon_dir_name
            wagon_dir.mkdir(exist_ok=True)

            best_roi_path = wagon_dir / "best_frame.jpg"
            cv2.imwrite(str(best_roi_path), best_frame.roi_image)

            quality_frames_dir = wagon_dir / "quality_frames"
            quality_frames_dir.mkdir(exist_ok=True)

            for idx, frame in enumerate(best_frames, 1):
                frame_filename = f"frame{idx:02d}_fid{frame.frame_id:06d}_q{frame.quality_score:.2f}.jpg"
                frame_path = quality_frames_dir / frame_filename
                cv2.imwrite(str(frame_path), frame.roi_image)

            wagon_data = WagonData(
                wagon_number=wagon_number,
                track_id=track_id,
                frames=best_frames,
                counted_at_frame=self.counted_frames.get(track_id, -1),
                total_frames=len(best_frames),
                best_frame_idx=best_frame_idx,
                best_frame=best_frame,
            )
            wagon_data_dict[track_id] = wagon_data

            metadata = {
                "wagon_number": wagon_number,
                "track_id": track_id,
                "counted_at_frame": wagon_data.counted_at_frame,
                "total_quality_frames": wagon_data.total_frames,
                "best_frame_id": best_frame.frame_id,
                "best_frame_confidence": float(best_frame.confidence),
                "best_frame_bbox_area": best_frame.bbox_area,
                "best_frame_distance_score": float(best_frame.distance_score),
                "best_frame_quality_score": float(best_frame.quality_score),
                "quality_frame_ids": [f.frame_id for f in best_frames],
                "quality_scores": [float(f.quality_score) for f in best_frames],
                "best_roi_path": str(best_roi_path),
                "quality_frames_dir": str(quality_frames_dir),
            }
            metadata_path = wagon_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            print(
                f"  Wagon#{wagon_number} (ID:{track_id}): "
                f"{len(best_frames)} quality frames saved (best idx={best_frame_idx})"
            )

        summary = {
            "video_name": video_name,
            "processing_timestamp": datetime.now().isoformat(),
            "total_wagons_counted": len(self.counted_ids),
            "total_frames_processed": self.frame_idx,
            "direction": self.direction,
            "roi_box": self.roi_box,
            "wagon_mapping": [
                {
                    "wagon_number": self.track_id_to_wagon_number[tid],
                    "track_id": tid,
                    "counted_at_frame": self.counted_frames.get(tid, -1),
                }
                for tid in sorted(self.counted_ids)
            ],
            "config": {
                "model_path": self.model_path,
                "conf_threshold": self.conf_threshold,
                "min_frames_in_roi": self.min_frames_in_roi,
                "roi_padding": self.roi_padding,
                "roi_pad_left_right": self.roi_pad_left_right,
                "roi_pad_top": self.roi_pad_top,
                "roi_pad_bottom": self.roi_pad_bottom,
                "min_confidence_for_storage": self.min_confidence_for_storage,
                "min_bbox_area": self.min_bbox_area,
                "max_frames_per_wagon": self.max_frames_per_wagon,
            },
        }

        summary_path = self.output_dirs["results"] / "stage1_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("\nPipeline outputs saved:")
        print(f" - Wagon ROIs: {self.output_dirs['wagon_rois']}")
        print(f" - Summary: {summary_path}")

        return wagon_data_dict

    def process_video(
        self,
        video_path: str,
        output_video_path: Optional[str] = None,
        save_annotated_frames: bool = False,
        on_wagon_counted: Optional[callable] = None,  # Callback for real-time count updates
    ) -> Dict[int, WagonData]:
        # Store callback for use in process_frame
        self._on_wagon_counted = on_wagon_counted
        self.reset()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print("\nVideo Properties:")
        print(f"  Resolution: {w}x{h}")
        print(f"  FPS: {fps}")
        print(f"  Total Frames: {total_frames}")
        print(f"  Direction: {self.direction}")

        # Calculate ROI dynamically based on frame dimensions
        self.roi_box = self._calculate_roi_from_frame(w, h)
        print(f"  ROI Box (calculated): {self.roi_box}")
        print(f"  ROI Logic: {self.direction} uses {'RIGHT' if self.direction == 'LR' else 'LEFT'} half of frame")

        out = None
        if output_video_path:
            Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
            # Use H.264 codec (avc1) for web browser compatibility
            # This ensures the video can be streamed and played in HTML5 video players
            out = cv2.VideoWriter(
                output_video_path,
                cv2.VideoWriter_fourcc(*"avc1"),  # H.264 codec for web compatibility
                fps,
                (w, h),
            )

        print("\nProcessing video...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame, current_count = self.process_frame(frame)

            if out:
                out.write(annotated_frame)

            if save_annotated_frames and self.frame_idx % 30 == 0:
                frame_path = self.output_dirs["annotated_frames"] / f"frame_{self.frame_idx:06d}.jpg"
                cv2.imwrite(str(frame_path), annotated_frame)

            if self.frame_idx % 100 == 0:
                progress = (self.frame_idx / max(1, total_frames)) * 100
                print(
                    f"  Progress: {progress:.1f}% | "
                    f"Frame: {self.frame_idx}/{total_frames} | "
                    f"Count: {current_count}"
                )

        cap.release()
        if out:
            out.release()

        print("\nVideo processing complete!")
        print(f"  Total wagons counted: {len(self.counted_ids)}")
        print(f"  Frames processed: {self.frame_idx}")

        video_name = Path(video_path).stem
        wagon_data_dict = self.save_pipeline_outputs(video_name)
        return wagon_data_dict


# ================= STANDALONE TEST =================

if __name__ == "__main__":
    MODEL_PATH = r"C:\Users\Viranch Patel\Desktop\Wagon_Inspection_System\models\weights\yolo12s_wagon_detection_&_counting_best.pt"
    VIDEO_PATH = r"C:\Users\Viranch Patel\Desktop\Wagon_Inspection_System\inputs\test5.mp4"
    OUTPUT_VIDEO_PATH = "outputs/videos/annotated_output5.mp4"
    DIRECTION = "RL"

    # Single-class door model: class 0 = door
    DOOR_MODEL_PATH = r"C:\Users\Viranch Patel\Desktop\Wagon_Inspection_System\models\weights\yolo12n_Door_ROI_best.pt"
    DOOR_CLASS_IDS = [0]

    detector = WagonDetectionCounter(
        model_path=MODEL_PATH,
        direction=DIRECTION,
        device="cuda",
        conf_threshold=0.15,
        min_frames_in_roi=3,
        roi_padding=15,
        output_base_dir="outputs",
        min_confidence_for_storage=0.25,
        min_bbox_area=15000,
        max_frames_per_wagon=10,
        roi_pad_left_right=0,   # no horizontal padding
        roi_pad_top=5,
        roi_pad_bottom=5,
        door_model_path=DOOR_MODEL_PATH,
        door_class_ids=DOOR_CLASS_IDS,
        door_conf_threshold=0.25,
    )

    wagon_data = detector.process_video(
        video_path=VIDEO_PATH,
        output_video_path=OUTPUT_VIDEO_PATH,
        save_annotated_frames=False,
    )

    print("\n" + "=" * 60)
    print("STAGE 1 OUTPUT - WAGON DATA FOR NEXT STAGE")
    print("=" * 60)
    for track_id, data in wagon_data.items():
        print(f"\nWagon#{data.wagon_number} (track_id={track_id}):")
        print(f"  - Quality frames stored: {data.total_frames}")
        print(f"  - Counted at frame: {data.counted_at_frame}")
        print(f"  - Best frame: {data.best_frame.frame_id}")
        print(f"  - Best frame quality score: {data.best_frame.quality_score:.3f}")
        print(f"  - Best frame confidence: {data.best_frame.confidence:.3f}")
        print(f"  - Best frame bbox area: {data.best_frame.bbox_area}")
