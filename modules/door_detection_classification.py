"""
STAGE 4: Door Detection & Classification Module

1. Detects door ROIs from enhanced wagon images (YOLOv12n)
2. Classifies doors as Good/Damaged/Missing (EfficientNet Few-Shot)
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO
import timm
from torchvision import transforms


# ===================== DOOR CLASSIFIER MODEL =====================

class EfficientNetDoorClassifier(nn.Module):
    """EfficientNet-B0 based door classifier (3 classes: good, damaged, missing)"""

    def __init__(self, num_classes=3, pretrained: bool = False):
        super().__init__()
        # Load EfficientNet-B0 backbone
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=0
        )
        # Get feature dimension
        feature_dim = self.backbone.num_features  # 1280 for EfficientNet-B0
        # Classification head with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


# ===================== MAIN STAGE 4 CLASS =====================

class DoorDetectionClassification:
    """
    STAGE 4: Door Detection & Classification System

    Pipeline:
    1. Load enhanced wagon ROIs from STAGE 3
    2. Detect door regions using YOLOv12n
    3. Filter invalid/spurious detections (size-based filtering)
    4. Merge overlapping doors so each physical door is counted once
    5. Classify each door as Good/Damaged/Missing using EfficientNet
    6. Save door ROIs and classification results
    """

    def __init__(
        self,
        stage3_output_dir: str = "outputs/stage3_enhanced_frames",
        output_base_dir: str = "outputs",
        # Door detection model
        door_detection_model_path: str = "models/weights/yolo12n_Door_ROI_best.pt",
        door_conf_threshold: float = 0.25,
        # Door classification model
        door_classifier_model_path: str = "models/weights/few_shot_classifier_door_best_model.pth",
        # Filtering parameters
        min_door_width_ratio: float = 0.3,   # Minimum door width as ratio of avg
        min_door_height_ratio: float = 0.3,  # Minimum door height as ratio of avg
        device: str = "cpu",
    ):
        """
        Initialize door detection & classification system

        Args:
            stage3_output_dir: Directory with enhanced wagon frames
            output_base_dir: Base directory for outputs
            door_detection_model_path: Path to YOLOv12n door detection model
            door_conf_threshold: Confidence threshold for door detection
            door_classifier_model_path: Path to EfficientNet classifier
            min_door_width_ratio: Minimum door width (relative to average)
            min_door_height_ratio: Minimum door height (relative to average)
            device: "cpu" or "cuda"
        """
        self.stage3_output_dir = Path(stage3_output_dir)
        self.output_base_dir = Path(output_base_dir)
        self.door_conf_threshold = door_conf_threshold
        self.min_door_width_ratio = min_door_width_ratio
        self.min_door_height_ratio = min_door_height_ratio
        self.device = device

        # Create output directories
        self.create_output_dirs()

        # Load door detection model (YOLO)
        self.load_door_detector(door_detection_model_path)

        # Load door classification model (EfficientNet)
        self.load_door_classifier(door_classifier_model_path)

        # Initialize transform for classifier
        self.classifier_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        # IoU threshold for merging overlapping doors
        self.merge_iou_threshold = 0.5  # can be tuned

    # ===================== DIRECTORY & LOADING =====================

    def create_output_dirs(self):
        """Create output directory structure"""
        self.output_dirs = {
            "door_rois": self.output_base_dir / "stage4_door_rois",
            "annotated_images": self.output_base_dir / "stage4_annotated",
            "results": self.output_base_dir / "stage4_results",
        }

        # Create subdirectories for each door class
        for class_name in ["good", "damaged", "missing"]:
            (self.output_dirs["door_rois"] / class_name).mkdir(
                parents=True, exist_ok=True
            )

        for dir_path in self.output_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        print("STAGE 4 output directories created")

    def load_door_detector(self, model_path: str):
        """Load YOLOv12n door detection model"""
        print("Loading door detection model...")
        print(f"  Model: {model_path}")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Door detection model not found: {model_path}")
        self.door_detector = YOLO(model_path)
        print("Door detector loaded successfully")

    def load_door_classifier(self, model_path: str):
        """Load EfficientNet door classifier"""
        print("Loading door classifier...")
        print(f"  Model: {model_path}")
        print(f"  Device: {self.device}")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Door classifier model not found: {model_path}")

        self.door_classifier = EfficientNetDoorClassifier(num_classes=3, pretrained=False)

        checkpoint = torch.load(model_path, map_location=self.device)
        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            self.door_classifier.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.door_classifier.load_state_dict(checkpoint)

        self.door_classifier.to(self.device)
        self.door_classifier.eval()
        print("Door classifier loaded successfully")

    # ===================== DOOR DETECTION =====================

    def detect_doors(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect door regions in wagon image

        Args:
            image: Input image (BGR)

        Returns:
            List of (x1, y1, x2, y2, confidence)
        """
        results = self.door_detector(
            image, conf=self.door_conf_threshold, verbose=False
        )[0]
        detections: List[Tuple[int, int, int, int, float]] = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append((x1, y1, x2, y2, conf))
        return detections

    def filter_spurious_detections(
        self,
        detections: List[Tuple[int, int, int, int, float]],
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Filter out spurious/invalid door detections based on size.

        Logic:
        1. Calculate average door size from all detections
        2. Remove detections smaller than threshold * average

        Args:
            detections: List of (x1, y1, x2, y2, confidence)
        Returns:
            Filtered list of detections
        """
        if len(detections) == 0:
            return []

        widths = [(x2 - x1) for x1, y1, x2, y2, _ in detections]
        heights = [(y2 - y1) for x1, y1, x2, y2, _ in detections]

        avg_width = float(np.mean(widths))
        avg_height = float(np.mean(heights))

        min_width = avg_width * self.min_door_width_ratio
        min_height = avg_height * self.min_door_height_ratio

        filtered: List[Tuple[int, int, int, int, float]] = []
        for x1, y1, x2, y2, conf in detections:
            width = x2 - x1
            height = y2 - y1
            if width >= min_width and height >= min_height:
                filtered.append((x1, y1, x2, y2, conf))
        return filtered

    # ===================== MERGE OVERLAPPING DOORS =====================

    @staticmethod
    def _iou(
        box_a: Tuple[int, int, int, int],
        box_b: Tuple[int, int, int, int],
    ) -> float:
        """Compute IoU between two boxes (x1,y1,x2,y2)."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return float(inter_area / union)

    def _merge_overlapping_doors(
        self,
        detections: List[Tuple[int, int, int, int, float]],
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Merge overlapping door detections so the same physical door is counted once.

        Strategy:
        - Build groups of boxes where IoU > merge_iou_threshold.
        - For each group:
          * Option A (current): create a union box covering all detections,
            confidence = max(conf in group).
          * Option B: keep only the largest-area box in the group
            (see commented line if you prefer that).

        Returns:
            List of merged detections (x1,y1,x2,y2,conf)
        """
        if not detections:
            return []

        boxes = np.array([[x1, y1, x2, y2] for x1, y1, x2, y2, _ in detections], dtype=np.float32)
        confs = np.array([conf for _, _, _, _, conf in detections], dtype=np.float32)

        n = len(detections)
        used = np.zeros(n, dtype=bool)
        merged: List[Tuple[int, int, int, int, float]] = []

        for i in range(n):
            if used[i]:
                continue
            group_indices = [i]
            used[i] = True
            box_i = boxes[i]

            # gather all overlapping boxes with IoU > threshold
            for j in range(i + 1, n):
                if used[j]:
                    continue
                box_j = boxes[j]
                iou_val = self._iou(
                    (int(box_i[0]), int(box_i[1]), int(box_i[2]), int(box_i[3])),
                    (int(box_j[0]), int(box_j[1]), int(box_j[2]), int(box_j[3])),
                )
                if iou_val > self.merge_iou_threshold:
                    used[j] = True
                    group_indices.append(j)

            # Merge group
            group_boxes = boxes[group_indices]
            group_confs = confs[group_indices]

            # Option A: union box (covers entire overlapping region)
            x1 = int(np.min(group_boxes[:, 0]))
            y1 = int(np.min(group_boxes[:, 1]))
            x2 = int(np.max(group_boxes[:, 2]))
            y2 = int(np.max(group_boxes[:, 3]))
            conf = float(np.max(group_confs))

            # Option B (alternative): keep only largest-area box.
            # areas = (group_boxes[:, 2] - group_boxes[:, 0]) * (group_boxes[:, 3] - group_boxes[:, 1])
            # k = int(np.argmax(areas))
            # x1, y1, x2, y2 = map(int, group_boxes[k])
            # conf = float(group_confs[k])

            merged.append((x1, y1, x2, y2, conf))

        return merged

    # ===================== DOOR CLASSIFICATION =====================

    def classify_door(
        self,
        door_roi: np.ndarray,
        tta: bool = True,
    ) -> Tuple[str, float, np.ndarray]:
        """
        Classify door ROI as good/damaged/missing

        Args:
            door_roi: Door region image (BGR)
            tta: Apply test-time augmentation

        Returns:
            (prediction, confidence, probabilities)
        """
        door_rgb = cv2.cvtColor(door_roi, cv2.COLOR_BGR2RGB)
        door_pil = Image.fromarray(door_rgb)

        self.door_classifier.eval()
        with torch.no_grad():
            if tta:
                # Test-time augmentation: original + flips + small rotations
                imgs = [
                    door_pil,
                    door_pil.transpose(Image.FLIP_LEFT_RIGHT),
                    door_pil.rotate(5),
                    door_pil.rotate(-5),
                ]
                all_logits = []
                for aug_img in imgs:
                    img_tensor = self.classifier_transform(aug_img).unsqueeze(0).to(
                        self.device
                    )
                    logits = self.door_classifier(img_tensor)
                    all_logits.append(logits)
                avg_logits = torch.mean(
                    torch.cat(all_logits, dim=0), dim=0, keepdim=True
                )
                probs = F.softmax(avg_logits, dim=1)
            else:
                img_tensor = self.classifier_transform(door_pil).unsqueeze(0).to(
                    self.device
                )
                logits = self.door_classifier(img_tensor)
                probs = F.softmax(logits, dim=1)

        confidence, predicted = torch.max(probs, 1)

        class_names = ["good", "damaged", "missing"]
        prediction = class_names[predicted.item()]
        confidence = confidence.item()
        all_probs = probs.cpu().numpy()[0]
        return prediction, confidence, all_probs

    # ===================== PROCESS SINGLE WAGON =====================

    def process_wagon(
        self,
        wagon_name: str,
        image_path: Path,
    ) -> Dict:
        """
        Process single wagon: detect doors + merge overlaps + classify

        Args:
            wagon_name: Wagon identifier
            image_path: Path to enhanced wagon image

        Returns:
            Dictionary with detection & classification results
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        # 1) Detect doors
        detections = self.detect_doors(img)

        # 2) Filter spurious detections
        filtered_detections = self.filter_spurious_detections(detections)

        # 3) Merge overlapping doors so same door isn't counted multiple times
        merged_detections = self._merge_overlapping_doors(filtered_detections)

        door_results = []
        annotated_img = img.copy()

        for door_id, (x1, y1, x2, y2, det_conf) in enumerate(merged_detections):
            door_roi = img[y1:y2, x1:x2]
            if door_roi.size == 0:
                continue

            prediction, class_conf, probs = self.classify_door(door_roi, tta=True)

            # Save door ROI
            door_filename = f"{wagon_name}_door_{door_id}.jpg"
            door_save_path = (
                self.output_dirs["door_rois"] / prediction / door_filename
            )
            cv2.imwrite(str(door_save_path), door_roi)

            door_results.append(
                {
                    "door_id": door_id,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "detection_confidence": float(det_conf),
                    "classification": prediction,
                    "classification_confidence": float(class_conf),
                    "probabilities": {
                        "good": float(probs[0]),
                        "damaged": float(probs[1]),
                        "missing": float(probs[2]),
                    },
                    "door_roi_path": str(door_save_path),
                }
            )

            color = {
                "good": (0, 255, 0),       # Green
                "damaged": (0, 165, 255),  # Orange
                "missing": (0, 0, 255),    # Red
            }[prediction]

            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
            label = f"{prediction.upper()} {class_conf:.2f}"
            cv2.putText(
                annotated_img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        annotated_path = (
            self.output_dirs["annotated_images"] / f"{wagon_name}_annotated.jpg"
        )
        cv2.imwrite(str(annotated_path), annotated_img)

        return {
            "wagon_name": wagon_name,
            "total_doors_detected": len(merged_detections),
            "doors_before_filtering": len(detections),
            "doors_after_filtering": len(filtered_detections),
            "door_results": door_results,
            "annotated_image_path": str(annotated_path),
            "door_counts": {
                "good": sum(1 for d in door_results if d["classification"] == "good"),
                "damaged": sum(
                    1 for d in door_results if d["classification"] == "damaged"
                ),
                "missing": sum(
                    1 for d in door_results if d["classification"] == "missing"
                ),
            },
        }

    # ===================== PROCESS ALL WAGONS =====================

    def process_all_wagons(self, on_door_count_update: Optional[callable] = None) -> Dict:
        """
        Process all wagons from STAGE 3

        Returns:
            Dictionary with all results
        """
        print("\nSTAGE 4: Door Detection & Classification")

        enhanced_images = sorted(self.stage3_output_dir.glob("*_enhanced.jpg"))
        if len(enhanced_images) == 0:
            raise ValueError(f"No enhanced images found in {self.stage3_output_dir}")

        print(f"  Processing {len(enhanced_images)} wagons")

        all_results: Dict[str, Dict] = {}
        running_total_doors = 0

        for img_path in tqdm(enhanced_images, desc="Processing wagons"):
            wagon_name = img_path.stem.replace("_enhanced", "")
            try:
                wagon_results = self.process_wagon(wagon_name, img_path)
                all_results[wagon_name] = wagon_results
                
                # Update running total and call callback
                running_total_doors += wagon_results["total_doors_detected"]
                if on_door_count_update:
                    try:
                        on_door_count_update(running_total_doors)
                    except Exception as e:
                        print(f"Callback error: {e}")
                        
            except Exception as e:
                print(f"  Warning: Error processing {wagon_name}: {e}")
                continue

        self.save_results(all_results)
        return all_results

    def save_results(self, all_results: Dict):
        """Save detection & classification results"""
        json_path = self.output_dirs["results"] / "stage4_door_results.json"
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)

        total_wagons = len(all_results)
        total_doors = sum(r["total_doors_detected"] for r in all_results.values())
        total_good = sum(r["door_counts"]["good"] for r in all_results.values())
        total_damaged = sum(r["door_counts"]["damaged"] for r in all_results.values())
        total_missing = sum(r["door_counts"]["missing"] for r in all_results.values())

        stats = {
            "total_wagons_processed": total_wagons,
            "total_doors_detected": total_doors,
            "door_classification_summary": {
                "good_doors": total_good,
                "damaged_doors": total_damaged,
                "missing_doors": total_missing,
            },
            "percentages": {
                "good": round(100 * total_good / total_doors, 2)
                if total_doors > 0
                else 0,
                "damaged": round(100 * total_damaged / total_doors, 2)
                if total_doors > 0
                else 0,
                "missing": round(100 * total_missing / total_doors, 2)
                if total_doors > 0
                else 0,
            },
        }

        stats_path = self.output_dirs["results"] / "stage4_statistics.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        print("\nSTAGE 4 Results Saved:")
        print(f"  - Door ROIs: {self.output_dirs['door_rois']}")
        print(f"  - Annotated Images: {self.output_dirs['annotated_images']}")
        print(f"  - Results: {json_path}")
        print(f"  - Statistics: {stats_path}")
        print("\nDoor Classification Summary:")
        print(f"  Total Wagons: {total_wagons}")
        print(f"  Total Doors: {total_doors}")
        print(f"  Good Doors: {total_good} ({stats['percentages']['good']}%)")
        print(f"  Damaged Doors: {total_damaged} ({stats['percentages']['damaged']}%)")
        print(f"  Missing Doors: {total_missing} ({stats['percentages']['missing']}%)")

# ================= STANDALONE TEST =================

if __name__ == "__main__":
    door_system = DoorDetectionClassification(
        stage3_output_dir=r"C:\Users\Viranch Patel\Desktop\Wagon_Inspection_System\outputs\stage3_enhanced_frames",
        output_base_dir=r"C:\Users\Viranch Patel\Desktop\Wagon_Inspection_System\outputs",
        door_detection_model_path=r"C:\Users\Viranch Patel\Desktop\Wagon_Inspection_System\models\weights\yolo12n_Door_ROI_best.pt",
        door_conf_threshold=0.25,
        door_classifier_model_path=r"C:\Users\Viranch Patel\Desktop\Wagon_Inspection_System\models\weights\few_shot_classifier_door_best_model.pth",
        min_door_width_ratio=0.3,
        min_door_height_ratio=0.3,
        device="cpu",
    )

    results = door_system.process_all_wagons()
    print("\n" + "=" * 60)
    print("STAGE 4 COMPLETE - DOOR DETECTION & CLASSIFICATION")
    print("=" * 60)
