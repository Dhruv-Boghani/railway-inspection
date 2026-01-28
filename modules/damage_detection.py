"""
STAGE 5: Damage Detection Module (Dent & Scratch Segmentation)
Detects and segments dents & scratches on wagon body using YOLOv8-seg
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from tqdm import tqdm
from ultralytics import YOLO


class DamageDetection:
    """
    STAGE 5: Damage Detection System
    
    Pipeline:
    1. Load enhanced wagon ROIs from STAGE 3
    2. Detect and segment dents & scratches using YOLOv8-seg
    3. Calculate damage statistics (area, count, severity)
    4. Save annotated images and damage masks
    5. Generate comprehensive damage reports
    """
    
    def __init__(
        self,
        stage3_output_dir: str = "outputs/stage3_enhanced_frames",
        output_base_dir: str = "outputs",
        # Damage detection model
        damage_model_path: str = "models/weights/yolo8s-seg_damage_best.pt",
        conf_threshold: float = 0.25,
        img_size: int = 640,
        device: str = "cpu",
        # Severity thresholds (percentage of image area)
        minor_damage_threshold: float = 0.5,  # < 0.5% of image
        moderate_damage_threshold: float = 2.0,  # 0.5-2% of image
        # major damage > 2%
    ):
        """
        Initialize damage detection system
        
        Args:
            stage3_output_dir: Directory with enhanced wagon frames
            output_base_dir: Base directory for outputs
            damage_model_path: Path to YOLOv8-seg damage detection model
            conf_threshold: Confidence threshold for detection
            img_size: Input image size for YOLO
            device: "cpu" or "cuda"
            minor_damage_threshold: Area threshold for minor damage (%)
            moderate_damage_threshold: Area threshold for moderate damage (%)
        """
        self.stage3_output_dir = Path(stage3_output_dir)
        self.output_base_dir = Path(output_base_dir)
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        self.device = device
        self.minor_threshold = minor_damage_threshold
        self.moderate_threshold = moderate_damage_threshold
        
        # Create output directories
        self.create_output_dirs()
        
        # Load damage detection model
        self.load_damage_model(damage_model_path)
    
    def create_output_dirs(self):
        """Create output directory structure"""
        self.output_dirs = {
            'annotated_images': self.output_base_dir / 'stage5_damage_annotated',
            'damage_masks': self.output_base_dir / 'stage5_damage_masks',
            'damage_rois': self.output_base_dir / 'stage5_damage_rois',
            'results': self.output_base_dir / 'stage5_results',
            'severity_analysis': self.output_base_dir / 'stage5_severity_analysis'
        }
        
        # Create severity subdirectories
        for severity in ['minor', 'moderate', 'severe']:
            (self.output_dirs['severity_analysis'] / severity).mkdir(parents=True, exist_ok=True)
        
        for dir_path in self.output_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ STAGE 5 output directories created")
    
    def load_damage_model(self, model_path: str):
        """Load YOLOv8-seg damage detection model"""
        print(f"🔧 Loading damage detection model...")
        print(f"   Model: {model_path}")
        print(f"   Device: {self.device}")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Damage model not found: {model_path}")
        
        self.damage_model = YOLO(model_path)
        print(f"✅ Damage detection model loaded successfully")
    
    # ===================== DAMAGE DETECTION =====================
    
    def detect_damage(self, image_path: Path) -> Dict:
        """
        Detect and segment damage in wagon image
        
        Args:
            image_path: Path to wagon image
            
        Returns:
            Dictionary with detection results
        """
        # Run YOLO segmentation
        results = self.damage_model.predict(
            source=str(image_path),
            imgsz=self.img_size,
            conf=self.conf_threshold,
            device=self.device,
            save=False,
            verbose=False
        )[0]
        
        # Load original image
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]
        image_area = h * w
        
        # Parse results
        detections = []
        
        if results.masks is not None:
            masks = results.masks.data.cpu().numpy()
            boxes = results.boxes.data.cpu().numpy()
            
            for idx, (mask, box) in enumerate(zip(masks, boxes)):
                x1, y1, x2, y2, conf, cls = box
                
                # Get class name (0=dent, 1=scratch)
                class_id = int(cls)
                class_name = results.names[class_id] if hasattr(results, 'names') else f"class_{class_id}"
                
                # Resize mask to image size
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                
                # Calculate damage area
                damage_pixels = np.sum(mask_resized > 0.5)
                damage_area_percent = (damage_pixels / image_area) * 100
                
                detections.append({
                    'damage_id': idx,
                    'class': class_name,
                    'class_id': class_id,
                    'confidence': float(conf),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'mask': mask_resized,
                    'damage_pixels': int(damage_pixels),
                    'damage_area_percent': float(damage_area_percent)
                })
        
        return {
            'total_detections': len(detections),
            'detections': detections,
            'image_size': (w, h),
            'image_area': image_area
        }
    
    # ===================== DAMAGE SEVERITY CLASSIFICATION =====================
    
    def classify_severity(self, damage_area_percent: float) -> str:
        """
        Classify damage severity based on area
        
        Args:
            damage_area_percent: Damage area as percentage of image
            
        Returns:
            Severity: "minor", "moderate", or "severe"
        """
        if damage_area_percent < self.minor_threshold:
            return "minor"
        elif damage_area_percent < self.moderate_threshold:
            return "moderate"
        else:
            return "severe"
    
    def analyze_wagon_damage(self, detections: List[Dict]) -> Dict:
        """
        Analyze overall wagon damage
        
        Args:
            detections: List of damage detections
            
        Returns:
            Damage analysis summary
        """
        if len(detections) == 0:
            return {
                'total_damage_count': 0,
                'total_damage_area_percent': 0.0,
                'severity': 'none',
                'damage_breakdown': {'dent': 0, 'scratch': 0},
                'severity_breakdown': {'minor': 0, 'moderate': 0, 'severe': 0}
            }
        
        # Calculate totals
        total_damage_area = sum(d['damage_area_percent'] for d in detections)
        
        # Count by damage type
        damage_counts = {}
        for d in detections:
            class_name = d['class']
            damage_counts[class_name] = damage_counts.get(class_name, 0) + 1
        
        # Count by severity
        severity_counts = {'minor': 0, 'moderate': 0, 'severe': 0}
        for d in detections:
            severity = self.classify_severity(d['damage_area_percent'])
            severity_counts[severity] += 1
        
        # Overall wagon severity (based on total damage area)
        overall_severity = self.classify_severity(total_damage_area)
        
        return {
            'total_damage_count': len(detections),
            'total_damage_area_percent': round(total_damage_area, 3),
            'severity': overall_severity,
            'damage_breakdown': damage_counts,
            'severity_breakdown': severity_counts
        }
    
    # ===================== VISUALIZATION =====================
    
    def create_annotated_image(
        self, 
        image_path: Path, 
        detections: List[Dict]
    ) -> np.ndarray:
        """
        Detect and segment damage in wagon image
        
        Args:
            image_path: Path to wagon image
            
        Returns:
            Dictionary with detection results
        """
        img = cv2.imread(str(image_path))
        overlay = img.copy()
        
        # Color map for damage types
        color_map = {
            'dent': (0, 0, 255),      # Red
            'scratch': (0, 165, 255),  # Orange
            0: (0, 0, 255),           # Red (fallback for class_id)
            1: (0, 165, 255)          # Orange (fallback for class_id)
        }
        
        for detection in detections:
            # Get color based on class
            class_key = detection['class'] if detection['class'] in color_map else detection['class_id']
            color = color_map.get(class_key, (255, 0, 0))  # Default to blue
            
            # Draw mask
            mask = detection['mask']
            overlay[mask > 0.5] = overlay[mask > 0.5] * 0.5 + np.array(color) * 0.5
            
            # Draw bounding box
            x1, y1, x2, y2 = detection['bbox']
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            severity = self.classify_severity(detection['damage_area_percent'])
            label = f"{detection['class']} ({severity}) {detection['confidence']:.2f}"
            cv2.putText(overlay, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return overlay
    
    def create_damage_mask_visualization(
        self, 
        image_path: Path, 
        detections: List[Dict]
    ) -> np.ndarray:
        """
        Create pure mask visualization (binary mask overlay)
        
        Args:
            image_path: Path to original image
            detections: List of damage detections
            
        Returns:
            Mask visualization
        """
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]
        
        # Create combined mask
        combined_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        for detection in detections:
            mask = detection['mask']
            class_name = detection['class']
            
            # Color by damage type
            if 'dent' in class_name.lower() or detection['class_id'] == 0:
                color = (0, 0, 255)  # Red for dents
            else:
                color = (0, 165, 255)  # Orange for scratches
            
            combined_mask[mask > 0.5] = color
        
        # Blend with original
        result = cv2.addWeighted(img, 0.6, combined_mask, 0.4, 0)
        
        return result
    
    # ===================== PROCESS SINGLE WAGON =====================
    
    def process_wagon(
        self, 
        wagon_name: str, 
        image_path: Path
    ) -> Dict:
        """
        Process single wagon: detect damage + analyze
        
        Args:
            wagon_name: Wagon identifier
            image_path: Path to enhanced wagon image
            
        Returns:
            Dictionary with damage analysis results
        """
        # Detect damage
        detection_results = self.detect_damage(image_path)
        detections = detection_results['detections']
        
        # Analyze damage
        damage_analysis = self.analyze_wagon_damage(detections)
        
        # Create visualizations
        annotated_img = self.create_annotated_image(image_path, detections)
        mask_viz = self.create_damage_mask_visualization(image_path, detections)
        
        # Save annotated image
        annotated_path = self.output_dirs['annotated_images'] / f"{wagon_name}_damage_annotated.jpg"
        cv2.imwrite(str(annotated_path), annotated_img)
        
        # Save mask visualization
        mask_path = self.output_dirs['damage_masks'] / f"{wagon_name}_damage_mask.jpg"
        cv2.imwrite(str(mask_path), mask_viz)
        
        # Save to severity folder
        severity = damage_analysis['severity']
        if severity != 'none':
            severity_path = self.output_dirs['severity_analysis'] / severity / f"{wagon_name}.jpg"
            cv2.imwrite(str(severity_path), annotated_img)
        
        # Save individual damage ROIs
        img = cv2.imread(str(image_path))
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            damage_roi = img[y1:y2, x1:x2]
            
            if damage_roi.size > 0:
                roi_filename = f"{wagon_name}_damage_{detection['damage_id']}_{detection['class']}.jpg"
                roi_path = self.output_dirs['damage_rois'] / roi_filename
                cv2.imwrite(str(roi_path), damage_roi)
        
        # Prepare results (without masks for JSON serialization)
        detection_results_json = []
        for d in detections:
            detection_results_json.append({
                'damage_id': d['damage_id'],
                'class': d['class'],
                'confidence': d['confidence'],
                'bbox': d['bbox'],
                'damage_pixels': d['damage_pixels'],
                'damage_area_percent': d['damage_area_percent'],
                'severity': self.classify_severity(d['damage_area_percent'])
            })
        
        return {
            'wagon_name': wagon_name,
            'total_detections': detection_results['total_detections'],
            'detections': detection_results_json,
            'damage_analysis': damage_analysis,
            'annotated_image_path': str(annotated_path),
            'mask_image_path': str(mask_path)
        }
    
    # ===================== PROCESS ALL WAGONS =====================
    
    def process_all_wagons(self, on_damage_processed: Optional[callable] = None) -> Dict:
        """
        Process all wagons from STAGE 3
        
        Returns:
            Dictionary with all damage detection results
        """
        print(f"\n🔍 STAGE 5: Damage Detection (Dent & Scratch)")
        
        # Get all enhanced wagon images
        enhanced_images = sorted(self.stage3_output_dir.glob("*_enhanced.jpg"))
        
        if len(enhanced_images) == 0:
            raise ValueError(f"No enhanced images found in {self.stage3_output_dir}")
        
        print(f"   Processing {len(enhanced_images)} wagons")
        
        all_results = {}
        
        # Counter for callback (cumulative defects)
        running_total_defects = 0
        
        for img_path in tqdm(enhanced_images, desc="Detecting damage"):
            # Extract wagon name
            wagon_name = img_path.stem.replace('_enhanced', '')
            
            try:
                wagon_results = self.process_wagon(wagon_name, img_path)
                all_results[wagon_name] = wagon_results
                
                # Update running total of defects
                defects_in_wagon = wagon_results.get('total_detections', 0)
                running_total_defects += defects_in_wagon
                
                # Call callback with total defects
                if on_damage_processed:
                    try:
                        on_damage_processed(running_total_defects)
                    except Exception as e:
                        pass
                
            except Exception as e:
                print(f"  ⚠️  Error processing {wagon_name}: {e}")
                continue
        
        # Save results
        self.save_results(all_results)
        
        return all_results
    
    def save_results(self, all_results: Dict):
        """Save damage detection results and statistics"""
        # Save JSON summary
        json_path = self.output_dirs['results'] / 'stage5_damage_results.json'
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Generate statistics
        total_wagons = len(all_results)
        wagons_with_damage = sum(1 for r in all_results.values() if r['total_detections'] > 0)
        total_damages = sum(r['total_detections'] for r in all_results.values())
        
        # Count by severity
        severity_counts = {'none': 0, 'minor': 0, 'moderate': 0, 'severe': 0}
        for r in all_results.values():
            severity = r['damage_analysis']['severity']
            severity_counts[severity] += 1
        
        # Count by damage type
        damage_type_counts = {}
        for r in all_results.values():
            for damage_type, count in r['damage_analysis']['damage_breakdown'].items():
                damage_type_counts[damage_type] = damage_type_counts.get(damage_type, 0) + count
        
        stats = {
            'total_wagons_processed': total_wagons,
            'wagons_with_damage': wagons_with_damage,
            'wagons_without_damage': total_wagons - wagons_with_damage,
            'total_damages_detected': total_damages,
            'severity_distribution': severity_counts,
            'damage_type_distribution': damage_type_counts,
            'damage_rate': round(100 * wagons_with_damage / total_wagons, 2) if total_wagons > 0 else 0
        }
        
        stats_path = self.output_dirs['results'] / 'stage5_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n✅ STAGE 5 Results Saved:")
        print(f"   - Annotated Images: {self.output_dirs['annotated_images']}")
        print(f"   - Damage Masks: {self.output_dirs['damage_masks']}")
        print(f"   - Damage ROIs: {self.output_dirs['damage_rois']}")
        print(f"   - Results: {json_path}")
        print(f"   - Statistics: {stats_path}")
        
        print(f"\n📊 Damage Detection Summary:")
        print(f"   Total Wagons: {total_wagons}")
        print(f"   Wagons with Damage: {wagons_with_damage} ({stats['damage_rate']}%)")
        print(f"   Total Damages Detected: {total_damages}")
        print(f"\n   Severity Distribution:")
        print(f"     - No Damage: {severity_counts['none']}")
        print(f"     - Minor: {severity_counts['minor']}")
        print(f"     - Moderate: {severity_counts['moderate']}")
        print(f"     - Severe: {severity_counts['severe']}")
        if damage_type_counts:
            print(f"\n   Damage Type Distribution:")
            for damage_type, count in damage_type_counts.items():
                print(f"     - {damage_type}: {count}")


# ================= STANDALONE TEST =================
if __name__ == "__main__":
    
    # Initialize damage detection
    damage_detector = DamageDetection(
        stage3_output_dir=r"C:\Users\Viranch Patel\Desktop\Wagon_Inspection_System\outputs\stage3_enhanced_frames",
        output_base_dir=r"C:\Users\Viranch Patel\Desktop\Wagon_Inspection_System\outputs",
        damage_model_path=r"C:\Users\Viranch Patel\Desktop\Wagon_Inspection_System\models\weights\yolo8s-seg_damage_best.pt",
        conf_threshold=0.25,
        img_size=640,
        device="cpu",
        minor_damage_threshold=0.5,
        moderate_damage_threshold=2.0
    )
    
    # Process all wagons
    results = damage_detector.process_all_wagons()
    
    print("\n" + "="*60)
    print("STAGE 5 COMPLETE - DAMAGE DETECTION")
    print("="*60)
