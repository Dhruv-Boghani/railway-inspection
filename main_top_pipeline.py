"""
TOP CAMERA ANGLE: Main Pipeline Orchestrator

5-Stage pipeline for top camera angle wagon inspection:
1. Top Wagon Detection & Counting
2. Quality Assessment (reuse)
3. Image Enhancement (reuse)
4. Top Damage Detection
5. Result Aggregation
"""

import sys
import time
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime
import json

from modules.top_wagon_detection import TopWagonDetectionCounter
from modules.quality_assessment import QualityAssessment
from modules.image_enhancement import ImageEnhancer
from modules.top_damage_detection import TopDamageDetection
from pipeline_config import PipelineConfig


class TopInspectionPipeline:
    """
    Complete Top Camera Angle Inspection Pipeline
    
    5 Stages:
    1. Wagon Detection & Counting (Top YOLO)
    2. Quality Assessment
    3. Image Enhancement  
    4. Damage Detection (Top Damage YOLO)
    5. Result Aggregation
    
    NO doors, NO OCR for top view.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None, verbose: bool = True):
        """Initialize top camera pipeline."""
        self.config = config or PipelineConfig()
        self.verbose = verbose
        
        # Module instances
        self.stage1_detector = None
        self.stage2_quality = None
        self.stage3_enhancer = None
        self.stage4_damage = None
        
        # Results storage
        self.results = {}
        
        if self.verbose:
            self.print_banner()
            
    def print_banner(self):
        """Print pipeline banner."""
        print("\n" + "=" * 70)
        print("TOP CAMERA ANGLE - WAGON INSPECTION PIPELINE")
        print("=" * 70)
        print("5-Stage Pipeline:")
        print("  Stage 1: Top Wagon Detection & Counting")
        print("  Stage 2: Quality Assessment")
        print("  Stage 3: Image Enhancement")
        print("  Stage 4: Top Damage Detection")
        print("  Stage 5: Result Aggregation")
        print("=" * 70 + "\n")
        
    def log(self, message: str):
        """Log message if verbose."""
        if self.verbose:
            print(f"[Top Pipeline] {message}")
            
    # ============================================================
    # STAGE 1: Top Wagon Detection & Counting
    # ============================================================
    
    def initialize_stage1(self):
        """Initialize Stage 1: Top Wagon Detection."""
        self.log("Initializing Stage 1: Top Wagon Detection")
        self.stage1_detector = TopWagonDetectionCounter(
            model_path=str(self.config.top_detection_model_path),
            device=self.config.stage1.device,  # FIXED: use stage1.device
            output_base_dir=str(self.config.output_root),
        )
        
    def run_stage1(self, video_path: str, output_video_path: Optional[str] = None, on_wagon_counted: Optional[callable] = None) -> Dict:
        """Run Stage 1: Top Wagon Detection."""
        if not self.stage1_detector:
            self.initialize_stage1()
            
        self.log("=" * 60)
        self.log("STAGE 1: Top Wagon Detection & Counting")
        self.log("=" * 60)
        
        start_time = time.time()
        wagon_data = self.stage1_detector.process_video(
            video_path, 
            output_video_path,
            on_wagon_counted=on_wagon_counted  # Pass callback for real-time updates
        )
        elapsed = time.time() - start_time
        
        self.log(f"Stage 1 Complete in {elapsed:.1f}s")
        self.log(f"Wagons detected: {len(wagon_data)}")
        
        self.results['stage1'] = wagon_data
        return wagon_data
        
    # ============================================================
    # STAGE 2: Quality Assessment (Reuse)
    # ============================================================
    
    def initialize_stage2(self):
        """Initialize Stage 2: Quality Assessment."""
        self.log("Initializing Stage 2: Quality Assessment")
        self.stage2_quality = QualityAssessment(
            stage1_output_dir=str(self.config.output_root / "stage1_top_detections"),
            output_base_dir=str(self.config.output_root),
        )
        
    def run_stage2(self, on_frame_assessed: Optional[callable] = None):
        """Run Stage 2: Quality Assessment."""
        if not self.stage2_quality:
            self.initialize_stage2()
            
        self.log("=" * 60)
        self.log("STAGE 2: Quality Assessment")
        self.log("=" * 60)
        
        start_time = time.time()
        quality_results = self.stage2_quality.process_all_wagons(on_frame_assessed=on_frame_assessed)  # FIXED: correct method name
        elapsed = time.time() - start_time
        
        self.log(f"Stage 2 Complete in {elapsed:.1f}s")
        self.results['stage2'] = quality_results
        
    # ============================================================
    # STAGE 3: Image Enhancement (Reuse)
    # ============================================================
    
    def initialize_stage3(self):
        """Initialize Stage 3: Image Enhancement."""
        self.log("Initializing Stage 3: Image Enhancement")
        # Stage 3 reads stage2 JSON from stage2_quality_results
        stage2_results_path = str(self.config.output_root / "stage2_quality_results" / "stage2_quality_summary.json")
        self.stage3_enhancer = ImageEnhancer(
            stage2_results_path=stage2_results_path,
            output_base_dir=str(self.config.output_root),
            mprnet_model_path=str(self.config.stage3.mprnet_model_path),
            device=self.config.stage1.device,
        )
        
    def run_stage3(self, on_frame_enhanced: Optional[callable] = None):
        """Run Stage 3: Image Enhancement."""
        if not self.stage3_enhancer:
            self.initialize_stage3()
            
        self.log("=" * 60)
        self.log("STAGE 3: Image Enhancement")
        self.log("=" * 60)
        
        start_time = time.time()
        enhancement_results = self.stage3_enhancer.process_all_wagons(on_frame_enhanced=on_frame_enhanced)  # FIXED: correct method name
        elapsed = time.time() - start_time
        
        self.log(f"Stage 3 Complete in {elapsed:.1f}s")
        
    # ============================================================
    # STAGE 4: Top Damage Detection
    # ============================================================
    
    def initialize_stage4(self):
        """Initialize Stage 4: Top Damage Detection."""
        self.log("Initializing Stage 4: Top Damage Detection")
        self.stage4_damage = TopDamageDetection(
            model_path=str(self.config.top_damage_model_path),
            device=self.config.stage1.device,  # FIXED: use stage1.device
            output_base_dir=str(self.config.output_root),
        )
        
    def run_stage4(self, on_damage_processed: Optional[callable] = None) -> Dict:
        """Run Stage 4: Top Damage Detection."""
        if not self.stage4_damage:
            self.initialize_stage4()
            
        self.log("=" * 60)
        self.log("STAGE 4: Top Damage Detection")
        self.log("=" * 60)
        
        start_time = time.time()
        damage_results = self.stage4_damage.process_folder(
            self.config.output_root / "stage3_enhanced_frames",
            on_damage_processed=on_damage_processed
        )
        elapsed = time.time() - start_time
        
        self.log(f"Stage 4 Complete in {elapsed:.1f}s")
        self.results['stage4_damage'] = damage_results
        return damage_results
        
    # ============================================================
    # STAGE 5: Aggregation
    # ============================================================
    
    def run_aggregation(self, damage_results: Dict, total_time: float = 0) -> Dict:
        """Aggregate final results (no doors, no OCR for top view)."""
        self.log("=" * 60)
        self.log("STAGE 5: Result Aggregation")
        self.log("=" * 60)
        
        # Aggregate wagon data
        wagon_summary = {}
        total_wagons = len(self.results.get('stage1', {}))
        wagons_with_damage = 0
        total_damage_count = 0
        
        for wagon_key, damage_data in damage_results.items():
            if damage_data.get('has_damage'):
                wagons_with_damage += 1
            total_damage_count += damage_data.get('total_detections', 0)
            
            wagon_summary[wagon_key] = {
                "wagon_name": wagon_key,
                "damage": damage_data,
            }
            
        # Final summary
        final_results = {
            "pipeline_type": "TOP",
            "summary": {
                "pipeline_info": {
                    "total_wagons": total_wagons,
                    "pipeline_type": "TOP",
                    "camera_angle": "TOP",
                    "total_time": round(total_time, 1),
                },
                "damage_summary": {
                    "wagons_with_damage": wagons_with_damage,
                    "total_damage_detections": total_damage_count,
                },
            },
            "wagons": wagon_summary,
            "stage1_detection": self.results.get('stage1', {}),
            "stage4_damage": damage_results,
        }
        
        # Save final results
        output_dir = self.config.output_root / "final_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = output_dir / f"final_results_{timestamp}.json"
        
        with open(result_path, "w") as f:
            json.dump(final_results, f, indent=2)
            
        self.log(f"Final results saved: {result_path}")
        self.log(f"Total wagons: {total_wagons}")
        self.log(f"Wagons with damage: {wagons_with_damage}")
        
        return final_results
        
    # ============================================================
    # Complete Pipeline
    # ============================================================
    
    def run_complete_pipeline(
        self,
        video_path: str,
        output_video_path: Optional[str] = None,
    ) -> Dict:
        """Run complete 5-stage top camera pipeline."""
        pipeline_start = time.time()
        
        self.log("Starting TOP camera angle pipeline...")
        
        # Stage 1: Top Wagon Detection
        wagon_data = self.run_stage1(video_path, output_video_path)
        
        if not wagon_data:
            self.log("No wagons detected. Pipeline stopped.")
            return {}
            
        # Stage 2: Quality Assessment
        self.run_stage2()
        
        # Stage 3: Image Enhancement
        self.run_stage3()
        
        # Stage 4: Top Damage Detection
        damage_results = self.run_stage4()
        
        # Stage 5: Aggregation (pass total_time)
        total_time = time.time() - pipeline_start
        final_results = self.run_aggregation(damage_results, total_time)
        
        total_time = time.time() - pipeline_start
        self.log(f"\n{'='*60}")
        self.log(f"TOP PIPELINE COMPLETE - Total Time: {total_time:.1f}s")
        self.log(f"{'='*60}\n")
        
        return final_results
