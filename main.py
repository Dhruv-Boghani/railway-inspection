"""
Wagon Inspection System - Main Pipeline
Complete End-to-End Pipeline: Video -> Wagon Inspection Results

Pipeline Stages
1. Wagon Detection / Tracking / Counting
2. Quality Assessment
3. Image Enhancement
4. Door Detection & Classification
5. Damage Detection
6. Wagon Number Extraction (OCR)
7. Results Aggregation

Author: Your Name
Date: 2026-01-04 (updated for Stage 1–6 naming & CUDA support)
"""

import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any
from dataclasses import asdict

from pathlib import Path
import sys

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))


# Import configuration
from pipeline_config import (
    PipelineConfig,
    get_config,
    validate_models_exist,
    setup_logging,
)

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
if str(ROOT_DIR / "modules") not in sys.path:
    sys.path.append(str(ROOT_DIR / "modules"))

from modules.wagon_detection_counting import WagonDetectionCounter
from modules.quality_assessment import QualityAssessment
from modules.image_enhancement import ImageEnhancer
from modules.door_detection_classification import DoorDetectionClassification
from modules.damage_detection import DamageDetection
from modules.wagon_number_extraction import WagonNumberExtractor




class WagonInspectionPipeline:
    """
    Complete Wagon Inspection Pipeline

    Orchestrates all stages from video input to final results:
    1: Wagon Detection / Tracking / Counting
    2: Quality Assessment
    3: Image Enhancement
    4: Door Detection & Classification
    5: Damage Detection
    6: Wagon Number Extraction (OCR)
    7: Results Aggregation
    """

    def __init__(self, config: Optional[PipelineConfig] = None, verbose: bool = True):
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration (uses default if None).
            verbose: Print progress messages.
        """
        self.config = config if config else get_config()
        self.verbose = verbose
        self.logger = setup_logging(self.config) if self.config.log_level else None

        # Lazy-loaded stage processors
        self.stage1_detector: Optional[WagonDetectionCounter] = None
        self.stage2_qa: Optional[QualityAssessment] = None
        self.stage3_enhancer: Optional[ImageEnhancer] = None
        self.stage4_door: Optional[DoorDetectionClassification] = None
        self.stage5_damage: Optional[DamageDetection] = None
        self.stage6_ocr: Optional[WagonNumberExtractor] = None

        # Pipeline statistics
        self.stats: Dict[str, Any] = {
            "total_wagons": 0,
            "stage_times": {},
            "start_time": None,
            "end_time": None,
        }

        if self.verbose:
            self.print_banner()

    # ------------------------------------------------------------------
    # Utility / logging helpers
    # ------------------------------------------------------------------
    def print_banner(self) -> None:
        print("=" * 80)
        print("WAGON INSPECTION SYSTEM - COMPLETE PIPELINE")
        print("=" * 80)
        print(f"Date      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output dir: {self.config.output_root}")
        print("=" * 80)

    def log(self, message: str, level: str = "info") -> None:
        if self.verbose:
            print(message)
        if self.logger:
            getattr(self.logger, level)(message)

    # ------------------------------------------------------------------
    # Stage initialization
    # ------------------------------------------------------------------
    def initialize_stage1(self) -> None:
        """Initialize Stage 1: Wagon Detection / Tracking / Counting."""
        if self.stage1_detector is None:
            self.log("Initializing Stage 1: Wagon Detection / Tracking / Counting")
            cfg = self.config.stage1
            # Use dynamic output_root for job-specific folder
            output_dir = str(self.config.output_root)
            self.stage1_detector = WagonDetectionCounter(
                model_path=cfg.model_path,
                direction=cfg.direction,
                device=cfg.device,
                img_size=cfg.img_size,
                conf_threshold=cfg.conf_threshold,
                min_frames_in_roi=cfg.min_frames_in_roi,
                roi_padding=cfg.roi_padding,
                output_base_dir=output_dir,  # Use job-specific output folder
                min_confidence_for_storage=cfg.min_confidence_for_storage,
                min_bbox_area=cfg.min_bbox_area,
                max_frames_per_wagon=cfg.max_frames_per_wagon,
            )
            self.log("Stage 1 initialized")

    def initialize_stage2(self) -> None:
        """Initialize Stage 2: Quality Assessment."""
        if self.stage2_qa is None:
            self.log("Initializing Stage 2: Quality Assessment")
            # Use dynamic output_root for job-specific folder
            output_dir = Path(self.config.output_root)
            self.stage2_qa = QualityAssessment(
                stage1_output_dir=str(output_dir / "stage1_wagon_rois"),
                output_base_dir=str(output_dir),
            )
            self.log("Stage 2 initialized")

    def initialize_stage3(self) -> None:
        """Initialize Stage 3: Image Enhancement."""
        if self.stage3_enhancer is None:
            self.log("Initializing Stage 3: Image Enhancement")
            cfg = self.config.stage3
            # Use dynamic output_root for job-specific folder
            output_dir = Path(self.config.output_root)
            self.stage3_enhancer = ImageEnhancer(
                stage2_results_path=str(output_dir / "stage2_quality_results" / "stage2_quality_summary.json"),
                output_base_dir=str(output_dir),
                mprnet_model_path=cfg.mprnet_model_path,
                device=cfg.device,
                # Enhancement parameters
                denoise_strength=cfg.denoise_strength,
                clahe_clip_limit=cfg.clahe_clip_limit,
                clahe_tile_size=cfg.clahe_tile_size,
                apply_mprnet=cfg.apply_mprnet,
                # MPRNet parameters
                n_feat=cfg.n_feat,
                scale_orsnetfeats=cfg.scale_orsnetfeats,
                num_cab=cfg.num_cab,
                # Performance
                max_image_size=cfg.max_image_size,
            )
            self.log("Stage 3 initialized")

    def initialize_stage4(self) -> None:
        """Initialize Stage 4: Door Detection & Classification."""
        if self.stage4_door is None:
            self.log("Initializing Stage 4: Door Detection & Classification")
            cfg = self.config.stage4
            # Use dynamic output_root for job-specific folder
            output_dir = Path(self.config.output_root)
            self.stage4_door = DoorDetectionClassification(
                stage3_output_dir=str(output_dir / "stage3_enhanced_frames"),
                output_base_dir=str(output_dir),
                door_detection_model_path=cfg.door_detection_model_path,
                door_conf_threshold=cfg.door_conf_threshold,
                door_classifier_model_path=cfg.door_classifier_model_path,
                min_door_width_ratio=cfg.min_door_width_ratio,
                min_door_height_ratio=cfg.min_door_height_ratio,
                device=cfg.device,
            )
            self.log("Stage 4 initialized")

    def initialize_stage5(self) -> None:
        """Initialize Stage 5: Damage Detection."""
        if self.stage5_damage is None:
            self.log("Initializing Stage 5: Damage Detection")
            cfg = self.config.stage5
            # Use dynamic output_root for job-specific folder
            output_dir = Path(self.config.output_root)
            self.stage5_damage = DamageDetection(
                stage3_output_dir=str(output_dir / "stage3_enhanced_frames"),
                output_base_dir=str(output_dir),
                damage_model_path=cfg.damage_model_path,
                conf_threshold=cfg.conf_threshold,
                img_size=cfg.img_size,
                device=cfg.device,
                minor_damage_threshold=cfg.minor_damage_threshold,
                moderate_damage_threshold=cfg.moderate_damage_threshold,
            )
            self.log("Stage 5 initialized")

    def initialize_stage6(self) -> None:
        """Initialize Stage 6: Wagon Number Extraction (OCR)."""
        if self.stage6_ocr is None:
            self.log("Initializing Stage 6: Wagon Number Extraction (OCR)")
            cfg = self.config.stage6
            # Use dynamic output_root for job-specific folder
            output_dir = str(self.config.output_root)
            self.stage6_ocr = WagonNumberExtractor(
                outputs_root=output_dir,
                stage3_dirname="stage3_enhanced_frames",
                craft_model_path=cfg.craft_model_path,
                ocr_server_url=cfg.ocr_server_url,
            )
            self.log("Stage 6 initialized")

    # ------------------------------------------------------------------
    # Stage run methods
    # ------------------------------------------------------------------
    def run_stage1(self, video_path: str, output_video_path: Optional[str] = None, on_wagon_counted: Optional[callable] = None) -> Dict:
        """Run Stage 1: Wagon Detection / Tracking / Counting."""
        self.log("=" * 80)
        self.log("=" * 80)
        self.log("STAGE 1: WAGON DETECTION / TRACKING / COUNTING")
        print("[MAIN PIPELINE] Entering Stage 1: Wagon Detection")
        self.log("=" * 80)
        self.initialize_stage1()
        start_time = time.time()

        wagon_data = self.stage1_detector.process_video(
            video_path=video_path,
            output_video_path=output_video_path,
            save_annotated_frames=self.config.stage1.save_annotated_frames,
            on_wagon_counted=on_wagon_counted,  # Pass callback for real-time updates
        )

        elapsed = time.time() - start_time
        self.stats["stage_times"]["stage1"] = elapsed
        self.stats["total_wagons"] = len(wagon_data)
        self.log(f"[1] Complete in {elapsed:.1f}s")
        self.log(f"Wagons detected: {len(wagon_data)}")
        return wagon_data

    def run_stage2(self, on_frame_assessed: Optional[callable] = None) -> Dict:
        """Run Stage 2: Quality Assessment."""
        self.log("=" * 80)
        self.log("STAGE 2: QUALITY ASSESSMENT")
        print("[MAIN PIPELINE] Entering Stage 2: Quality Assessment")
        self.log("=" * 80)
        self.initialize_stage2()
        start_time = time.time()

        quality_results = self.stage2_qa.process_all_wagons(on_frame_assessed=on_frame_assessed)

        elapsed = time.time() - start_time
        self.stats["stage_times"]["stage2"] = elapsed
        self.log(f"[2] Complete in {elapsed:.1f}s")
        return quality_results

    def run_stage3(self, on_frame_enhanced: Optional[callable] = None) -> Dict:
        """Run Stage 3: Image Enhancement."""
        self.log("=" * 80)
        self.log("STAGE 3: IMAGE ENHANCEMENT")
        print("[MAIN PIPELINE] Entering Stage 3: Image Enhancement")
        self.log("=" * 80)
        self.initialize_stage3()
        start_time = time.time()

        enhancement_results = self.stage3_enhancer.process_all_wagons(on_frame_enhanced=on_frame_enhanced)

        elapsed = time.time() - start_time
        self.stats["stage_times"]["stage3"] = elapsed
        self.log(f"[3] Complete in {elapsed:.1f}s")
        self.log(f"Wagons enhanced: {len(enhancement_results)}")
        return enhancement_results

    def run_stage4(self, on_door_counted: Optional[callable] = None) -> Dict:
        """Run Stage 4: Door Detection & Classification."""
        self.log("=" * 80)
        self.log("STAGE 4: DOOR DETECTION & CLASSIFICATION")
        print("[MAIN PIPELINE] Entering Stage 4: Door Detection")
        self.log("=" * 80)
        self.initialize_stage4()
        start_time = time.time()

        door_results = self.stage4_door.process_all_wagons(on_door_count_update=on_door_counted)

        elapsed = time.time() - start_time
        self.stats["stage_times"]["stage4"] = elapsed
        total_doors = sum(r["total_doors_detected"] for r in door_results.values())
        self.log(f"[4] Complete in {elapsed:.1f}s")
        self.log(f"Total doors detected: {total_doors}")
        return door_results

    def run_stage5(self, on_damage_processed: Optional[callable] = None) -> Dict:
        """Run Stage 5: Damage Detection."""
        self.log("=" * 80)
        self.log("STAGE 5: DAMAGE DETECTION")
        print("[MAIN PIPELINE] Entering Stage 5: Damage Detection")
        self.log("=" * 80)
        self.initialize_stage5()
        start_time = time.time()

        damage_results = self.stage5_damage.process_all_wagons(on_damage_processed=on_damage_processed)

        elapsed = time.time() - start_time
        self.stats["stage_times"]["stage5"] = elapsed
        wagons_with_damage = sum(
            1 for r in damage_results.values() if r["total_detections"] > 0
        )
        self.log(f"[5] Complete in {elapsed:.1f}s")
        self.log(f"Wagons with damage: {wagons_with_damage}/{len(damage_results)}")
        return damage_results

    def run_stage6(self, on_ocr_processed: Optional[callable] = None) -> Dict:
        """Run Stage 6: Wagon Number Extraction (OCR)."""
        self.log("=" * 80)
        self.log("STAGE 6: WAGON NUMBER EXTRACTION (OCR)")
        print("[MAIN PIPELINE] Entering Stage 6: OCR Extraction")
        self.log("=" * 80)
        self.initialize_stage6()
        start_time = time.time()

        # Use dynamic output_root for job-specific folder
        stage3_dir = Path(self.config.output_root) / "stage3_enhanced_frames"
        enhanced_images = sorted(stage3_dir.glob("*.jpg"))

        if not enhanced_images:
            self.log("No enhanced images found. Skipping Stage 6.")
            self.stats["stage_times"]["stage6"] = 0.0
            return {}

        # Run pipeline stages
        self.log(f"Running Stage 6 pipeline...")
        ocr_results_dict = self.stage6_ocr.run_full_pipeline(on_ocr_processed=on_ocr_processed)
        
        # Convert dictionary results to list for consistency with original interface
        ocr_results = list(ocr_results_dict.values())

        elapsed = time.time() - start_time
        self.stats["stage_times"]["stage6"] = elapsed

        total_numbers = sum(1 for r in ocr_results if r.detected_number)
        valid_numbers = sum(
            1
            for r in ocr_results
            for w in (r.candidate_numbers or [])
            # Simplified validation check for now
            if True
        )

        self.log(f"[6] Complete in {elapsed:.1f}s")
        self.log(f"Wagon numbers found: {total_numbers} (valid: {valid_numbers})")

        return {
            "results": ocr_results,
            "total_numbers": total_numbers,
            "valid_numbers": valid_numbers,
        }

    def run_aggregation(self, stage4: Dict, stage5: Dict, stage6: Dict) -> Dict:
        """Final aggregation after Stage 6."""
        self.log("=" * 80)
        self.log("FINAL AGGREGATION & REPORTING")
        print("[MAIN PIPELINE] Entering Aggregation Stage")
        self.log("=" * 80)
        start_time = time.time()

        final_results: Dict[str, Any] = {
            "pipeline_info": {
                "timestamp": datetime.now().isoformat(),
                "total_wagons": self.stats["total_wagons"],
                "stage_times": self.stats["stage_times"],
                "total_time": sum(self.stats["stage_times"].values()),
            },
            "stage4_doors": stage4,
            "stage5_damage": stage5,
            "stage4_doors": stage4,
            "stage5_damage": stage5,
            "stage6_ocr": {
                **stage6,
                "results": [asdict(r) for r in stage6["results"]] if "results" in stage6 and isinstance(stage6["results"], list) else stage6.get("results", [])
            },
        }

        output_dir = Path(self.config.output_root) / "final_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.config.aggregation.save_json:
            json_path = output_dir / f"final_results_{timestamp}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)
            self.log(f"JSON saved: {json_path}")

        elapsed = time.time() - start_time
        self.stats["stage_times"]["aggregation"] = elapsed
        self.log(f"[AGG] Complete in {elapsed:.1f}s")
        return final_results

    # ------------------------------------------------------------------
    # Complete pipeline
    # ------------------------------------------------------------------
    def run_complete_pipeline(
        self, video_path: str, output_video_path: Optional[str] = None
    ) -> Dict:
        """Run complete end-to-end pipeline."""
        self.stats["start_time"] = time.time()
        try:
            # Stage 1
            wagon_data = self.run_stage1(video_path, output_video_path)
            if len(wagon_data) == 0:
                self.log("No wagons detected. Aborting pipeline.")
                return {"status": "failed", "reason": "no_wagons_detected"}

            # Stage 2
            _ = self.run_stage2()

            # Stage 3
            _ = self.run_stage3()

            # Stage 4, 5, 6 (currently sequential)
            self.log("=" * 80)
            self.log("STAGES 4–6: DOORS, DAMAGE, OCR")
            self.log("=" * 80)

            door_results = self.run_stage4()
            damage_results = self.run_stage5()
            ocr_results = self.run_stage6()

            # Aggregation
            final_results = self.run_aggregation(
                stage4=door_results,
                stage5=damage_results,
                stage6=ocr_results,
            )

            self.stats["end_time"] = time.time()
            total_time = self.stats["end_time"] - self.stats["start_time"]
            self.print_final_summary(total_time)

            final_results["status"] = "success"
            final_results["total_time_sec"] = total_time
            return final_results

        except Exception as e:
            self.log(f"Pipeline failed: {e}", level="error")
            if self.config.debug_mode:
                import traceback

                traceback.print_exc()
            return {"status": "failed", "error": str(e)}

    def print_final_summary(self, total_time: float) -> None:
        """Print final pipeline summary."""
        print("=" * 80)
        print("PIPELINE COMPLETE!")
        print("=" * 80)
        print(f"Time        : {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"Total wagons: {self.stats['total_wagons']}")
        print("Stage breakdown:")
        for stage, elapsed in self.stats["stage_times"].items():
            pct = (elapsed / total_time) * 100 if total_time > 0 else 0.0
            print(f"  - {stage:12s}: {elapsed:6.1f}s ({pct:4.1f}%)")
        print(f"Output dir  : {self.config.output_root}")
        print("=" * 80)


# ----------------------------------------------------------------------
# Command-line interface
# ----------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wagon Inspection System - Complete Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --video inputs/wagon_video.mp4 --direction LR
  python main.py --video inputs/test.mp4 --output results --direction RL
  python main.py --video inputs/test.mp4 --device cuda --direction LR
  python main.py --video inputs/test.mp4 --debug --direction LR
""",
    )

    # Required arguments
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Input video file path",
    )

    # Pipeline options
    parser.add_argument(
        "--direction",
        type=str,
        choices=["LR", "RL"],
        default="LR",
        help="Wagon movement direction (LR=left-to-right, RL=right-to-left)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Output directory for results",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        help="Path to save annotated video (optional)",
    )

    # Device options
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "mps", "cpu"],
        default="cpu",
        help="Device for processing (auto, cuda, mps, or cpu)",
    )

    # Advanced options
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom config JSON file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    parser.add_argument(
        "--validate-models",
        action="store_true",
        help="Validate model files and exit",
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = PipelineConfig.load(args.config)
    else:
        config = get_config()

    # Resolve device
    # When you want full CUDA run: use `--device cuda`
    resolved_device = args.device if args.device != "auto" else "cpu"

    # Override config with CLI arguments
    config.stage1.direction = args.direction
    config.stage1.device = resolved_device
    config.stage3.device = resolved_device
    config.stage4.device = resolved_device
    config.stage5.device = resolved_device
    # Chandra OCR device gets the raw choice (can stay "auto" if you like)
    config.stage6.chandra_device = args.device

    config.output_root = Path(args.output)
    config.debug_mode = args.debug

    # Validate models only
    if args.validate_models:
        print("Validating model files...")
        if validate_models_exist(config):
            print("All model files validated successfully!")
            return 0
        else:
            print("Model validation failed!")
            return 1

    # Check video exists
    if not Path(args.video).exists():
        print(f"Video file not found: {args.video}")
        return 1

    # Validate models before running
    print("Checking required models...")
    if not validate_models_exist(config):
        print("Some model files are missing. Please download them first.")
        return 1

    # Initialize and run pipeline
    pipeline = WagonInspectionPipeline(config=config, verbose=True)
    results = pipeline.run_complete_pipeline(
        video_path=args.video,
        output_video_path=args.output_video,
    )

    if results.get("status") != "success":
        print("Pipeline failed:", results.get("error", "Unknown error"))
        return 1

    print("Pipeline completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
