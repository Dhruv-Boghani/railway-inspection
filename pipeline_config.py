"""
Wagon Inspection System - Configuration

Centralized configuration for all pipeline stages
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# ==================== PATHS ====================

BASE_DIR = Path(__file__).parent

MODELS_DIR = BASE_DIR / "models"
WEIGHTS_DIR = MODELS_DIR / "weights"
REPOS_DIR = MODELS_DIR / "repos"

# Top Camera Model Paths
TOP_DETECTION_MODEL = WEIGHTS_DIR / "top_detection_best.pt"
TOP_DAMAGE_MODEL = WEIGHTS_DIR / "top_damage_best.pt"

CONFIG_DIR = BASE_DIR / "config"
OUTPUTS_DIR = BASE_DIR / "outputs"
INPUTS_DIR = BASE_DIR / "inputs"

# ==================== STAGE 1: WAGON DETECTION / TRACKING / COUNTING ====================

@dataclass
class Stage1Config:
    """Stage 1: Wagon Detection, Tracking, and Counting (WagonDetectionCounter)"""

    # Model path
    model_path: str = str(WEIGHTS_DIR / "yolo12s_wagon_detection_&_counting_best.pt")

    # Video processing
    direction: str = "LR"  # "LR" (left-to-right) or "RL" (right-to-left)
    device: str = "cpu"    # "cpu", "cuda", or "mps"
    img_size: int = 960

    # Detection parameters
    conf_threshold: float = 0.15  # Detection confidence threshold

    # ROI counting zone parameters
    min_frames_in_roi: int = 3    # Minimum frames wagon must be in ROI to count
    roi_padding: int = 15         # Padding when extracting wagon ROIs (pixels)

    # Quality filtering parameters
    min_confidence_for_storage: float = 0.25  # Only store frames with conf >= this
    min_bbox_area: int = 15000                # Minimum bbox area for quality frames
    max_frames_per_wagon: int = 10            # Store max N best frames per wagon

    # Output options
    output_base_dir: str = str(OUTPUTS_DIR)
    save_annotated_frames: bool = False       # Save individual annotated frames (slow)


# ==================== STAGE 2: QUALITY ASSESSMENT ====================

@dataclass
class Stage2Config:
    """Stage 2: Quality Assessment (QualityAssessment)"""

    # Input/Output directories
    stage1_output_dir: str = str(OUTPUTS_DIR / "stage1_wagon_rois")
    output_base_dir: str = str(OUTPUTS_DIR)

    # Blur detection thresholds
    blur_laplacian_threshold: float = 290.0   # Laplacian variance threshold
    blur_tenengrad_threshold: float = 5200.0  # Tenengrad threshold

    # Low-light detection
    lowlight_p10_threshold: float = 4.0       # Brightness P10 threshold (darkness)

    # Noise detection
    noise_threshold: float = 9.0              # Noise score threshold

    # Output options
    save_quality_reports: bool = True
    save_selected_frames: bool = True


# ==================== STAGE 3: IMAGE ENHANCEMENT ====================

@dataclass
class Stage3Config:
    """Stage 3: Image Enhancement (ImageEnhancer)"""

    # Input/Output directories
    stage2_output_dir: str = str(OUTPUTS_DIR / "stage2_selected_frames")
    stage2_results_path: str = str(
        OUTPUTS_DIR / "stage2_quality_results" / "stage2_quality_summary.json"
    )
    output_base_dir: str = str(OUTPUTS_DIR)

    # MPRNet model
    mprnet_model_path: str = str(WEIGHTS_DIR / "MPRNET.pth")
    apply_mprnet: bool = True

    # Enhancement parameters
    denoise_strength: int = 5        # Bilateral filter strength
    clahe_clip_limit: float = 2.0    # CLAHE clip limit
    clahe_tile_size: int = 8         # CLAHE tile grid size

    # MPRNet architecture params
    n_feat: int = 80
    scale_orsnetfeats: int = 32
    num_cab: int = 6

    # Device
    device: str = "cpu"              # "cpu" or "cuda"

    # Output options
    max_image_size: int = 1024


# ==================== STAGE 4: DOOR DETECTION & CLASSIFICATION ====================

@dataclass
class Stage4Config:
    """Stage 4: Door Detection and Classification (DoorDetectionClassification)"""

    # Input/Output directories
    stage3_output_dir: str = str(OUTPUTS_DIR / "stage3_enhanced_frames")
    output_base_dir: str = str(OUTPUTS_DIR)

    # Model paths
    door_detection_model_path: str = str(WEIGHTS_DIR / "yolo12n_Door_ROI_best.pt")
    door_classifier_model_path: str = str(
        WEIGHTS_DIR / "few_shot_classifier_door_best_model.pth"
    )

    # Detection parameters
    door_conf_threshold: float = 0.25  # Door detection confidence

    # Filtering parameters
    min_door_width_ratio: float = 0.3   # Minimum door width as ratio of average
    min_door_height_ratio: float = 0.3  # Minimum door height as ratio of average

    # Classification
    door_classes: List[str] = field(
        default_factory=lambda: ["good", "damaged", "missing"]
    )
    use_tta: bool = True               # Test-time augmentation for classification

    # Device
    device: str = "cpu"                # "cpu" or "cuda"

    # Output options
    save_door_rois: bool = True
    save_annotated_images: bool = True


# ==================== STAGE 5: DAMAGE DETECTION ====================

@dataclass
class Stage5Config:
    """Stage 5: Damage Detection (DamageDetection)"""

    # Input/Output directories
    stage3_output_dir: str = str(OUTPUTS_DIR / "stage3_enhanced_frames")
    output_base_dir: str = str(OUTPUTS_DIR)

    # Model path
    damage_model_path: str = str(WEIGHTS_DIR / "yolo8s-seg_damage_best.pt")

    # Detection parameters
    conf_threshold: float = 0.25  # Damage detection confidence
    img_size: int = 640          # Input image size
    device: str = "cpu"          # "cpu" or "cuda"

    # Severity thresholds (% of image area)
    minor_damage_threshold: float = 0.5  # 0–0.5% = minor
    moderate_damage_threshold: float = 2.0  # 0.5–2% = moderate, >2% = severe

    # Output options
    save_annotated_images: bool = True
    save_damage_masks: bool = True
    save_damage_rois: bool = True


# ==================== STAGE 6: WAGON NUMBER EXTRACTION (OCR) ====================

@dataclass
class Stage6Config:
    """Stage 6: Wagon Number Extraction - CRAFT + Chandra OCR Server"""

    # Input/Output directories
    outputs_root: str = str(OUTPUTS_DIR)
    stage3_dirname: str = "stage3_enhanced_frames"

    # CRAFT Text Detection
    craft_model_path: str = str(WEIGHTS_DIR / "craft_mlt_25k.pth")

    # Chandra OCR Server
    ocr_server_url: str = "http://127.0.0.1:8001/ocr"



# ==================== RESULTS AGGREGATION / REPORTING ====================

@dataclass
class AggregationConfig:
    """Final Results Aggregation and Reporting (after Stage 6)"""

    # Output formats
    save_json: bool = True
    save_csv: bool = True
    save_excel: bool = False

    # Report generation
    generate_summary_report: bool = True
    generate_per_wagon_report: bool = True

    # Visualization
    create_annotated_video: bool = False  # Create annotated output video
    video_fps: int = 30
    video_codec: str = "mp4v"

    # Data retention
    keep_intermediate_files: bool = True
    compress_outputs: bool = False


# ==================== MAIN PIPELINE CONFIG ====================

@dataclass
class PipelineConfig:
    """Main Pipeline Configuration"""

    # Stage configurations (1–6)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    stage3: Stage3Config = field(default_factory=Stage3Config)
    stage4: Stage4Config = field(default_factory=Stage4Config)  # doors
    stage5: Stage5Config = field(default_factory=Stage5Config)  # damage
    stage6: Stage6Config = field(default_factory=Stage6Config)  # OCR

    # Final aggregation / reporting
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)

    # Global settings
    verbose: bool = True
    debug_mode: bool = False
    log_level: str = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"

    # Parallel processing (for stages 4–6, if you add later)
    enable_parallel_stage4_6: bool = False
    num_workers: int = 3

    # Output structure
    output_root: Path = OUTPUTS_DIR
    create_timestamped_folders: bool = True
    
    # Top Camera Models (added for top camera angle support)
    top_detection_model_path: Path = TOP_DETECTION_MODEL
    top_damage_model_path: Path = TOP_DAMAGE_MODEL

    def __post_init__(self):
        """Validate and create directories."""
        # Create essential directories
        self.output_root.mkdir(exist_ok=True)
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        REPOS_DIR.mkdir(parents=True, exist_ok=True)
        INPUTS_DIR.mkdir(exist_ok=True)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        from dataclasses import asdict

        return asdict(self)

    def save(self, path: str):
        """Save configuration to JSON file."""
        import json

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path_obj, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Config saved: {path_obj}")

    @classmethod
    def load(cls, path: str) -> "PipelineConfig":
        """Load configuration from JSON file (simple version)."""
        import json

        with open(path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        # Simple load: construct default and let code override as needed
        return cls()


# ==================== DEFAULT CONFIG ====================

DEFAULT_CONFIG = PipelineConfig()


# ==================== HELPER FUNCTIONS ====================

def get_config() -> PipelineConfig:
    """Get default pipeline configuration."""
    return DEFAULT_CONFIG


def create_custom_config(**kwargs) -> PipelineConfig:
    """Create custom configuration with overrides."""
    return PipelineConfig(**kwargs)


def validate_models_exist(config: PipelineConfig) -> bool:
    """Validate that all required model files exist."""
    required_models = [
        config.stage1.model_path,
        config.stage3.realesrgan_model_path,
        config.stage4.door_detection_model_path,
        config.stage4.door_classifier_model_path,
        config.stage5.damage_model_path,
        config.stage6.craft_model_path,
    ]

    missing = []
    for model_path in required_models:
        if not Path(model_path).exists():
            missing.append(model_path)

    if missing:
        print("Missing model files:")
        for m in missing:
            print(f" - {m}")
        return False

    print("All model files found!")
    return True


def setup_logging(config: PipelineConfig):
    """Setup logging for pipeline."""
    import logging
    from datetime import datetime

    # Create logs directory
    logs_dir = BASE_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"pipeline_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    logger = logging.getLogger("WagonInspection")
    logger.info(f"Logging initialized: {log_file}")
    return logger


# ==================== TESTING ====================

if __name__ == "__main__":
    print("=" * 80)
    print("WAGON INSPECTION SYSTEM - CONFIGURATION TEST")
    print("=" * 80)

    config = get_config()

    print("\nDirectory Structure:")
    print(f" Base   : {BASE_DIR}")
    print(f" Models : {MODELS_DIR}")
    print(f" Weights: {WEIGHTS_DIR}")
    print(f" Outputs: {OUTPUTS_DIR}")

    print("\nStage 1 (Wagon Detection):")
    print(f" Model      : {config.stage1.model_path}")
    print(f" Direction  : {config.stage1.direction}")
    print(f" Device     : {config.stage1.device}")
    print(f" Conf Thresh: {config.stage1.conf_threshold}")
    print(f" Max Frames : {config.stage1.max_frames_per_wagon}")

    print("\nStage 2 (Quality Assessment):")
    print(f" Blur Laplacian: {config.stage2.blur_laplacian_threshold}")
    print(f" Blur Tenengrad: {config.stage2.blur_tenengrad_threshold}")
    print(f" Low-light P10 : {config.stage2.lowlight_p10_threshold}")
    print(f" Noise         : {config.stage2.noise_threshold}")

    print("\nStage 3 (Enhancement):")
    print(f" Real-ESRGAN: {config.stage3.realesrgan_model_path}")
    print(f" Scale      : {config.stage3.realesrgan_scale}x")
    print(f" Apply      : {config.stage3.apply_realesrgan}")

    print("\nStage 4 (Door Detection & Classification):")
    print(f" Detection : {config.stage4.door_detection_model_path}")
    print(f" Classifier: {config.stage4.door_classifier_model_path}")
    print(f" Classes   : {config.stage4.door_classes}")

    print("\nStage 5 (Damage Detection):")
    print(f" Model      : {config.stage5.damage_model_path}")
    print(f" Conf       : {config.stage5.conf_threshold}")
    print(
        f" Thresholds : Minor={config.stage5.minor_damage_threshold}%, "
        f"Moderate={config.stage5.moderate_damage_threshold}%"
    )

    print("\nStage 6 (OCR):")
    print(f" CRAFT   : {config.stage6.craft_model_path}")
    print(f" Chandra : {config.stage6.chandra_model_name}")
    print(f" 8-bit   : {config.stage6.chandra_use_8bit}")

    print("\nValidating models...")
    validate_models_exist(config)

    print("\nSaving config...")
    config.save("config/pipeline_config.json")

    print("\n" + "=" * 80)
    print("Configuration test complete!")
    print("=" * 80)
