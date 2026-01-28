"""
STAGE 2: Quality Assessment Module (Stricter blur)

Analyzes ONE best frame per wagon (from Stage 1) for:
- Blur (Laplacian, Tenengrad)
- Low-light (mean brightness, dark pixel ratio)

Stricter blur thresholds so visibly blurred best frames are detected.
"""

import cv2
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
from tqdm import tqdm

# =====================================================
# CONFIG – GLOBAL GUARDS (stricter)
# =====================================================
# For your 640x360 wagon crops these values force detection
# of clearly blurred frames while still allowing good ones.
MIN_LAP_HARD = 550.0       # below this, almost always blurred in your data
MIN_TEN_HARD = 9500.0      # below this, gradients noticeably weak

MIN_MEAN_BRIGHT_HARD = 40.0    # clearly darker than normal day shots
MAX_DARK_RATIO_HARD = 0.80     # very dark overall

# Relative factors against per-wagon medians
LAP_REL_FACTOR = 0.95          # 5% lower than own median => candidate blur
TEN_REL_FACTOR = 0.95
BRIGHT_REL_FACTOR = 0.9        # as before
DARK_REL_FACTOR = 1.1


@dataclass
class WagonQuality:
    """Quality metrics for the best frame of one wagon."""
    wagon_name: str
    image_path: str
    laplacian_score: float
    tenengrad_score: float
    mean_brightness: float
    dark_pixel_ratio: float
    lap_thresh_wagon: float
    ten_thresh_wagon: float
    bright_thresh_wagon: float
    dark_ratio_thresh_wagon: float
    blur_label: str
    light_label: str
    needs_enhancement: bool  # True if blur or low_light


class QualityAssessment:
    """
    STAGE 2: Quality Assessment System

    - Assumes Stage 1 already selected ONE best frame per wagon.
    - Reads best frame per wagon from stage1_output_dir/<wagon_name>/best_frame.jpg.
    - Computes blur and low-light metrics.
    - Produces JSON + CSV with blur/light labels per wagon.
    """

    def __init__(
        self,
        stage1_output_dir: str = "outputs/stage1_wagon_rois",
        output_base_dir: str = "outputs",
        resize_dim=(640, 360),
        best_frame_name: str = "best_frame.jpg",
    ):
        self.stage1_output_dir = Path(stage1_output_dir)
        self.output_base_dir = Path(output_base_dir)
        self.resize_dim = resize_dim
        self.best_frame_name = best_frame_name

        self.output_dir_results = self.output_base_dir / "stage2_quality_results"
        self.output_dir_results.mkdir(parents=True, exist_ok=True)

        print("✅ STAGE 2 output directory created:")
        print(f" - Results: {self.output_dir_results}")

    # =====================================================
    # METRICS
    # =====================================================

    @staticmethod
    def laplacian_score(img_gray: np.ndarray) -> float:
        return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())

    @staticmethod
    def tenengrad_score(img_gray: np.ndarray) -> float:
        gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        return float(np.mean(gx ** 2 + gy ** 2))

    @staticmethod
    def brightness_mean(img_gray: np.ndarray) -> float:
        return float(np.mean(img_gray))

    @staticmethod
    def dark_pixel_ratio(img_gray: np.ndarray, thresh: int = 50) -> float:
        return float(np.mean(img_gray < thresh))

    # =====================================================
    # CLASSIFICATION LOGIC
    # =====================================================

    @staticmethod
    def classify_blur(
        lap: float,
        ten: float,
        lap_th: float,
        ten_th: float,
    ) -> str:
        """
        Blur decision (stricter):

        - Hard blur:
            If Laplacian OR Tenengrad are below MIN_*_HARD -> blur.
        - Relative blur:
            If EITHER metric is below 0.95×median (its threshold)
            AND the other is not clearly much higher (sharp),
            then treat as blur.

        This catches many more blurred frames, including cases
        where only one metric drops sharply.
        """

        # 1) Hard blur for clearly soft images
        if lap < MIN_LAP_HARD or ten < MIN_TEN_HARD:
            return "blur"

        # 2) Relative blur
        lap_is_low = lap < lap_th
        ten_is_low = ten < ten_th

        # High-confidence sharp band: 20% above threshold
        lap_is_very_good = lap > lap_th * 1.20
        ten_is_very_good = ten > ten_th * 1.20

        # If one metric is low and the other is not very good -> blur
        if (lap_is_low and not ten_is_very_good) or (ten_is_low and not lap_is_very_good):
            return "blur"

        # 3) Everything else is sharp enough
        return "sharp"

    @staticmethod
    def classify_light(
        mean_b: float,
        dark_r: float,
        b_th: float,
        d_th: float,
    ) -> str:
        """
        Low-light decision:

        - Hard low-light:
            Very low brightness (< 40) or extremely high dark ratio (> 0.8).
        - Relative low-light:
            brightness < 0.9×median AND dark_ratio > 1.1×median.
        """
        if mean_b < MIN_MEAN_BRIGHT_HARD or dark_r > MAX_DARK_RATIO_HARD:
            return "low_light"

        if (mean_b < b_th) and (dark_r > d_th):
            return "low_light"

        return "normal_light"

    # =====================================================
    # CORE PIPELINE
    # =====================================================

    def _collect_best_frames(self) -> pd.DataFrame:
        """Collect ONE best frame per wagon from Stage 1 output."""
        if not self.stage1_output_dir.exists():
            raise ValueError(f"STAGE 1 output directory not found: {self.stage1_output_dir}")

        records: List[Dict[str, str]] = []

        for wagon_dir in sorted(self.stage1_output_dir.iterdir()):
            if not wagon_dir.is_dir():
                continue

            wagon_id = wagon_dir.name
            best_frame_path = wagon_dir / self.best_frame_name
            if not best_frame_path.exists():
                jpgs = list(wagon_dir.glob("*.jpg"))
                if not jpgs:
                    continue
                best_frame_path = jpgs[0]

            records.append(
                {
                    "wagon_id": wagon_id,
                    "frame_name": best_frame_path.name,
                    "frame_path": str(best_frame_path),
                }
            )

        df = pd.DataFrame(records)
        if df.empty:
            raise RuntimeError(f"No best frame images found in {self.stage1_output_dir}")

        print(f"✅ Found {len(df)} wagons with best frames")
        return df

    def _compute_metrics(self, df: pd.DataFrame, on_frame_assessed: Optional[callable] = None) -> pd.DataFrame:
        """Compute metrics for each best frame (one per wagon)."""
        df["laplacian_score"] = np.nan
        df["tenengrad_score"] = np.nan
        df["mean_brightness"] = np.nan
        df["dark_pixel_ratio"] = np.nan

        # Counter for callback
        count = 0

        for idx, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc="Computing quality metrics (best frames only)",
        ):
            img = cv2.imread(row["frame_path"], cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, self.resize_dim)

            df.at[idx, "laplacian_score"] = self.laplacian_score(img)
            df.at[idx, "tenengrad_score"] = self.tenengrad_score(img)
            df.at[idx, "mean_brightness"] = self.brightness_mean(img)
            df.at[idx, "dark_pixel_ratio"] = self.dark_pixel_ratio(img)
            
            # Call callback if provided
            count += 1
            if on_frame_assessed:
                try:
                    on_frame_assessed(count)
                except Exception as e:
                    pass # suppress callback errors

        df.dropna(inplace=True)
        return df

    def _apply_adaptive_thresholds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-wagon medians and adaptive thresholds."""
        df["lap_median_wagon"] = df.groupby("wagon_id")["laplacian_score"].transform("median")
        df["ten_median_wagon"] = df.groupby("wagon_id")["tenengrad_score"].transform("median")
        df["bright_median_wagon"] = df.groupby("wagon_id")["mean_brightness"].transform("median")
        df["dark_median_wagon"] = df.groupby("wagon_id")["dark_pixel_ratio"].transform("median")

        df["lap_thresh_wagon"] = df["lap_median_wagon"] * LAP_REL_FACTOR
        df["ten_thresh_wagon"] = df["ten_median_wagon"] * TEN_REL_FACTOR
        df["bright_thresh_wagon"] = df["bright_median_wagon"] * BRIGHT_REL_FACTOR
        df["dark_ratio_thresh_wagon"] = df["dark_median_wagon"] * DARK_REL_FACTOR

        return df

    def _classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify blur and low-light for each wagon (best frame only)."""
        df["blur_label"] = df.apply(
            lambda r: self.classify_blur(
                r["laplacian_score"],
                r["tenengrad_score"],
                r["lap_thresh_wagon"],
                r["ten_thresh_wagon"],
            ),
            axis=1,
        )

        df["light_label"] = df.apply(
            lambda r: self.classify_light(
                r["mean_brightness"],
                r["dark_pixel_ratio"],
                r["bright_thresh_wagon"],
                r["dark_ratio_thresh_wagon"],
            ),
            axis=1,
        )

        df["needs_enhancement"] = (
            (df["blur_label"] == "blur") | (df["light_label"] == "low_light")
        )

        return df

    def process_all_wagons(self, on_frame_assessed: Optional[callable] = None) -> Dict[str, Any]:
        """
        Main entry for Stage 2.

        Returns:
            Dict[wagon_id] = {
                'wagon_id': str,
                'image_path': str,
                'metrics': {...},
                'labels': {...}
            }
        """
        df = self._collect_best_frames()
        df = self._compute_metrics(df, on_frame_assessed=on_frame_assessed)
        df = self._apply_adaptive_thresholds(df)
        df = self._classify(df)

        csv_path = self.output_dir_results / "stage2_quality_report.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n📁 CSV report saved at: {csv_path}")

        results: Dict[str, Any] = {}
        for _, row in df.iterrows():
            wagon_id = row["wagon_id"]
            results[wagon_id] = {
                "wagon_id": wagon_id,
                "image_path": row["frame_path"],
                "metrics": {
                    "laplacian_score": float(row["laplacian_score"]),
                    "tenengrad_score": float(row["tenengrad_score"]),
                    "mean_brightness": float(row["mean_brightness"]),
                    "dark_pixel_ratio": float(row["dark_pixel_ratio"]),
                    "lap_thresh_wagon": float(row["lap_thresh_wagon"]),
                    "ten_thresh_wagon": float(row["ten_thresh_wagon"]),
                    "bright_thresh_wagon": float(row["bright_thresh_wagon"]),
                    "dark_ratio_thresh_wagon": float(row["dark_ratio_thresh_wagon"]),
                },
                "labels": {
                    "blur_label": str(row["blur_label"]),
                    "light_label": str(row["light_label"]),
                    "needs_enhancement": bool(row["needs_enhancement"]),
                },
            }

        json_path = self.output_dir_results / "stage2_quality_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"📄 JSON summary saved at: {json_path}")

        print("\n✅ Blur + Low-Light classification completed\n")
        print(df.groupby(["blur_label", "light_label"]).size())
        print(f"\nTotal wagons: {df['wagon_id'].nunique()}")
        print(f"Blurred wagons: {(df['blur_label'] == 'blur').sum()}")
        print(f"Low-light wagons: {(df['light_label'] == 'low_light').sum()}")

        return results


# ================= STANDALONE TEST =================

if __name__ == "__main__":
    qa = QualityAssessment(
        stage1_output_dir=r"C:\Users\Viranch Patel\Desktop\Wagon_Inspection_System\outputs\stage1_wagon_rois",
        output_base_dir=r"C:\Users\Viranch Patel\Desktop\Wagon_Inspection_System\outputs",
        resize_dim=(640, 360),
        best_frame_name="best_frame.jpg",
    )
    res = qa.process_all_wagons()
    print("\nSTAGE 2 COMPLETE - RESULTS FOR STAGE 3\n")
    for wagon, data in sorted(
        res.items(),
        key=lambda x: int(x[0].split('_')[0].replace('wagon', ''))
        if x[0].startswith("wagon")
        else 999999,
    ):
        print(
            f"{wagon}: "
            f"blur={data['labels']['blur_label']}, "
            f"light={data['labels']['light_label']}, "
            f"needs_enhancement={data['labels']['needs_enhancement']}"
        )
