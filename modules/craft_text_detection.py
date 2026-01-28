"""
CRAFT Text Detection Module (Wrapper for Official Repo)

Stage 6: CRAFT + OCR

- CRAFTDetector: low-level wrapper over official CRAFT-pytorch repo.
- Stage6CraftTextDetector: pipeline wrapper that:
  1) Runs CRAFT raw detection.
  2) Adds 15% padding to each text box.
  3) Merges overlapping / near padded boxes.
  4) Saves annotated wagon frames with merged padded boxes.
  5) Crops merged padded boxes.
  6) Saves crops into stage6_craft_crops with wagon-prefixed names.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import torch
import json

# Add official CRAFT repo to path
CRAFT_REPO_PATH = Path(__file__).parent.parent / "models" / "repos" / "craft"
sys.path.insert(0, str(CRAFT_REPO_PATH))

from craft import CRAFT
import imgproc


class CRAFTDetector:
    """CRAFT Text Detector (Official Implementation Wrapper)."""

    def __init__(
        self,
        model_path: str = "models/weights/craft_mlt_25k.pth",
        text_threshold: float = 0.7,
        link_threshold: float = 0.3,
        low_text: float = 0.3,
        canvas_size: int = 1600,
        mag_ratio: float = 2.0,
        device: str = "cpu",
        refine: bool = False,
        refine_model_path: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        self.device = device
        self.refine = refine
        self.refine_model_path = refine_model_path

        self.net = self.load_craft_model()
        self.refine_net = None
        if self.refine and self.refine_model_path:
            self.refine_net = self.load_refiner_model()

    def load_craft_model(self):
        print(f" Loading CRAFT model from {self.model_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"CRAFT model not found: {self.model_path}\n"
                f"Download it using:\n"
                f"  wget https://huggingface.co/spaces/amitesh863/craft/resolve/main/craft_mlt_25k.pth "
                f"-O models/weights/craft_mlt_25k.pth"
            )

        net = CRAFT()
        print(" Loading weights...")
        if self.device == "cuda":
            net.load_state_dict(self.copyStateDict(torch.load(str(self.model_path))))
            net = net.cuda()
            net = torch.nn.DataParallel(net)
        else:
            net.load_state_dict(
                self.copyStateDict(
                    torch.load(str(self.model_path), map_location="cpu")
                )
            )
        net.eval()
        print(" CRAFT model loaded successfully")
        return net

    def load_refiner_model(self):
        from refinenet import RefineNet

        print(" Loading RefineNet model...")
        refine_net = RefineNet()
        if self.device == "cuda":
            refine_net.load_state_dict(
                self.copyStateDict(torch.load(self.refine_model_path))
            )
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(
                self.copyStateDict(
                    torch.load(self.refine_model_path, map_location="cpu")
                )
            )
        refine_net.eval()
        print(" RefineNet loaded")
        return refine_net

    @staticmethod
    def copyStateDict(state_dict):
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = {}
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict

    def test_net(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        img_resized, target_ratio, _ = imgproc.resize_aspect_ratio(
            image,
            self.canvas_size,
            interpolation=cv2.INTER_LINEAR,
            mag_ratio=self.mag_ratio,
        )
        ratio_h = ratio_w = 1 / target_ratio

        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)
        x = x.unsqueeze(0)
        if self.device == "cuda":
            x = x.cuda()

        with torch.no_grad():
            y, feature = self.net(x)

        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()

        if self.refine_net is not None:
            with torch.no_grad():
                y_refiner = self.refine_net(y, feature)
            score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

        return score_text, score_link, ratio_h, ratio_w

    def get_boxes_from_scores(
        self,
        score_text: np.ndarray,
        score_link: np.ndarray,
        ratio_h: float,
        ratio_w: float,
    ) -> List[np.ndarray]:
        img_h, img_w = score_text.shape

        _, text_score = cv2.threshold(
            score_text, self.low_text, 1, cv2.THRESH_BINARY
        )
        _, link_score = cv2.threshold(
            score_link, self.link_threshold, 1, cv2.THRESH_BINARY
        )

        text_score_comb = np.clip(text_score + link_score, 0, 1)

        nLabels, labels, stats, _ = cv2.connectedComponentsWithStats(
            text_score_comb.astype(np.uint8), connectivity=4
        )

        det: List[np.ndarray] = []
        for k in range(1, nLabels):
            size = stats[k, cv2.CC_STAT_AREA]
            if size < 50:
                continue

            if np.max(score_text[labels == k]) < self.text_threshold:
                continue

            x = stats[k, cv2.CC_STAT_LEFT]
            y = stats[k, cv2.CC_STAT_TOP]
            w = stats[k, cv2.CC_STAT_WIDTH]
            h = stats[k, cv2.CC_STAT_HEIGHT]

            aspect = max(w, h) / max(1, min(w, h))
            if aspect > 10 and h > w:
                continue

            niter = int(np.sqrt(size * min(w, h) / (w * h)) * 2)
            sx, ex = max(x - niter, 0), min(x + w + niter + 1, img_w)
            sy, ey = max(y - niter, 0), min(y + h + niter + 1, img_h)

            box = np.array([[sx, sy], [ex, sy], [ex, ey], [sx, ey]], dtype=np.float32)
            box[:, 0] *= ratio_w * 2
            box[:, 1] *= ratio_h * 2
            det.append(box)

        return det

    def filter_wagon_number_boxes(
        self,
        boxes: List[np.ndarray],
        image_shape: Tuple[int, int, int],
        min_area: int = 500,
        max_rel_height: float = 0.25,
        min_rel_height: float = 0.02,
    ) -> List[np.ndarray]:
        H, W = image_shape[:2]
        filtered: List[np.ndarray] = []

        for b in boxes:
            xs = b[:, 0]
            ys = b[:, 1]
            x_min, x_max = float(xs.min()), float(xs.max())
            y_min, y_max = float(ys.min()), float(ys.max())
            w = x_max - x_min
            h = y_max - y_min
            area = w * h
            if area < min_area:
                continue

            rel_h = h / H
            if rel_h < min_rel_height or rel_h > max_rel_height:
                continue

            # Removed bottom 20% exclusion - wagon numbers can be at any position

            filtered.append(b)

        return filtered

    def pad_boxes(
        self,
        boxes: List[np.ndarray],
        image_shape: Tuple[int, int, int],
        padding_ratio: float = 0.15,
    ) -> List[np.ndarray]:
        """
        Add padding to all sides of each box.
        padding_ratio = 0.15 means 15% padding on width and height.
        """
        H, W = image_shape[:2]
        padded_boxes: List[np.ndarray] = []

        for b in boxes:
            xs = b[:, 0]
            ys = b[:, 1]
            x_min = float(xs.min())
            y_min = float(ys.min())
            x_max = float(xs.max())
            y_max = float(ys.max())

            box_w = x_max - x_min
            box_h = y_max - y_min
            if box_w <= 0 or box_h <= 0:
                continue

            pad_x = box_w * padding_ratio
            pad_y = box_h * padding_ratio

            x_min_p = max(0.0, x_min - pad_x)
            y_min_p = max(0.0, y_min - pad_y)
            x_max_p = min(float(W), x_max + pad_x)
            y_max_p = min(float(H), y_max + pad_y)

            padded_boxes.append(
                np.array(
                    [
                        [x_min_p, y_min_p],
                        [x_max_p, y_min_p],
                        [x_max_p, y_max_p],
                        [x_min_p, y_max_p],
                    ],
                    dtype=np.float32,
                )
            )

        return padded_boxes

    def merge_padded_boxes(
        self,
        boxes: List[np.ndarray],
        iou_threshold: float = 0.3,
        center_distance_ratio: float = 0.08,
    ) -> List[np.ndarray]:
        """
        Merge padded boxes that overlap (IoU) or whose centers are very close.
        Uses padded boxes only; returns merged padded boxes.
        """
        if not boxes:
            return boxes

        rects = []
        for b in boxes:
            xs = b[:, 0]
            ys = b[:, 1]
            rects.append(
                [
                    float(xs.min()),
                    float(ys.min()),
                    float(xs.max()),
                    float(ys.max()),
                ]
            )

        rects = np.array(rects, dtype=np.float32)
        centers = np.column_stack(
            (
                0.5 * (rects[:, 0] + rects[:, 2]),
                0.5 * (rects[:, 1] + rects[:, 3]),
            )
        )

        used = np.zeros(len(rects), dtype=bool)
        merged_boxes: List[np.ndarray] = []

        for i in range(len(rects)):
            if used[i]:
                continue
            group = [i]
            used[i] = True
            for j in range(i + 1, len(rects)):
                if used[j]:
                    continue

                # IoU
                x1 = max(rects[i][0], rects[j][0])
                y1 = max(rects[i][1], rects[j][1])
                x2 = min(rects[i][2], rects[j][2])
                y2 = min(rects[i][3], rects[j][3])
                inter = 0.0
                if x2 > x1 and y2 > y1:
                    inter = (x2 - x1) * (y2 - y1)
                area_i = (rects[i][2] - rects[i][0]) * (rects[i][3] - rects[i][1])
                area_j = (rects[j][2] - rects[j][0]) * (rects[j][3] - rects[j][1])
                union = area_i + area_j - inter
                iou_val = inter / union if union > 0 else 0.0

                # center distance
                cx_i, cy_i = centers[i]
                cx_j, cy_j = centers[j]
                dist = np.hypot(cx_i - cx_j, cy_i - cy_j)
                diag_i = np.hypot(
                    rects[i][2] - rects[i][0], rects[i][3] - rects[i][1]
                )
                diag_j = np.hypot(
                    rects[j][2] - rects[j][0], rects[j][3] - rects[j][1]
                )
                scale = max(diag_i, diag_j)
                close_centers = dist < center_distance_ratio * scale

                if iou_val > iou_threshold or close_centers:
                    used[j] = True
                    group.append(j)

            g_rects = rects[group]
            x1 = float(np.min(g_rects[:, 0]))
            y1 = float(np.min(g_rects[:, 1]))
            x2 = float(np.max(g_rects[:, 2]))
            y2 = float(np.max(g_rects[:, 3]))
            merged_boxes.append(
                np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            )

        return merged_boxes

    def detect(
        self,
        image_path: str,
    ) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        """
        Detection pipeline (steps 1–3):
          1) raw boxes
          2) filter wagon-like boxes
          3) pad them
          4) merge padded boxes
        Returns final merged padded boxes.
        """
        image = imgproc.loadImage(str(image_path))
        score_text, score_link, ratio_h, ratio_w = self.test_net(image)
        raw_boxes = self.get_boxes_from_scores(score_text, score_link, ratio_h, ratio_w)
        filtered_boxes = self.filter_wagon_number_boxes(raw_boxes, image.shape)
        padded_boxes = self.pad_boxes(filtered_boxes, image.shape, padding_ratio=0.15)
        merged_boxes = self.merge_padded_boxes(
            padded_boxes, iou_threshold=0.3, center_distance_ratio=0.08
        )
        return merged_boxes, score_text, score_link

    def visualize(
        self,
        image_path: str,
        boxes: List[np.ndarray],
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Step 4: annotate wagon frame with final merged padded boxes.
        """
        image = cv2.imread(str(image_path))
        for box in boxes:
            poly = np.array(box).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [poly], True, color=(0, 255, 0), thickness=2)
        if output_path:
            cv2.imwrite(output_path, image)
        return image

    def extract_text_rois(
        self,
        image_path: str,
        boxes: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Step 5: crop using final merged padded boxes (no extra padding here).
        """
        image = cv2.imread(str(image_path))
        h, w = image.shape[:2]
        rois: List[np.ndarray] = []

        for box in boxes:
            x_coords = box[:, 0]
            y_coords = box[:, 1]
            x_min = max(0, int(np.min(x_coords)))
            y_min = max(0, int(np.min(y_coords)))
            x_max = min(w, int(np.max(x_coords)))
            y_max = min(h, int(np.max(y_coords)))

            if x_max <= x_min or y_max <= y_min:
                continue

            roi = image[y_min:y_max, x_min:x_max]
            if roi.size > 0:
                rois.append(roi)

        return rois


class Stage6CraftTextDetector:
    """
    Stage 6 wrapper around CRAFTDetector with clean output structure.

    Output structure under output_base_dir:

        stage6_craft_crops/
            wagon{n}_id_{tid}_crop_01.jpg
            wagon{n}_id_{tid}_crop_02.jpg
            ...

        stage6_craft_annotated/
            <wagon_name>_stage6_craft_annotated.jpg

        stage6_craft_results/
            stage6_craft_summary.json
            stage6_craft_boxes.json
    """

    def __init__(
        self,
        output_base_dir: str = "outputs",
        craft_model_path: str = "models/weights/craft_mlt_25k.pth",
        device: str = "cpu",
        text_threshold: float = 0.7,
        link_threshold: float = 0.3,
        low_text: float = 0.3,
        canvas_size: int = 1600,
        mag_ratio: float = 2.0,
    ):
        self.output_base_dir = Path(output_base_dir)
        self.stage6_root = self.output_base_dir

        self.dir_crops = self.stage6_root / "stage6_craft_crops"
        self.dir_annotated = self.stage6_root / "stage6_craft_annotated"
        self.dir_results = self.stage6_root / "stage6_craft_results"

        self.dir_crops.mkdir(parents=True, exist_ok=True)
        self.dir_annotated.mkdir(parents=True, exist_ok=True)
        self.dir_results.mkdir(parents=True, exist_ok=True)

        print("Stage 6 CRAFT output dirs:")
        print(f" - Crops:      {self.dir_crops}")
        print(f" - Annotated:  {self.dir_annotated}")
        print(f" - Results:    {self.dir_results}")

        self.summary: Dict[str, Any] = {}
        self.boxes_dict: Dict[str, Any] = {}

        self.craft = CRAFTDetector(
            model_path=craft_model_path,
            text_threshold=text_threshold,
            link_threshold=link_threshold,
            low_text=low_text,
            canvas_size=canvas_size,
            mag_ratio=mag_ratio,
            device=device,
        )

        self.valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def _parse_wagon_name(self, wagon_name: str) -> Dict[str, Any]:
        """
        Parse wagon number and track_id from names like 'wagon3_id_25'.
        """
        info: Dict[str, Any] = {"raw_name": wagon_name}
        try:
            if "wagon" in wagon_name and "_id_" in wagon_name:
                prefix, tid = wagon_name.split("_id_")
                num_str = prefix.replace("wagon", "")
                info["wagon_number"] = int(num_str)
                info["track_id"] = int(tid)
        except Exception:
            pass
        return info

    def _crop_filename(self, wagon_name: str, crop_idx: int) -> str:
        info = self._parse_wagon_name(wagon_name)
        if "wagon_number" in info and "track_id" in info:
            return f"wagon{info['wagon_number']}_id_{info['track_id']}_crop_{crop_idx:02d}.jpg"
        else:
            return f"{wagon_name}_crop_{crop_idx:02d}.jpg"

    def process_wagon_image(self, wagon_name: str, image_path: str) -> Dict[str, Any]:
        """
        Run stage 6 for a single wagon image (steps 1–6).
        """
        p = Path(image_path)
        if not p.is_file():
            raise ValueError(
                f"process_wagon_image expects a single image file, got: {image_path}"
            )

        wagon_info = self._parse_wagon_name(wagon_name)

        # Steps 1–3 (plus filtering) inside detect
        boxes, score_text, score_link = self.craft.detect(str(p))

        # Step 4: annotated frame
        annotated_filename = f"{wagon_name}_stage6_craft_annotated.jpg"
        annotated_path = self.dir_annotated / annotated_filename
        _ = self.craft.visualize(str(p), boxes, str(annotated_path))

        # Steps 5–6: crop and save
        rois = self.craft.extract_text_rois(str(p), boxes)
        crop_paths: List[str] = []
        for idx, roi in enumerate(rois, start=1):
            crop_name = self._crop_filename(wagon_name, idx)
            crop_path = self.dir_crops / crop_name
            cv2.imwrite(str(crop_path), roi)
            crop_paths.append(str(crop_path))

        boxes_list = [b.astype(float).tolist() for b in boxes]
        result = {
            "wagon_name": wagon_name,
            "wagon_info": wagon_info,
            "source_image_path": str(p),
            "num_boxes": len(boxes),
            "num_crops": len(crop_paths),
            "annotated_image_path": str(annotated_path),
            "crop_paths": crop_paths,
            "boxes": boxes_list,
        }

        self.summary[wagon_name] = {
            "wagon_name": wagon_name,
            "wagon_info": wagon_info,
            "source_image_path": str(p),
            "num_craft_boxes": len(boxes),
            "num_craft_crops": len(crop_paths),
            "annotated_image_path": str(annotated_path),
        }
        self.boxes_dict[wagon_name] = {
            "wagon_name": wagon_name,
            "boxes": boxes_list,
            "crop_paths": crop_paths,
        }

        return result

    def process_folder(self, folder_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Process all images in a folder. Each image is treated as one wagon.
        """
        folder = Path(folder_path)
        if not folder.is_dir():
            raise ValueError(f"process_folder expects a directory, got: {folder_path}")

        results: Dict[str, Dict[str, Any]] = {}
        for img_path in sorted(folder.iterdir()):
            if img_path.suffix.lower() not in self.valid_exts:
                continue
            wagon_name = img_path.stem
            print(f" Processing wagon image: {wagon_name}")
            res = self.process_wagon_image(wagon_name, str(img_path))
            results[wagon_name] = res

        return results

    def save_stage6_results(self):
        summary_path = self.dir_results / "stage6_craft_summary.json"
        boxes_path = self.dir_results / "stage6_craft_boxes.json"

        with open(summary_path, "w") as f:
            json.dump(self.summary, f, indent=2)

        with open(boxes_path, "w") as f:
            json.dump(self.boxes_dict, f, indent=2)

        print("\nStage 6 CRAFT results saved:")
        print(f" - Summary: {summary_path}")
        print(f" - Boxes:   {boxes_path}")


# ================= STANDALONE TEST =================


if __name__ == "__main__":
    print("=" * 80)
    print("CRAFT TEXT DETECTION - STAGE 6 STANDALONE TEST")
    print("=" * 80)

    MODE = "folder"  # or "folder"

    MODEL_PATH = (
        r"C:\Users\Viranch Patel\Desktop\Wagon_Inspection_System\models\weights\craft_mlt_25k.pth"
    )
    OUTPUT_DIR = (
        r"C:\Users\Viranch Patel\Desktop\Wagon_Inspection_System\outputs"
    )

    stage6 = Stage6CraftTextDetector(
        output_base_dir=OUTPUT_DIR,
        craft_model_path=MODEL_PATH,
        device="cuda",
        text_threshold=0.8,
        link_threshold=0.25,
        low_text=0.4,
        canvas_size=1600,
        mag_ratio=2.0,
    )

    try:
        if MODE == "single":
            TEST_IMAGE = (
                r"C:\Users\Viranch Patel\Desktop\Wagon_Inspection_System\testing\test_image_4.jpg"
            )
            wagon_name = "wagon_test_id_0"
            result = stage6.process_wagon_image(wagon_name, TEST_IMAGE)
            print(f"\nSingle image processed: {wagon_name}")
            print(f"  Source: {result['source_image_path']}")
            print(f"  Boxes: {result['num_boxes']}")
            print(f"  Crops: {result['num_crops']}")
            print(f"  Annotated: {result['annotated_image_path']}")
        elif MODE == "folder":
            TEST_FOLDER = (
                r"C:\Users\Viranch Patel\Desktop\Wagon_Inspection_System\outputs\stage3_enhanced_frames"
            )
            all_results = stage6.process_folder(TEST_FOLDER)
            print(f"\nFolder processed, wagons: {len(all_results)}")
        else:
            raise ValueError(f"Unknown MODE: {MODE}")

        stage6.save_stage6_results()
    except Exception as e:
        print(f" Error in stage 6 test: {e}")
        sys.exit(1)

    print("=" * 80)
