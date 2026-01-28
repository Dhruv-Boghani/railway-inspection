"""
STAGE 3: Image Enhancement Module (Updated)

- Reads Stage 2 quality summary (new format) for each wagon.
- Uses blur/light labels to decide if enhancement is needed.
- Applies:
    - Low-light enhancement (CLAHE)
    - Deblurring (Simplified MPRNet)
    - Optional denoising
- Saves enhanced frames to outputs/stage3_enhanced_frames.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import json
from tqdm import tqdm
import gc
import time
from datetime import datetime

# ============================================================================
# SimplifiedMPRNet Architecture Components
# ============================================================================

class ChannelAttention(nn.Module):
    def __init__(self, num_feat, reduction=4, bias=False):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // reduction, num_feat, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attention(x)


class CAB(nn.Module):
    def __init__(self, num_feat, kernel_size=3, reduction=4, bias=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, kernel_size, padding=kernel_size // 2, bias=bias),
            nn.PReLU(),
            nn.Conv2d(num_feat, num_feat, kernel_size, padding=kernel_size // 2, bias=bias)
        )
        self.ca = ChannelAttention(num_feat, reduction, bias)

    def forward(self, x):
        res = self.conv(x)
        res = self.ca(res)
        return res + x


class SAM(nn.Module):
    def __init__(self, num_feat, kernel_size=3, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, kernel_size, padding=kernel_size // 2, bias=bias)
        self.conv2 = nn.Conv2d(num_feat, 3, kernel_size, padding=kernel_size // 2, bias=bias)
        self.conv3 = nn.Conv2d(3, num_feat, kernel_size, padding=kernel_size // 2, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img


class SimplifiedEncoder(nn.Module):
    def __init__(self, num_feat, kernel_size, reduction, bias, num_cab):
        super().__init__()
        self.encoder_level1 = nn.Sequential(*[CAB(num_feat, kernel_size, reduction, bias) for _ in range(num_cab)])
        self.down12 = nn.Conv2d(num_feat, num_feat * 2, 4, stride=2, padding=1, bias=bias)
        self.encoder_level2 = nn.Sequential(*[CAB(num_feat * 2, kernel_size, reduction, bias) for _ in range(num_cab)])
        self.down23 = nn.Conv2d(num_feat * 2, num_feat * 4, 4, stride=2, padding=1, bias=bias)
        self.encoder_level3 = nn.Sequential(*[CAB(num_feat * 4, kernel_size, reduction, bias) for _ in range(num_cab)])

    def forward(self, x):
        enc1 = self.encoder_level1(x)
        x = self.down12(enc1)
        enc2 = self.encoder_level2(x)
        x = self.down23(enc2)
        enc3 = self.encoder_level3(x)
        return [enc1, enc2, enc3]


class SimplifiedDecoder(nn.Module):
    def __init__(self, num_feat, kernel_size, reduction, bias, num_cab):
        super().__init__()
        self.decoder_level3 = nn.Sequential(*[CAB(num_feat * 4, kernel_size, reduction, bias) for _ in range(num_cab)])
        self.up32 = nn.ConvTranspose2d(num_feat * 4, num_feat * 2, 2, stride=2, bias=bias)
        self.decoder_level2 = nn.Sequential(*[CAB(num_feat * 2, kernel_size, reduction, bias) for _ in range(num_cab)])
        self.up21 = nn.ConvTranspose2d(num_feat * 2, num_feat, 2, stride=2, bias=bias)
        self.decoder_level1 = nn.Sequential(*[CAB(num_feat, kernel_size, reduction, bias) for _ in range(num_cab)])

    def forward(self, enc_features):
        enc1, enc2, enc3 = enc_features
        x = self.decoder_level3(enc3)
        x = self.up32(x)
        x = x + enc2
        x = self.decoder_level2(x)
        x = self.up21(x)
        x = x + enc1
        x = self.decoder_level1(x)
        return x


class SimplifiedORSNet(nn.Module):
    def __init__(self, kernel_size, bias, scale_orsnetfeats):
        super().__init__()
        self.conv_in = nn.Conv2d(3, scale_orsnetfeats, kernel_size, padding=kernel_size // 2, bias=bias)
        self.conv_mid = nn.Conv2d(scale_orsnetfeats, scale_orsnetfeats, kernel_size, padding=kernel_size // 2, bias=bias)
        self.prelu = nn.PReLU()
        self.conv_out = nn.Conv2d(scale_orsnetfeats, 3, kernel_size, padding=kernel_size // 2, bias=bias)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.prelu(x)
        x = self.conv_mid(x)
        x = self.prelu(x)
        x = self.conv_out(x)
        return x


class SimplifiedMPRNet(nn.Module):
    """
    Simplified MPRNet - Single Scale (NO Multi-Patch Hierarchy)
    Memory efficient for constrained GPUs.
    """

    def __init__(
        self,
        in_c=3,
        out_c=3,
        n_feat=80,
        scale_orsnetfeats=32,
        num_cab=6,
        kernel_size=3,
        reduction=4,
        bias=False,
    ):
        super().__init__()

        self.config = {
            "n_feat": n_feat,
            "scale_orsnetfeats": scale_orsnetfeats,
            "num_cab": num_cab,
            "kernel_size": kernel_size,
            "reduction": reduction,
        }

        # Stage 3
        self.conv_in_stage3 = nn.Conv2d(in_c, n_feat, kernel_size, padding=kernel_size // 2, bias=bias)
        self.encoder_stage3 = SimplifiedEncoder(n_feat, kernel_size, reduction, bias, num_cab)
        self.decoder_stage3 = SimplifiedDecoder(n_feat, kernel_size, reduction, bias, num_cab)
        self.sam_stage3 = SAM(n_feat, kernel_size, bias)

        # Stage 2
        self.conv_in_stage2 = nn.Conv2d(in_c, n_feat, kernel_size, padding=kernel_size // 2, bias=bias)
        self.encoder_stage2 = SimplifiedEncoder(n_feat, kernel_size, reduction, bias, num_cab)
        self.decoder_stage2 = SimplifiedDecoder(n_feat, kernel_size, reduction, bias, num_cab)
        self.sam_stage2 = SAM(n_feat, kernel_size, bias)

        # Stage 1
        self.conv_in_stage1 = nn.Conv2d(in_c, n_feat, kernel_size, padding=kernel_size // 2, bias=bias)
        self.encoder_stage1 = SimplifiedEncoder(n_feat, kernel_size, reduction, bias, num_cab)
        self.decoder_stage1 = SimplifiedDecoder(n_feat, kernel_size, reduction, bias, num_cab)
        self.sam_stage1 = SAM(n_feat, kernel_size, bias)

        # ORSNet
        self.orsnet = SimplifiedORSNet(kernel_size, bias, scale_orsnetfeats)

    def forward(self, x):
        # Stage 3
        x3 = self.conv_in_stage3(x)
        enc3 = self.encoder_stage3(x3)
        dec3 = self.decoder_stage3(enc3)
        sam3, img3 = self.sam_stage3(dec3, x)

        # Stage 2
        x2 = self.conv_in_stage2(img3)
        enc2 = self.encoder_stage2(x2)
        dec2 = self.decoder_stage2(enc2)
        sam2, img2 = self.sam_stage2(dec2, img3)

        # Stage 1
        x1 = self.conv_in_stage1(img2)
        enc1 = self.encoder_stage1(x1)
        dec1 = self.decoder_stage1(enc1)
        sam1, img1 = self.sam_stage1(dec1, img2)

        # ORSNet refinement
        img_refined = img1 + self.orsnet(img1)
        return [img3, img2, img_refined]


# ============================================================================
# ImageEnhancer Class (updated to use new Stage 2 JSON)
# ============================================================================

class ImageEnhancer:
    """
    STAGE 3: Image Enhancement System (updated)

    - Reads Stage 2 quality summary (new format):
        {
          "wagon1_id_1": {
            "wagon_id": "...",
            "image_path": "...",
            "metrics": {...},
            "labels": {
              "blur_label": "blur"/"sharp",
              "light_label": "low_light"/"normal_light",
              "needs_enhancement": bool
            }
          },
          ...
        }
    - Enhances only the best frame per wagon (image_path).
    """

    def __init__(
        self,
        stage2_results_path: str = None,  # Made optional for single-image enhancement
        output_base_dir: str = "outputs",
        mprnet_model_path: str = "models/weights/MPRNET.pth",
        device: str = "cpu",
        # Enhancement parameters
        denoise_strength: int = 5,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: int = 8,
        apply_mprnet: bool = True,
        # MPRNet parameters
        n_feat: int = 80,
        scale_orsnetfeats: int = 32,
        num_cab: int = 6,
        # Performance
        max_image_size: int = 1024,
        verbose: bool = True,
    ):
        self.stage2_results_path = Path(stage2_results_path) if stage2_results_path else None
        self.output_base_dir = Path(output_base_dir)
        self.mprnet_model_path = mprnet_model_path
        self.device = device

        # Enhancement parameters
        self.denoise_strength = denoise_strength
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.apply_mprnet = apply_mprnet

        # MPRNet
        self.n_feat = n_feat
        self.scale_orsnetfeats = scale_orsnetfeats
        self.num_cab = num_cab

        # Performance
        self.max_image_size = max_image_size
        self.verbose = verbose

        self.timing_stats = {"denoise": [], "low_light": [], "deblur": [], "total": []}

        print("\n" + "=" * 80)
        print("🚀 INITIALIZING STAGE 3: IMAGE ENHANCEMENT")
        print("=" * 80)
        print(f"  Device: {self.device}")
        print(f"  Apply MPRNet: {self.apply_mprnet}")
        print(f"  Max image size: {self.max_image_size}px")
        print("=" * 80 + "\n")

        # Load Stage 2 results only if path provided and exists
        self.stage2_results = []
        if self.stage2_results_path and self.stage2_results_path.exists():
            self.load_stage2_results()
        elif self.stage2_results_path:
            self.log(f"Stage 2 results not found at {self.stage2_results_path}, skipping load", "WARNING")

        # Init MPRNet if needed
        if self.apply_mprnet:
            self.init_mprnet()

        # Output dirs
        self.create_output_dirs()

    def log(self, msg: str, level: str = "INFO"):
        if not self.verbose and level not in ["ERROR", "WARNING"]:
            return
        ts = datetime.now().strftime("%H:%M:%S")
        prefix = {
            "INFO": "ℹ️",
            "SUCCESS": "✅",
            "ERROR": "❌",
            "WARNING": "⚠️",
            "PROGRESS": "🔄",
        }.get(level, "")
        print(f"[{ts}] {prefix} {msg}")

    def create_output_dirs(self):
        self.output_dirs: Dict[str, Path] = {
            "enhanced_frames": self.output_base_dir / "stage3_enhanced_frames",
            "comparison": self.output_base_dir / "stage3_comparison",
            "results": self.output_base_dir / "stage3_results",
        }
        for p in self.output_dirs.values():
            p.mkdir(parents=True, exist_ok=True)
        self.log("STAGE 3 output directories created", "SUCCESS")

    # -------- Stage 2 integration (new format) --------

    def load_stage2_results(self):
        if not self.stage2_results_path.exists():
            raise FileNotFoundError(f"Stage 2 summary not found: {self.stage2_results_path}")

        with open(self.stage2_results_path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)

        # Normalize into a list
        self.stage2_results = []
        for wagon_name, info in data.items():
            labels = info.get("labels", {})
            self.stage2_results.append(
                {
                    "wagon_name": wagon_name,
                    "image_path": info.get("image_path", ""),
                    "metrics": info.get("metrics", {}),
                    "blur_label": labels.get("blur_label", "sharp"),
                    "light_label": labels.get("light_label", "normal_light"),
                    "needs_enhancement": labels.get("needs_enhancement", False),
                }
            )

        self.log(f"Loaded Stage 2 results for {len(self.stage2_results)} wagons", "SUCCESS")

    # -------- MPRNet init --------

    def init_mprnet(self):
        self.log("Initializing SimplifiedMPRNet...", "PROGRESS")
        if not Path(self.mprnet_model_path).exists():
            raise FileNotFoundError(f"MPRNet weights not found: {self.mprnet_model_path}")

        self.mprnet_model = SimplifiedMPRNet(
            in_c=3,
            out_c=3,
            n_feat=self.n_feat,
            scale_orsnetfeats=self.scale_orsnetfeats,
            num_cab=self.num_cab,
            kernel_size=3,
            reduction=4,
            bias=False,
        )
        # COMPATIBILITY HACK: Mapping numpy._core to numpy.core
        # This serves to allow loading models saved with numpy 2.0+ on numpy 1.x
        import sys
        import numpy
        if "numpy._core" not in sys.modules:
            sys.modules["numpy._core"] = numpy.core
        if "numpy._core.multiarray" not in sys.modules:
            sys.modules["numpy._core.multiarray"] = numpy.core.multiarray
            
        ckpt = torch.load(self.mprnet_model_path, map_location=self.device)

        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            elif "model" in ckpt:
                state_dict = ckpt["model"]
            elif "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        new_state = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "") if k.startswith("module.") else k
            new_state[name] = v

        self.mprnet_model.load_state_dict(new_state, strict=True)
        self.mprnet_model.to(self.device)
        self.mprnet_model.eval()

        total_params = sum(p.numel() for p in self.mprnet_model.parameters())
        self.log(f"SimplifiedMPRNet params: {total_params/1e6:.2f}M", "SUCCESS")

        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    # -------- Enhancement primitives --------

    def denoise_image(self, img: np.ndarray) -> np.ndarray:
        start = time.time()
        self.log(" → Denoising (bilateral filter)...")
        denoised = cv2.bilateralFilter(
            img,
            d=9,
            sigmaColor=self.denoise_strength * 10,
            sigmaSpace=self.denoise_strength * 10,
        )
        dt = time.time() - start
        self.timing_stats["denoise"].append(dt)
        self.log(f" → Denoise done in {dt:.3f}s")
        return denoised

    def enhance_low_light(self, img: np.ndarray) -> np.ndarray:
        start = time.time()
        self.log(" → Low-light enhancement (CLAHE)...")
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=(self.clahe_tile_size, self.clahe_tile_size),
        )
        l_enh = clahe.apply(l)
        lab_enh = cv2.merge([l_enh, a, b])
        enhanced = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2BGR)
        dt = time.time() - start
        self.timing_stats["low_light"].append(dt)
        self.log(f" → Low-light enhancement done in {dt:.3f}s")
        return enhanced

    def deblur_with_mprnet(self, img: np.ndarray) -> np.ndarray:
        start = time.time()
        self.log(" → Deblurring (SimplifiedMPRNet)...")
        h0, w0 = img.shape[:2]
        original_size = (h0, w0)

        # Downscale if too large
        if max(h0, w0) > self.max_image_size:
            scale = self.max_image_size / max(h0, w0)
            nh, nw = int(h0 * scale), int(w0 * scale)
            self.log(f" → Resizing from {h0}x{w0} to {nh}x{nw}")
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        # BGR → RGB, to tensor
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        t = t.to(self.device)

        h, w = t.shape[2], t.shape[3]
        factor = 8
        H = ((h + factor) // factor) * factor
        W = ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        if padh > 0 or padw > 0:
            t = F.pad(t, (0, padw, 0, padh), mode="reflect")

        with torch.no_grad():
            out_list = self.mprnet_model(t)
            out = torch.clamp(out_list[-1], 0, 1)

        if padh > 0 or padw > 0:
            out = out[:, :, :h, :w]

        out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out = (out * 255.0).astype(np.uint8)
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

        # Resize back
        if max(original_size) > self.max_image_size:
            out_bgr = cv2.resize(
                out_bgr,
                (original_size[1], original_size[0]),
                interpolation=cv2.INTER_CUBIC,
            )

        del t, out_list, out
        if self.device == "cuda":
            torch.cuda.empty_cache()

        dt = time.time() - start
        self.timing_stats["deblur"].append(dt)
        self.log(f" → Deblur done in {dt:.3f}s")
        return out_bgr

    # -------- Policy: decide what to apply from Stage 2 labels --------

    def decide_actions(self, blur_label: str, light_label: str) -> str:
        """
        Build a simple action string based on labels:
          - low_light -> low-light enhancement
          - blur      -> deblur
          - optional denoise if both issues present (you can tweak).
        """
        actions = []

        # Optional: if both blur and low_light, add denoise first
        if blur_label == "blur" and light_label == "low_light":
            actions.append("Denoise")

        if light_label == "low_light":
            actions.append("Low-Light")

        if blur_label == "blur":
            actions.append("Deblur")

        if not actions:
            return "None"

        return " → ".join(actions)

    def enhance_image(
        self,
        img: np.ndarray,
        actions_str: str,
        wagon_name: str,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        log_data: Dict[str, Any] = {
            "wagon_name": wagon_name,
            "original_shape": img.shape,
            "actions_from_stage2": actions_str,
            "applied_steps": [],
            "timings": {},
        }

        if actions_str == "None":
            self.log(" No enhancement needed (good quality).")
            log_data["applied_steps"].append("None")
            log_data["final_shape"] = img.shape
            return img, log_data

        steps = [a.strip() for a in actions_str.split("→")]
        self.log(f" Applying enhancements: {actions_str}")

        cur = img.copy()

        for step in steps:
            if "Denoise" in step or "denoise" in step.lower():
                t0 = time.time()
                cur = self.denoise_image(cur)
                log_data["applied_steps"].append("Denoise")
                log_data["timings"]["denoise"] = time.time() - t0
            elif "Low-Light" in step or "low" in step.lower():
                t0 = time.time()
                cur = self.enhance_low_light(cur)
                log_data["applied_steps"].append("Low-Light")
                log_data["timings"]["low_light"] = time.time() - t0
            elif "Deblur" in step or "blur" in step.lower():
                if self.apply_mprnet:
                    t0 = time.time()
                    cur = self.deblur_with_mprnet(cur)
                    log_data["applied_steps"].append("Deblur (MPRNet)")
                    log_data["timings"]["deblur"] = time.time() - t0
                else:
                    self.log(" MPRNet disabled, skipping deblur", "WARNING")

        if not log_data["applied_steps"]:
            log_data["applied_steps"].append("None")

        log_data["final_shape"] = cur.shape
        return cur, log_data

    # -------- Main loop --------

    def process_all_wagons(self, on_frame_enhanced: Optional[callable] = None) -> Dict[str, Any]:
        self.log(f"Processing {len(self.stage2_results)} wagons...", "INFO")

        results: Dict[str, Any] = {}
        success = 0
        failed = 0
        
        # Counter for callback
        count = 0

        for idx, w in enumerate(
            tqdm(self.stage2_results, desc="Enhancing wagons", unit="wagon", ncols=90)
        ):
            wagon_name = w.get("wagon_name", f"wagon_{idx}")
            img_path = w.get("image_path", "")
            blur_label = w.get("blur_label", "sharp")
            light_label = w.get("light_label", "normal_light")
            needs_enh = w.get("needs_enhancement", False)

            self.log(
                f"[{idx+1}/{len(self.stage2_results)}] {wagon_name} - "
                f"blur={blur_label}, light={light_label}, needs={needs_enh}",
                "INFO",
            )

            # Decide actions from labels
            actions_str = self.decide_actions(blur_label, light_label)

            # If Stage 2 says no enhancement needed, we can still just copy original
            if not img_path:
                self.log(f"No image path for {wagon_name}, skipping", "ERROR")
                failed += 1
                continue

            img_path = Path(img_path)
            if not img_path.exists():
                self.log(f"Image not found: {img_path}", "ERROR")
                failed += 1
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                self.log(f"cv2 could not read image: {img_path}", "ERROR")
                failed += 1
                continue

            start = time.time()

            try:
                enhanced, log_data = self.enhance_image(img, actions_str, wagon_name)

                # Save enhanced image (or original if no enhancement)
                out_name = f"{wagon_name}_enhanced.jpg"
                out_path = self.output_dirs["enhanced_frames"] / out_name
                cv2.imwrite(str(out_path), enhanced)

                # Optional comparison
                comp = np.hstack([img, enhanced])
                comp_path = self.output_dirs["comparison"] / f"{wagon_name}_comparison.jpg"
                cv2.imwrite(str(comp_path), comp)

                total_t = time.time() - start
                self.timing_stats["total"].append(total_t)

                results[wagon_name] = {
                    **log_data,
                    "image_path_stage2": str(img_path),
                    "enhanced_frame_path": str(out_path),
                    "comparison_path": str(comp_path),
                    "success": True,
                    "processing_time": total_t,
                    "labels": {
                        "blur_label": blur_label,
                        "light_label": light_label,
                        "needs_enhancement": needs_enh,
                    },
                }

                self.log(
                    f" Done {wagon_name} in {total_t:.2f}s, steps: "
                    f"{' → '.join(log_data['applied_steps'])}",
                    "SUCCESS",
                )
                success += 1
                
                # Callback
                count += 1
                if on_frame_enhanced:
                    try:
                        on_frame_enhanced(count)
                    except Exception as e:
                        pass

            except Exception as e:
                self.log(f"Error enhancing {wagon_name}: {e}", "ERROR")
                results[wagon_name] = {
                    "wagon_name": wagon_name,
                    "success": False,
                    "error": str(e),
                }
                failed += 1

            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

        # Save results JSON
        res_path = self.output_dirs["results"] / "stage3_enhancement_results.json"
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        self.print_final_stats(success, failed, res_path)
        return results

    def print_final_stats(self, success: int, failed: int, res_path: Path):
        print("\n" + "=" * 80)
        print("✅ STAGE 3 COMPLETE")
        print("=" * 80)
        total = success + failed
        print(f" Total wagons: {total}")
        print(f" Successful: {success}")
        print(f" Failed: {failed}")
        if total > 0:
            print(f" Success rate: {success/total*100:.1f}%")
        if self.timing_stats["total"]:
            print(f" Avg time/wagon: {np.mean(self.timing_stats['total']):.2f}s")
            print(f" Total time: {sum(self.timing_stats['total']):.2f}s")
        print("\nOutputs:")
        print(f" Enhanced frames: {self.output_dirs['enhanced_frames']}")
        print(f" Comparisons: {self.output_dirs['comparison']}")
        print(f" Results JSON: {res_path}")
        print("=" * 80 + "\n")


# ============================================================================
# Standalone test
# ============================================================================

if __name__ == "__main__":
    enhancer = ImageEnhancer(
        stage2_results_path=r"C:\Users\Viranch Patel\Desktop\Wagon_Inspection_System\outputs\stage2_quality_results\stage2_quality_summary.json",
        output_base_dir=r"C:\Users\Viranch Patel\Desktop\Wagon_Inspection_System\outputs",
        mprnet_model_path=r"C:\Users\Viranch Patel\Desktop\Wagon_Inspection_System\models\weights\MPRNET.pth",
        device="cuda" if torch.cuda.is_available() else "cpu",
        denoise_strength=5,
        clahe_clip_limit=2.0,
        clahe_tile_size=8,
        apply_mprnet=True,
        n_feat=80,
        scale_orsnetfeats=32,
        num_cab=6,
        max_image_size=1024,
        verbose=True,
    )
    enhancer.process_all_wagons()
