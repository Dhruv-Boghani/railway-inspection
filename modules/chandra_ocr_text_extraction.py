"""
Chandra OCR Text Extraction Module

Multilingual OCR using Chandra (Qwen3-VL based) with quantization.

This version is wired to Stage 6 CRAFT crops:
- Reads crop images from outputs/stage6_craft_crops
- Runs OCR on each crop
- Saves:
    - outputs/stage6_ocr_results/chandra_stage6_crops.json   (per-crop results)
    - outputs/stage6_ocr_results/chandra_stage6_summary.json (per-wagon summary)
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
import time
import gc
from datetime import datetime
import json


class ChandraOCR:
    """
    Chandra OCR Engine (Quantized)

    Optimized for Indian Railway wagon text extraction:
    - Supports English + Hindi/Devanagari
    - 4/8-bit quantization for memory efficiency
    - Handles low-quality and blurry images
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "auto",
        use_8bit: bool = True,
        trust_remote_code: bool = True
    ):
        """
        Initialize Chandra OCR

        Args:
            model_name: Hugging Face model name
            device: "cpu", "cuda", or "auto"
            use_8bit: Use 8-bit quantization (reduces memory 30GB -> 8GB)
            trust_remote_code: Trust remote code for custom models
        """
        self.model_name = model_name
        self.use_8bit = use_8bit
        self.trust_remote_code = trust_remote_code

        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print("=" * 80)
        print("[INFO] INITIALIZING CHANDRA OCR")
        print("=" * 80)

        # Check GPU
        self.check_gpu()

        # Load model and processor
        self.processor = None
        self.model = None
        self.load_model()

    def check_gpu(self):
        """Check GPU availability and memory"""
        if self.device == "cuda":
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"[INFO] GPU: {gpu_name}")
                print(f"[INFO] Total VRAM: {total_vram:.1f} GB")
                print(f"[INFO] CUDA: {torch.version.cuda}")
                if total_vram < 8 and self.use_8bit:
                    print(f"[WARN] Only {total_vram:.1f}GB VRAM")
                    print("       8-bit quantization typically requires ~8GB")
                elif not self.use_8bit and total_vram < 30:
                    print(f"[WARN] Full precision needs ~30GB")
                    print("       Enabling 8-bit quantization...")
                    self.use_8bit = True
            else:
                print("[ERROR] CUDA not available, falling back to CPU")
                self.device = "cpu"
        else:
            print("[INFO] Using CPU (slow but works)")
        print("=" * 80)

    def load_model(self):
        """Load Chandra OCR model with safe optional 4-bit quantization"""

        print("\n[INFO] Loading Chandra OCR model...")
        print(f"[INFO] Model: {self.model_name}")
        print("[INFO] GPU inference preferred, CPU fallback enabled")

        start_time = time.time()

        # ----------------------------
        # Cleanup memory
        # ----------------------------
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # ----------------------------
        # Load processor
        # ----------------------------
        print("\n[INFO] [1/2] Loading processor...")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        print("[INFO] Processor loaded")

        # ----------------------------
        # Prepare quantization (SAFE)
        # ----------------------------
        quantization_config = None
        use_cuda = torch.cuda.is_available() and self.device == "cuda"

        if use_cuda:
            try:
                from transformers import BitsAndBytesConfig
                import bitsandbytes  # noqa: F401
                print("[INFO] bitsandbytes available — enabling 4-bit NF4 quantization")

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )

            except ImportError:
                print("[WARN] bitsandbytes not available — running WITHOUT quantization")
                quantization_config = None

        # ----------------------------
        # Load model
        # ----------------------------
        print("\n[INFO] [2/2] Loading model...")

        if use_cuda:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
            )
        else:
            print("[WARN] CUDA not available — using CPU mode")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                device_map={"": "cpu"},
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )

        # ----------------------------
        # Finalize
        # ----------------------------
        self.model.eval()

        elapsed = time.time() - start_time

        print("\n" + "=" * 80)
        print(f"[INFO] MODEL LOADED in {elapsed:.1f}s ({elapsed/60:.1f} min)")

        if use_cuda:
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"[INFO] GPU Memory: {allocated:.2f}GB allocated / {reserved:.2f}GB reserved")
            print("[INFO] NOTE: RTX 2050 (4GB) may still OOM on large images")

        print("=" * 80 + "\n")

    def _filter_test_patterns(self, text: str) -> str:
        """
        Filter out obvious test patterns and invalid wagon numbers.
        Returns "Empty" for common test sequences.
        """
        text = text.strip()
        test_patterns = [
            "1234567890", "0123456789", "123456789", "012345678",
            "11111111111", "00000000000", "99999999999"
        ]

        if text in test_patterns:
            return "Empty"

        if len(text) >= 7:
            first_char = text[0]
            if all(c == first_char for c in text):
                return "Empty"

        if len(text) >= 11 and text.startswith("0" * 5):
            return "Empty"

        return text

    def extract_text(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        prompt: str = (
            "Extract all text from this image. Include all visible numbers, "
            "letters, symbols, and words. Be thorough and accurate."
        ),
        max_new_tokens: int = 2048,
        verbose: bool = True
    ) -> Dict:
        """Extract text from image using Chandra OCR (Qwen2-VL)"""

        # Load image
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
            image_name = Path(image).name
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image_name = "array_image"
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
            image_name = "pil_image"
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        w, h = pil_image.size
        if verbose:
            print(f"[INFO] Processing: {image_name} ({w}x{h}px)")

        # Qwen2-VL chat format
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt}
            ]
        }]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        if verbose:
            print("[INFO] Running OCR...")
        start_time = time.time()

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None
            )

        elapsed = time.time() - start_time

        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0].strip()

        # Extract assistant part and clean
        if "assistant\n" in output_text:
            output_text = output_text.split("assistant\n")[1].strip()
        elif "<|im_start|>assistant\n" in output_text:
            output_text = output_text.split("assistant\n")[1].strip()
        else:
            output_text = output_text[150:].strip()

        output_text = self._filter_test_patterns(output_text)
        output_text = output_text.replace("\n", "").replace(" ", "").strip()

        if verbose:
            print(f"[INFO] Completed: {elapsed:.2f}s")
            print(f"[INFO] Wagon numbers text: '{output_text}'")

        # Cleanup
        del inputs, generated_ids
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return {
            "text": output_text,
            "image_name": image_name,
            "dimensions": f"{w}x{h}",
            "time_sec": round(elapsed, 2),
            "chars": len(output_text),
            "words": len(output_text.split()),
            "timestamp": datetime.now().isoformat()
        }

    def batch_extract(
        self,
        images: List[Union[str, Path, np.ndarray, Image.Image]],
        prompt: str = (
            "Extract all text from this image. Include all visible numbers, "
            "letters, symbols, and words. Be thorough and accurate."
        ),
        verbose: bool = True
    ) -> List[Dict]:
        """
        Extract text from multiple images.
        """
        results: List[Dict] = []
        print("\n" + "=" * 80)
        print(f"[INFO] PROCESSING {len(images)} IMAGES")
        print("=" * 80 + "\n")

        for idx, image in enumerate(images, 1):
            if verbose:
                print(f"[{idx}/{len(images)}]")
            try:
                result = self.extract_text(image, prompt, verbose=verbose)
                result["status"] = "success"
                results.append(result)
            except Exception as e:
                print(f"[ERROR] {e}")
                results.append({
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__
                })

            if verbose and idx < len(images):
                print()

        # Summary
        success = sum(1 for r in results if r.get("status") == "success")
        print("\n" + "=" * 80)
        print("[INFO] BATCH SUMMARY")
        print("=" * 80)
        print(f"[INFO] Success: {success}/{len(images)} ({success/len(images)*100:.0f}%)")
        if success > 0:
            avg_time = sum(
                r["time_sec"] for r in results if r.get("status") == "success"
            ) / success
            total_chars = sum(
                r["chars"] for r in results if r.get("status") == "success"
            )
            total_words = sum(
                r["words"] for r in results if r.get("status") == "success"
            )
            print(f"[INFO] Avg time: {avg_time:.2f}s/image")
            print(f"[INFO] Total: {total_chars:,} chars, {total_words:,} words")
        print("=" * 80 + "\n")

        return results

    # ----------------- NEW: Stage 6 crop handling -----------------

    @staticmethod
    def _parse_crop_filename(filename: str) -> Dict[str, Any]:
        """
        Parse wagon_number, track_id, crop_idx from crop filename.

        Expected CRAFT Stage 6 pattern: wagon{n}_id_{tid}_crop_XX.jpg
        Fallback: <wagon_name>_crop_XX.jpg
        """
        name = Path(filename).stem
        parts = name.split("_crop_")
        base = parts[0]
        crop_idx = None
        if len(parts) > 1:
            try:
                crop_idx = int(parts[1])
            except ValueError:
                crop_idx = None

        info: Dict[str, Any] = {"raw_name": name, "crop_idx": crop_idx}

        try:
            if base.startswith("wagon") and "_id_" in base:
                prefix, tid = base.split("_id_")
                num_str = prefix.replace("wagon", "")
                info["wagon_number"] = int(num_str)
                info["track_id"] = int(tid)
            else:
                info["wagon_name"] = base
        except Exception:
            info["wagon_name"] = base

        return info

    def run_on_stage6_crops(
        self,
        outputs_root: Union[str, Path] = "outputs",
        crops_subdir: str = "stage6_craft_crops",
        ocr_results_subdir: str = "stage6_ocr_results",
        prompt: str = (
            "Recognise only the 11 numbers/digits from the image.\n"
            "NOTE:\n"
            "- If you see any kind of English or hindi text in the image then strictly return \"Empty\".\n"
            "- If you see a number then make sure that \"|\" is not to be ignored and is to be counted as digit \"1\".\n"
            "- If there are missing numbers and not 11 digit number then try to recognise and make it 11 digit number.\n"
            "- Only include digits 0–9.\n"
            "- Return output only for images that have digits.\n"
            "- Do not convert letters to digits.\n"
            "- Return exactly an 11 digit number if present, otherwise \"Empty\"."
        ),
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run OCR on all Stage 6 CRAFT crops and save structured results.

        - Reads crops from: <outputs_root>/stage6_craft_crops
        - Saves per-crop JSON:   <outputs_root>/stage6_ocr_results/chandra_stage6_crops.json
        - Saves per-wagon JSON:  <outputs_root>/stage6_ocr_results/chandra_stage6_summary.json
        """
        outputs_root = Path(outputs_root)
        crops_dir = outputs_root / crops_subdir
        ocr_dir = outputs_root / ocr_results_subdir
        ocr_dir.mkdir(parents=True, exist_ok=True)

        per_crop_json = ocr_dir / "chandra_stage6_crops.json"
        summary_json = ocr_dir / "chandra_stage6_summary.json"

        if not crops_dir.exists():
            raise FileNotFoundError(
                f"Stage 6 CRAFT crops directory not found: {crops_dir}. "
                f"Run Stage6CraftTextDetector first."
            )

        # Collect crop images
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        crop_paths = sorted(
            p for p in crops_dir.iterdir() if p.suffix.lower() in valid_exts
        )

        if not crop_paths:
            if verbose:
                print(f"[WARN] No crop images found in {crops_dir}")
            return {
                "status": "no_crops",
                "num_crops": 0,
                "crops_json": str(per_crop_json),
                "summary_json": str(summary_json),
            }

        if verbose:
            print("=" * 80)
            print("CHANDRA OCR ON STAGE 6 CRAFT CROPS")
            print("=" * 80)
            print(f"[INFO] Crops dir: {crops_dir}")
            print(f"[INFO] Crops found: {len(crop_paths)}\n")

        # Run batch OCR
        results = self.batch_extract(
            images=[str(p) for p in crop_paths],
            prompt=prompt,
            verbose=verbose,
        )

        # Build per-crop and per-wagon structures
        per_crop: Dict[str, Any] = {}
        per_wagon: Dict[str, Any] = {}

        for img_path, ocr_res in zip(crop_paths, results):
            fname = img_path.name
            info = self._parse_crop_filename(fname)
            if "wagon_number" in info and "track_id" in info:
                wagon_key = f"wagon{info['wagon_number']}_id_{info['track_id']}"
            else:
                wagon_key = info.get("wagon_name", info["raw_name"])

            entry = {
                "image_path": str(img_path),
                "filename": fname,
                "wagon_info": info,
                "ocr_text": ocr_res.get("text", ""),
                "time_sec": ocr_res.get("time_sec", 0.0),
                "status": ocr_res.get("status", "success"),
            }
            per_crop[fname] = entry

            if wagon_key not in per_wagon:
                per_wagon[wagon_key] = {
                    "wagon_key": wagon_key,
                    "wagon_info": info,
                    "crops": [],
                    "texts": [],
                }

            per_wagon[wagon_key]["crops"].append(entry)
            if entry["ocr_text"]:
                per_wagon[wagon_key]["texts"].append(entry["ocr_text"])

        # Save JSONs
        with open(per_crop_json, "w", encoding="utf-8") as f:
            json.dump(per_crop, f, indent=2, ensure_ascii=False)

        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(per_wagon, f, indent=2, ensure_ascii=False)

        if verbose:
            print("\n[INFO] Chandra OCR Stage 6 results saved:")
            print(f" - Per-crop:   {per_crop_json}")
            print(f" - Per-wagon:  {summary_json}")

        return {
            "status": "success",
            "num_crops": len(crop_paths),
            "num_wagons": len(per_wagon),
            "crops_json": str(per_crop_json),
            "summary_json": str(summary_json),
        }

    def cleanup(self):
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    base_outputs = r"C:\Users\Viranch Patel\Desktop\Wagon_Inspection_System\outputs"
    ocr = ChandraOCR(
        model_name="Qwen/Qwen2-VL-7B-Instruct",
        device="cuda",
        use_8bit=False,
    )
    result = ocr.run_on_stage6_crops(
        outputs_root=base_outputs,
        crops_subdir="stage6_craft_crops",
        ocr_results_subdir="stage6_ocr_results",
        verbose=True,
    )
    print("\nFINAL:", result)
    ocr.cleanup()
