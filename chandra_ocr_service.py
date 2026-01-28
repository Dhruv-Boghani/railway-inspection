"""
Chandra OCR Service - Runs in OCR environment
"""

import sys
import json
import time
from pathlib import Path
from typing import List

from modules.chandra_ocr_text_extraction import ChandraOCR


# ----------------- Utility: simple progress bar -----------------


def print_progress_bar(
    current: int,
    total: int,
    prefix: str = "",
    length: int = 30,
    start_time: float = None,
):
    """
    Simple text progress bar with optional ETA.
    Example:
        [#####-----]  5/20 (25.0%) | 12.3s elapsed, 36.9s ETA
    """
    if total <= 0:
        total = 1

    frac = current / total
    filled = int(length * frac)
    bar = "#" * filled + "-" * (length - filled)
    percent = frac * 100.0

    # Timing
    timing = ""
    if start_time is not None and current > 0:
        elapsed = time.time() - start_time
        avg_per_item = elapsed / current
        remaining = avg_per_item * (total - current)
        timing = f" | {elapsed:.1f}s elapsed, {remaining:.1f}s ETA"

    line = f"{prefix} [{bar}] {current}/{total} ({percent:.1f}%)" + timing
    # Print on same line
    sys.stdout.write("\r" + line)
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")
        sys.stdout.flush()


# ----------------- Single image mode -----------------


def process_single_image(
    image_path: str,
    output_path: str,
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
):
    """Process one image with detailed logs."""
    print(f"[INFO] Processing SINGLE image: {image_path}")
    image_path = str(image_path)
    output_path = str(output_path)

    # Log model loading with timing
    print("=" * 80)
    print(f"[INFO] Initializing Chandra OCR model: {model_name}")
    model_start = time.time()
    ocr = ChandraOCR(model_name=model_name, device="auto", use_8bit=False)
    model_load_time = time.time() - model_start
    print(f"[INFO] Model initialized in {model_load_time:.1f} seconds")
    print("=" * 80)

    # Run OCR
    print("[INFO] Running OCR on image...")
    t0 = time.time()
    result = ocr.extract_text(
        image=image_path,
        prompt=(
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
        max_new_tokens=256,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"[INFO] OCR completed in {elapsed:.1f} seconds")

    # Save result
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved OCR result to: {out_path}")
    ocr.cleanup()
    return result


# ----------------- Batch mode -----------------


def process_batch(
    input_queue: str,
    output_dir: str,
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
):
    """Process multiple images with model load + progress bar logs."""
    input_queue = str(input_queue)
    output_dir = str(output_dir)

    # Load queue
    with open(input_queue, "r", encoding="utf-8") as f:
        queue_data = json.load(f)

    image_paths: List[str] = [str(p) for p in queue_data.get("images", [])]

    if not image_paths:
        print("[WARN] No images in queue file, exiting.")
        return

    total_images = len(image_paths)
    print("=" * 80)
    print("[INFO] BATCH OCR MODE")
    print(f"[INFO] Total images: {total_images}")
    print("=" * 80)

    # Initialize model once with detailed timing
    print(f"[INFO] Initializing Chandra OCR model: {model_name}")
    model_start = time.time()
    ocr = ChandraOCR(model_name=model_name, device="auto", use_8bit=False)
    model_load_time = time.time() - model_start
    print(f"[INFO] Model initialized in {model_load_time:.1f} seconds")
    print("=" * 80)

    results = []
    batch_start = time.time()

    print("\n[INFO] Running OCR on batch of images...")
    for idx, img_path in enumerate(image_paths, 1):
        img_path = str(img_path)
        print(f"\n[INFO] [{idx}/{total_images}] {img_path}")

        img_start = time.time()
        try:
            result = ocr.extract_text(
                image=img_path,
                prompt=(
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
                max_new_tokens=256,
                verbose=False,
            )
            img_time = time.time() - img_start
            result["image_path"] = img_path
            result["status"] = "success"
            result["time_sec"] = img_time
            results.append(result)
            print(f"[INFO] Done in {img_time:.1f}s")
        except Exception as e:
            img_time = time.time() - img_start
            print(f"[ERROR] After {img_time:.1f}s: {e}")
            results.append(
                {
                    "image_path": img_path,
                    "status": "error",
                    "error": str(e),
                    "text": "",
                    "time_sec": img_time,
                }
            )

        # Update progress bar after each image
        print_progress_bar(
            current=idx,
            total=total_images,
            prefix="Progress",
            length=40,
            start_time=batch_start,
        )

    total_elapsed = time.time() - batch_start
    print(f"\n[INFO] Batch OCR completed in {total_elapsed:.1f} seconds")

    # Save batch results JSON
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    batch_file = output_path / "chandra_ocr_results.json"
    with open(batch_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_images": total_images,
                "successful": sum(
                    1 for r in results if r["status"] == "success"
                ),
                "results": results,
                "timestamp": time.time(),
                "model_load_time_sec": model_load_time,
                "batch_time_sec": total_elapsed,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[INFO] Saved batch OCR results to: {batch_file}")
    ocr.cleanup()


# ----------------- CLI entrypoint -----------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chandra OCR Service")
    parser.add_argument("--mode", choices=["single", "batch"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")

    args = parser.parse_args()

    if args.mode == "single":
        process_single_image(args.input, args.output, args.model)
    elif args.mode == "batch":
        process_batch(args.input, args.output, args.model)
