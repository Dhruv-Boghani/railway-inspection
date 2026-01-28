"""
STAGE 6: Wagon Number Extraction Module (CRAFT + Chandra HTTP OCR + IR decoder)

Pipeline:

1) Take Stage 3 enhanced frames from: outputs/stage3_enhanced_frames
2) Run Stage6CraftTextDetector to:
   - Detect text regions per wagon
   - Save crops into outputs/stage6_craft_crops
   - Save metadata JSON in outputs/stage6_craft_results
3) Use Chandra OCR HTTP server (running in chandra_ocr_venv) to OCR all Stage-6 crops.
4) For each wagon:
   - Aggregate all OCR texts
   - Extract all 11-digit wagon number candidates
   - Validate check digit as per IR 11-digit scheme
   - Decode type, owning railway, year of manufacture, serial, check digit status
5) Save:
   - outputs/stage6_wagon_number_results/stage6_wagon_numbers.json
   - outputs/stage6_wagon_number_results/stage6_wagon_numbers.csv
   - outputs/stage6_wagon_number_results/stage6_wagon_numbers_detailed.json
   - outputs/stage6_wagon_number_results/stage6_ocr_crops.json
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re
import json

import cv2
import numpy as np
import pandas as pd
import requests

from craft_text_detection import Stage6CraftTextDetector
# from chandra_ocr_text_extraction import ChandraOCR  # if you use it



# =====================================================
# DATA STRUCTURES
# =====================================================

@dataclass
class OCRCropResult:
    wagon_key: str
    crop_path: str
    filename: str
    text: str
    time_sec: float
    error: Optional[str]


@dataclass
class WagonNumberDetails:
    full_number: str                    # 11-digit string
    is_valid_check_digit: bool          # check digit matches?
    expected_check_digit: int           # from first 10 digits
    actual_check_digit: int             # from 11th digit
    wagon_type_code: str                # C1-C2
    owning_railway_code: str            # C3-C4
    year_of_manufacture: str            # C5-C6 as "20YY" or "19YY" heuristic
    raw_year_digits: str                # original two digits
    individual_serial: str              # C7-C10
    parsed_ok: bool                     # True if parsing succeeded


@dataclass
class WagonNumberResult:
    wagon_name: str                     # e.g. "wagon2_id_5"
    detected_number: Optional[str]      # chosen 11-digit number or None
    candidate_numbers: List[str]        # all unique 11-digit candidates (sorted)
    success: bool
    reason: str
    crop_texts: List[str]               # raw texts from all crops
    candidates_details: Dict[str, WagonNumberDetails]


# =====================================================
# HELPERS: TEXT FILTERS & VALIDATION
# =====================================================

def is_repetitive(text: str) -> bool:
    """True if all characters in text are the same (e.g., '11111', '9999')."""
    if not text:
        return False
    return len(set(text)) == 1

def is_sequential(text: str) -> bool:
    """True if characters form a sequence (e.g., '123456', '987654')."""
    if len(text) < 3:
        return False
    
    # Check ascending keys
    ascending = "0123456789"
    if text in ascending:
        return True
        
    # Check descending
    descending = "9876543210"
    if text in descending:
        return True
        
    return False

def clean_ocr_text(raw: str) -> str:
    """Clean Chandra OCR output to prepare for digit extraction."""
    if not raw:
        return ""
    t = raw.upper()
    # Replace common look-alikes if any slipped through OCR prompt
    t = t.replace("|", "1")
    # Keep only digits and letters
    t = re.sub(r"[^A-Z0-9]", "", t)
    return t


def extract_11digit_candidates(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"\d{11}", text)


def parse_wagon_key_from_crop_name(filename: str) -> str:
    """
    Match the naming used by Stage6CraftTextDetector._crop_filename:
    - wagon{n}_id_{tid}_crop_XX.jpg -> wagon{n}_id_{tid}
    - fallback: <stem>_crop_XX.jpg -> <stem>
    """
    stem = Path(filename).stem
    if "_crop_" in stem:
        base = stem.split("_crop_")[0]
    else:
        base = stem
    return base


# =====================================================
# HELPERS: INDIAN 11-DIGIT WAGON NUMBER DECODER
# =====================================================

def compute_ir_wagon_check_digit(first10: str) -> int:
    """
    Compute the Indian Railways wagon check digit for the first 10 digits.

    Algorithm (C1..C10) [Type, Owning, Year, Serial]: [web:51][web:60][web:65]
      1) S1 = sum of digits at even positions (C2, C4, C6, C8, C10)
      2) S1 *= 3
      3) S2 = sum of digits at odd positions (C1, C3, C5, C7, C9)
      4) S3 = S1 + S2
      5) Next multiple of 10 above S3 = M
      6) Check digit = M - S3
    """
    if len(first10) != 10 or not first10.isdigit():
        raise ValueError("first10 must be a 10-digit numeric string")

    digits = [int(c) for c in first10]

    # positions are 1-based in spec; Python list is 0-based
    even_pos_indices = [1, 3, 5, 7, 9]   # 2,4,6,8,10
    odd_pos_indices = [0, 2, 4, 6, 8]    # 1,3,5,7,9

    s1 = sum(digits[i] for i in even_pos_indices)
    s1 *= 3
    s2 = sum(digits[i] for i in odd_pos_indices)
    s3 = s1 + s2

    # next multiple of 10
    next_mult_10 = ((s3 + 9) // 10) * 10
    check_digit = next_mult_10 - s3
    return check_digit


def decode_wagon_number(number: str) -> WagonNumberDetails:
    """
    Decode an 11-digit Indian Railway wagon number into its components. [web:51][web:60][web:65]

    Positions:
      C1-C2 : wagon_type_code
      C3-C4 : owning_railway_code
      C5-C6 : year_of_manufacture (last 2 digits)
      C7-C10: individual_serial
      C11   : check digit
    """
    # Basic numeric/length validation
    if len(number) != 11 or not number.isdigit():
        return WagonNumberDetails(
            full_number=number,
            is_valid_check_digit=False,
            expected_check_digit=-1,
            actual_check_digit=-1,
            wagon_type_code=number[0:2] if len(number) >= 2 else "",
            owning_railway_code=number[2:4] if len(number) >= 4 else "",
            year_of_manufacture="",
            raw_year_digits=number[4:6] if len(number) >= 6 else "",
            individual_serial=number[6:10] if len(number) >= 10 else "",
            parsed_ok=False,
        )

    c1c2 = number[0:2]
    c3c4 = number[2:4]
    c5c6 = number[4:6]
    c7c10 = number[6:10]
    c11 = number[10]

    # Compute expected check digit
    expected = compute_ir_wagon_check_digit(number[:10])
    actual = int(c11)
    is_valid = (expected == actual)

    # Year heuristic: 00-79 => 2000-2079, 80-99 => 1980-1999 (can be tuned) [web:60][web:69]
    raw_year = c5c6
    yy = int(raw_year)
    if yy <= 79:
        year_full = f"20{raw_year:0>2}"
    else:
        year_full = f"19{raw_year:0>2}"

    return WagonNumberDetails(
        full_number=number,
        is_valid_check_digit=is_valid,
        expected_check_digit=expected,
        actual_check_digit=actual,
        wagon_type_code=c1c2,
        owning_railway_code=c3c4,
        year_of_manufacture=year_full,
        raw_year_digits=raw_year,
        individual_serial=c7c10,
        parsed_ok=True,
    )


# =====================================================
# MAIN EXTRACTOR
# =====================================================

class WagonNumberExtractor:
    """
    Full Stage 6: CRAFT + Chandra (HTTP server) + IR wagon-number decoding.
    """

    def __init__(
        self,
        outputs_root: str = "outputs",
        stage3_dirname: str = "stage3_enhanced_frames",
        craft_model_path: Optional[str] = None,
        # HTTP OCR server config
        ocr_server_url: str = "http://127.0.0.1:8001/ocr",
        min_ocr_box_score: float = 0.0,  # reserved for future scoring
    ):
        self.outputs_root = Path(outputs_root)
        self.stage3_dir = self.outputs_root / stage3_dirname

        # Stage 6 CRAFT directories (must match craft_text_detection.py)
        self.stage6_crops_dir = self.outputs_root / "stage6_craft_crops"
        self.stage6_results_dir = self.outputs_root / "stage6_craft_results"

        # Final wagon-number results
        self.results_dir = self.outputs_root / "stage6_wagon_number_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Init CRAFT stage-6 wrapper
        if craft_model_path is None:
            craft_model_path = "models/weights/craft_mlt_25k.pth"

        self.craft_stage6 = Stage6CraftTextDetector(
            output_base_dir=str(self.outputs_root),
            craft_model_path=craft_model_path,
            device="cuda" if cv2.ocl.haveOpenCL() else "cpu",
            text_threshold=0.8,
            link_threshold=0.25,
            low_text=0.4,
            canvas_size=1600,
            mag_ratio=2.0,
        )

        # HTTP OCR server endpoint
        self.ocr_server_url = ocr_server_url
        self.min_ocr_box_score = min_ocr_box_score

        print("[INFO] WagonNumberExtractor initialized")
        print(f" - Stage3 frames: {self.stage3_dir}")
        print(f" - Stage6 crops:  {self.stage6_crops_dir}")
        print(f" - Results dir:   {self.results_dir}")
        print(f" - OCR server:    {self.ocr_server_url}")

    # -------------------------------------------------

    def run_stage6_craft(self) -> Dict[str, Any]:
        """
        Run Stage 6 CRAFT on all Stage 3 enhanced frames.
        """
        if not self.stage3_dir.exists():
            raise FileNotFoundError(
                f"Stage 3 enhanced frames dir not found: {self.stage3_dir}"
            )

        print("\n=== STAGE 6: CRAFT ON ENHANCED FRAMES ===")
        results = self.craft_stage6.process_folder(str(self.stage3_dir))
        self.craft_stage6.save_stage6_results()
        print(f"[INFO] Stage 6 CRAFT done for {len(results)} wagons.")
        return results

    # -------------------------------------------------

    def _call_chandra_http(self, image_path: str) -> Dict[str, Any]:
        """
        Call the Chandra OCR HTTP server for a single image.

        The OCR server must be running in chandra_ocr_venv on self.ocr_server_url.
        """
        prompt = (
            "Recognise only the 11 numbers/digits from the image.\n"
            "NOTE:\n"
            "- If you see any kind of English or hindi text in the image then strictly return \"Empty\".\n"
            "- If you see a number then make sure that \"|\" is not to be ignored and is to be counted as digit \"1\".\n"
            "- If there are missing numbers and not 11 digit number then try to recognise and make it 11 digit number.\n"
            "- Only include digits 0–9.\n"
            "- Return output only for images that have digits.\n"
            "- Do not convert letters to digits.\n"
            "- Return exactly an 11 digit number if present, otherwise \"Empty\"."
        )
        
        payload = {
            "image_path": image_path,
            "prompt": prompt,
            "max_new_tokens": 512,
        }
        resp = requests.post(self.ocr_server_url, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()

    def run_chandra_on_crops(self, on_ocr_processed: Optional[callable] = None) -> List[OCRCropResult]:
        """
        Use Chandra OCR HTTP server to OCR all Stage 6 crops.
        Returns list of OCRCropResult.
        Also saves a per-crop JSON (similar to chandra_stage6_crops.json).
        """
        if not self.stage6_crops_dir.exists():
            raise FileNotFoundError(
                f"Stage 6 crops dir not found: {self.stage6_crops_dir}. "
                f"Run run_stage6_craft() first."
            )

        crop_paths = sorted(
            p
            for p in self.stage6_crops_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        )
        if not crop_paths:
            print("[WARN] No crops found for Chandra OCR.")
            return []

        print(f"\n=== STAGE 6: CHANDRA OCR (HTTP server) ON {len(crop_paths)} CROPS ===")

        crop_results: List[OCRCropResult] = []
        per_crop: Dict[str, Any] = {}
        
        # Counter for callback
        count = 0

        for idx, img_path in enumerate(crop_paths, 1):
            print(f"[{idx}/{len(crop_paths)}] {img_path}")
            try:
                res = self._call_chandra_http(str(img_path))
                text = res.get("text", "") or ""
                t_sec = res.get("time_sec", 0.0)
                err = res.get("error")
            except Exception as e:
                text = ""
                t_sec = 0.0
                err = str(e)

            fname = img_path.name
            wagon_key = parse_wagon_key_from_crop_name(fname)

            crop_result = OCRCropResult(
                wagon_key=wagon_key,
                crop_path=str(img_path),
                filename=fname,
                text=text,
                time_sec=float(t_sec),
                error=err,
            )
            crop_results.append(crop_result)

            per_crop[fname] = {
                "wagon_key": wagon_key,
                "image_path": str(img_path),
                "filename": fname,
                "ocr_text": text,
                "time_sec": float(t_sec),
                "error": err,
            }
            
            # Callback
            count += 1
            if on_ocr_processed:
                try:
                    on_ocr_processed(count)
                except Exception as e:
                    pass

        # Save per-crop OCR JSON for debugging / auditing
        per_crop_path = self.results_dir / "stage6_ocr_crops.json"
        with open(per_crop_path, "w", encoding="utf-8") as f:
            json.dump(per_crop, f, indent=2, ensure_ascii=False)

        print(f"[INFO] Per-crop OCR saved: {per_crop_path}")

        return crop_results

    # -------------------------------------------------

    def _build_wagon_texts(
        self, crop_results: List[OCRCropResult]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Organize OCR crops per wagon, in ascending wagon order when possible.
        Similar to ChandraOCR.run_on_stage6_crops summary JSON. [file:63]
        """
        per_wagon: Dict[str, Dict[str, Any]] = {}

        for cr in crop_results:
            wagon_key = cr.wagon_key
            if wagon_key not in per_wagon:
                per_wagon[wagon_key] = {
                    "wagon_key": wagon_key,
                    "crops": [],
                    "texts": [],
                }
            entry = {
                "image_path": cr.crop_path,
                "filename": cr.filename,
                "ocr_text": cr.text,
                "time_sec": cr.time_sec,
                "error": cr.error,
            }
            per_wagon[wagon_key]["crops"].append(entry)
            if cr.text:
                per_wagon[wagon_key]["texts"].append(cr.text)

        # Sort wagons in ascending order by numeric part if present (wagonX_id_Y)
        def wagon_sort_key(k: str) -> Tuple[int, int, str]:
            # default if parse fails
            num = 9999
            tid = 9999
            base = k
            try:
                if k.startswith("wagon") and "_id_" in k:
                    prefix, tid_str = k.split("_id_")
                    num_str = prefix.replace("wagon", "")
                    num = int(num_str)
                    tid = int(tid_str)
            except Exception:
                pass
            return (num, tid, base)

        sorted_keys = sorted(per_wagon.keys(), key=wagon_sort_key)
        ordered = {k: per_wagon[k] for k in sorted_keys}
        return ordered

    def _select_numbers_per_wagon(
        self,
        crop_results: List[OCRCropResult],
    ) -> Dict[str, WagonNumberResult]:
        """
        Aggregate crop OCR outputs per wagon and select final 11-digit number.
        Logic Update:
          1. Filter out Repetitive ("11111") and Sequential ("12345") noise.
          2. Check individual valid 11-digit numbers.
          3. MERGE partials: If no perfect single match, try combining snippets.
        """
        per_wagon = self._build_wagon_texts(crop_results)

        final: Dict[str, WagonNumberResult] = {}

        for wagon_key, wagon_data in per_wagon.items():
            all_texts = wagon_data.get("texts", [])
            
            # 1. Clean and Filter Candidates
            valid_candidates = []
            
            for text in all_texts:
                cleaned = clean_ocr_text(text)
                
                # Basic length/digit filter
                if not cleaned or not cleaned.isdigit():
                    continue

                # REPETITIVE / SEQUENTIAL FILTER
                if is_repetitive(cleaned) or is_sequential(cleaned):
                     continue # Treat as "Empty" / Noise
                
                valid_candidates.append(cleaned)

            # Deduplicate while preserving order
            valid_candidates = list(dict.fromkeys(valid_candidates))

            detected_number = None
            candidate_details_map: Dict[str, WagonNumberDetails] = {}

            # 2. Direct Check: Look for perfect 11-digit matches in single crops
            for cand in valid_candidates:
                if len(cand) == 11:
                    details = decode_wagon_number(cand)
                    candidate_details_map[cand] = details
                    if details.is_valid_check_digit and detected_number is None:
                        detected_number = cand
            
            # 3. Merging Logic: If no direct match, try combining partials
            if detected_number is None and len(valid_candidates) > 1:
                # Try every pair (A + B)
                import itertools
                for a, b in itertools.permutations(valid_candidates, 2):
                    merged = a + b
                    # Must be exactly 11 digits
                    if len(merged) == 11:
                         # Validate
                        details = decode_wagon_number(merged)
                        # Store as a candidate
                        if merged not in candidate_details_map:
                             candidate_details_map[merged] = details
                        
                        if details.is_valid_check_digit:
                            detected_number = merged
                            break # Found a valid merged number!

            # Collect all logical candidates (direct 11s + valid merged ones)
            final_candidate_numbers = list(candidate_details_map.keys())

            # Determine success and reason
            if detected_number:
                success = True
                reason = "OK (Valid 11-digit Number)"
                # If it was a merged number, note that? 
            elif final_candidate_numbers:
                success = False
                reason = "Candidates found but invalid check digit"
            else:
                success = False
                reason = "No valid 11-digit candidate found"

            final[wagon_key] = WagonNumberResult(
                wagon_name=wagon_key,
                detected_number=detected_number,
                candidate_numbers=final_candidate_numbers,
                success=success,
                reason=reason,
                crop_texts=all_texts,
                candidates_details=candidate_details_map,
            )

        return final

    # -------------------------------------------------

    def save_results(self, results: Dict[str, WagonNumberResult]):
        """
        Save:
          - stage6_wagon_numbers.json    (compact per wagon result)
          - stage6_wagon_numbers.csv     (flat table)
          - stage6_wagon_numbers_detailed.json (full details, including decoded fields)
        """
        json_path = self.results_dir / "stage6_wagon_numbers.json"
        csv_path = self.results_dir / "stage6_wagon_numbers.csv"
        detailed_json_path = self.results_dir / "stage6_wagon_numbers_detailed.json"

        # JSON (compact)
        serializable: Dict[str, Any] = {}
        for key, res in results.items():
            serializable[key] = {
                "wagon_name": res.wagon_name,
                "detected_number": res.detected_number,
                "candidate_numbers": res.candidate_numbers,
                "success": res.success,
                "reason": res.reason,
                "crop_texts": res.crop_texts,
            }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        # Detailed JSON
        detailed: Dict[str, Any] = {}
        for key, res in results.items():
            cand_details_dict = {
                num: asdict(details)
                for num, details in res.candidates_details.items()
            }
            detailed[key] = {
                "wagon_name": res.wagon_name,
                "detected_number": res.detected_number,
                "success": res.success,
                "reason": res.reason,
                "crop_texts": res.crop_texts,
                "candidates": cand_details_dict,
            }
        with open(detailed_json_path, "w", encoding="utf-8") as f:
            json.dump(detailed, f, indent=2, ensure_ascii=False)

        # CSV (flat)
        rows = []
        for key, res in results.items():
            dn = res.detected_number or ""
            decoded = res.candidates_details.get(dn) if dn else None
            row = {
                "wagon_name": res.wagon_name,
                "detected_number": dn,
                "success": res.success,
                "reason": res.reason,
                "candidate_numbers": " | ".join(res.candidate_numbers),
            }
            # Add decoded columns for selected number (if valid)
            if decoded and decoded.parsed_ok:
                row.update(
                    {
                        "wagon_type_code": decoded.wagon_type_code,
                        "owning_railway_code": decoded.owning_railway_code,
                        "year_of_manufacture": decoded.year_of_manufacture,
                        "raw_year_digits": decoded.raw_year_digits,
                        "individual_serial": decoded.individual_serial,
                        "check_digit_valid": decoded.is_valid_check_digit,
                        "expected_check_digit": decoded.expected_check_digit,
                        "actual_check_digit": decoded.actual_check_digit,
                    }
                )
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

        print("\n[INFO] Wagon number extraction outputs saved:")
        print(f" - JSON:            {json_path}")
        print(f" - CSV:             {csv_path}")
        print(f" - Detailed JSON:   {detailed_json_path}")

    # -------------------------------------------------

    def run_full_pipeline(self, on_ocr_processed: Optional[callable] = None):
        """
        Convenience method: run CRAFT + Chandra + aggregation + decoding end-to-end.
        """
        self.run_stage6_craft()
        crop_results = self.run_chandra_on_crops(on_ocr_processed=on_ocr_processed)
        wagon_results = self._select_numbers_per_wagon(crop_results)
        self.save_results(wagon_results)
        return wagon_results


# =====================================================
# STANDALONE ENTRY
# =====================================================

if __name__ == "__main__":
    base_outputs = r"C:\Users\Viranch Patel\Desktop\Wagon_Inspection_System\outputs"
    craft_weights = (
        r"C:\Users\Viranch Patel\Desktop\Wagon_Inspection_System\models\weights\craft_mlt_25k.pth"
    )

    extractor = WagonNumberExtractor(
        outputs_root=base_outputs,
        stage3_dirname="stage3_enhanced_frames",
        craft_model_path=craft_weights,
        ocr_server_url="http://127.0.0.1:8001/ocr",
    )

    res = extractor.run_full_pipeline()
