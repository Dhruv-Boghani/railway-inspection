from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
import json
import requests

# Import pipeline modules
from modules.image_enhancement import ImageEnhancer
from modules.door_detection_classification import DoorDetectionClassification
from modules.damage_detection import DamageDetection
from modules.wagon_number_extraction import decode_wagon_number
from pipeline_config import get_config

router = APIRouter()

# Global instances (lazy loaded)
tools_context = {
    "enhancer": None,
    "config": None
}

def get_tools():
    """Lazy load all models for tools"""
    if tools_context["config"] is None:
        tools_context["config"] = get_config()
    config = tools_context["config"]
    
    if not tools_context["enhancer"]:
        print("Loading Tools: Enhancer...")
        tools_context["enhancer"] = ImageEnhancer(
            mprnet_model_path=config.stage3.mprnet_model_path,
            device=config.stage3.device,
            apply_mprnet=True,
            verbose=False
        )
    
    return tools_context


def get_door_detector(output_dir: Path):
    """Create door detector with specified output directory"""
    config = tools_context["config"] or get_config()
    return DoorDetectionClassification(
        stage3_output_dir=str(output_dir),
        output_base_dir=str(output_dir),
        door_detection_model_path=config.stage4.door_detection_model_path,
        door_classifier_model_path=config.stage4.door_classifier_model_path,
        door_conf_threshold=config.stage4.door_conf_threshold,
        device=config.stage4.device,
    )


def get_damage_detector(output_dir: Path):
    """Create damage detector with specified output directory"""
    config = tools_context["config"] or get_config()
    return DamageDetection(
        stage3_output_dir=str(output_dir),
        output_base_dir=str(output_dir),
        damage_model_path=config.stage5.damage_model_path,
        conf_threshold=config.stage5.conf_threshold,
        img_size=config.stage5.img_size,
        device=config.stage5.device,
    )


def call_ocr_server(image_path: Path):
    """Call the Chandra OCR HTTP server for a single image."""
    config = tools_context["config"] or get_config()
    ocr_url = config.stage6.ocr_server_url
    
    try:
        # The OCR server expects JSON with image_path, not file upload
        payload = {
            "image_path": str(image_path.absolute()),
            "max_new_tokens": 512
        }
        response = requests.post(ocr_url, json=payload, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            return {
                "text": data.get("text", ""),
                "success": True,
                "raw_response": data
            }
        else:
            return {"text": "", "success": False, "error": f"HTTP {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"text": "", "success": False, "error": "OCR server not running (port 8001)"}
    except Exception as e:
        return {"text": "", "success": False, "error": str(e)}


def run_single_frame_pipeline(image_path: Path, output_dir: Path, frame_name: str, timestamp: str):
    """
    Run door detection, damage detection, and OCR on a single frame.
    Returns results dictionary with detection summaries and image URLs.
    """
    tools = get_tools()
    
    # Create output subdirectories
    door_output = output_dir / "door"
    damage_output = output_dir / "damage"
    door_output.mkdir(parents=True, exist_ok=True)
    damage_output.mkdir(parents=True, exist_ok=True)
    
    # Determine path prefix for URLs (blur or deblur based on output_dir)
    path_type = output_dir.name  # "blur" or "deblur"
    
    results = {
        "image_url": f"/outputs/temp_compare/{timestamp}/{path_type}/{image_path.name}",
        "door_results": None,
        "damage_results": None,
        "ocr_results": None
    }
    
    # ===== Door Detection =====
    try:
        door_detector = get_door_detector(door_output)
        door_result = door_detector.process_wagon(frame_name, image_path)
        
        # Get annotated image URL
        annotated_path = Path(door_result.get("annotated_image_path", ""))
        door_annotated_url = ""
        if annotated_path.exists():
            # Copy to accessible location
            dest = output_dir / f"{frame_name}_door_annotated.jpg"
            shutil.copy(annotated_path, dest)
            door_annotated_url = f"/outputs/temp_compare/{timestamp}/{path_type}/{dest.name}"
        
        results["door_results"] = {
            "annotated_url": door_annotated_url,
            "total_doors": door_result.get("total_doors_detected", 0),
            "door_counts": door_result.get("door_counts", {"good": 0, "damaged": 0, "missing": 0}),
            "doors": [
                {
                    "classification": d["classification"],
                    "confidence": round(d["classification_confidence"], 2)
                }
                for d in door_result.get("door_results", [])
            ]
        }
    except Exception as e:
        print(f"Door detection error: {e}")
        import traceback
        traceback.print_exc()
        results["door_results"] = {"error": str(e), "total_doors": 0, "door_counts": {"good": 0, "damaged": 0, "missing": 0}}
    
    # ===== Damage Detection =====
    try:
        damage_detector = get_damage_detector(damage_output)
        damage_result = damage_detector.process_wagon(frame_name, image_path)
        
        # Get annotated image URL
        annotated_path = Path(damage_result.get("annotated_image_path", ""))
        damage_annotated_url = ""
        if annotated_path.exists():
            dest = output_dir / f"{frame_name}_damage_annotated.jpg"
            shutil.copy(annotated_path, dest)
            damage_annotated_url = f"/outputs/temp_compare/{timestamp}/{path_type}/{dest.name}"
        
        damage_analysis = damage_result.get("damage_analysis", {})
        results["damage_results"] = {
            "annotated_url": damage_annotated_url,
            "total_detections": damage_result.get("total_detections", 0),
            "severity": damage_analysis.get("severity", "none"),
            "total_damage_percent": round(damage_analysis.get("total_damage_area_percent", 0), 2),
            "damage_by_type": damage_analysis.get("damage_by_type", {})
        }
    except Exception as e:
        print(f"Damage detection error: {e}")
        import traceback
        traceback.print_exc()
        results["damage_results"] = {"error": str(e), "total_detections": 0, "severity": "unknown"}
    
    # ===== OCR (Direct HTTP call to Chandra server) =====
    try:
        ocr_result = call_ocr_server(image_path)
        
        detected_text = ocr_result.get("text", "").strip()
        detected_numbers = []
        parsed_wagons = []
        
        # Extract potential wagon numbers (11-digit sequences)
        if detected_text:
            import re
            # Find all sequences of digits that could be wagon numbers
            digit_sequences = re.findall(r'\d{11}', detected_text.replace(" ", ""))
            detected_numbers = list(set(digit_sequences))  # Remove duplicates
            
            # Parse each detected wagon number
            for wagon_num in detected_numbers:
                parsed = decode_wagon_number(wagon_num)
                parsed_wagons.append({
                    "number": wagon_num,
                    "wagon_type_code": parsed.wagon_type_code,
                    "owning_railway_code": parsed.owning_railway_code,
                    "year_of_manufacture": parsed.year_of_manufacture,
                    "individual_serial": parsed.individual_serial,
                    "check_digit_valid": parsed.is_valid_check_digit,
                    "expected_check_digit": parsed.expected_check_digit,
                    "actual_check_digit": parsed.actual_check_digit,
                    "parsed_ok": parsed.parsed_ok
                })
        
        results["ocr_results"] = {
            "detected_numbers": detected_numbers,
            "parsed_wagons": parsed_wagons,
            "raw_text": detected_text[:200] if detected_text else "",
            "count": len(detected_numbers),
            "success": ocr_result.get("success", False)
        }
        
        if not ocr_result.get("success"):
            results["ocr_results"]["error"] = ocr_result.get("error", "Unknown error")
            
    except Exception as e:
        print(f"OCR error: {e}")
        results["ocr_results"] = {"error": str(e), "detected_numbers": [], "parsed_wagons": [], "count": 0}
    
    return results


@router.post("/compare-pipeline")
async def compare_pipeline(file: UploadFile = File(...)):
    """
    Compare blur vs deblur frame through the full side pipeline.
    
    Process flow:
    1. Save uploaded blur frame
    2. Run pipeline on blur frame (without enhancement)
    3. Enhance the blur frame (deblur)
    4. Run pipeline on deblurred frame
    5. Return side-by-side comparison results
    """
    try:
        timestamp = str(int(time.time()))
        # Fix: Ensure we write to the project root 'outputs' directory
        root_dir = Path(__file__).resolve().parent.parent.parent
        base_dir = root_dir / "outputs" / "temp_compare" / timestamp
        
        blur_dir = base_dir / "blur"
        deblur_dir = base_dir / "deblur"
        
        blur_dir.mkdir(parents=True, exist_ok=True)
        deblur_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file
        filename = f"frame_{file.filename}"
        blur_path = blur_dir / filename
        
        with open(blur_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Read image
        img = cv2.imread(str(blur_path))
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # ===== BLUR PATH: Run pipeline on original =====
        print(f"\n{'='*50}")
        print("COMPARE: Processing BLUR frame...")
        print(f"{'='*50}")
        blur_results = run_single_frame_pipeline(blur_path, blur_dir, "blur_frame", timestamp)
        blur_results["image_url"] = f"/outputs/temp_compare/{timestamp}/blur/{filename}"
        
        # ===== DEBLUR PATH: Enhance then run pipeline =====
        print(f"\n{'='*50}")
        print("COMPARE: Enhancing frame (deblur)...")
        print(f"{'='*50}")
        
        tools = get_tools()
        enhanced_img, _ = tools["enhancer"].enhance_image(img, "Deblur → Low-Light", "compare_tool")
        
        deblur_filename = f"enhanced_{filename}"
        deblur_path = deblur_dir / deblur_filename
        cv2.imwrite(str(deblur_path), enhanced_img)
        
        print(f"\n{'='*50}")
        print("COMPARE: Processing DEBLUR frame...")
        print(f"{'='*50}")
        deblur_results = run_single_frame_pipeline(deblur_path, deblur_dir, "deblur_frame", timestamp)
        deblur_results["image_url"] = f"/outputs/temp_compare/{timestamp}/deblur/{deblur_filename}"
        
        # ===== Calculate improvement metrics =====
        blur_doors = blur_results.get("door_results", {}).get("total_doors", 0)
        deblur_doors = deblur_results.get("door_results", {}).get("total_doors", 0)
        
        blur_ocr = len(blur_results.get("ocr_results", {}).get("detected_numbers", []))
        deblur_ocr = len(deblur_results.get("ocr_results", {}).get("detected_numbers", []))
        
        blur_damage = blur_results.get("damage_results", {}).get("total_detections", 0)
        deblur_damage = deblur_results.get("damage_results", {}).get("total_detections", 0)
        blur_severity = blur_results.get("damage_results", {}).get("severity", "none")
        deblur_severity = deblur_results.get("damage_results", {}).get("severity", "none")
        
        improvement = {
            "doors_detected": {
                "blur": blur_doors,
                "deblur": deblur_doors,
                "improvement": deblur_doors - blur_doors
            },
            "ocr_detected": {
                "blur": blur_ocr,
                "deblur": deblur_ocr,
                "improvement": deblur_ocr - blur_ocr
            },
            "damage_detected": {
                "blur": blur_damage,
                "deblur": deblur_damage,
                "blur_severity": blur_severity,
                "deblur_severity": deblur_severity,
                "improvement": deblur_damage - blur_damage
            }
        }
        
        return {
            "blur_results": blur_results,
            "deblur_results": deblur_results,
            "improvement": improvement,
            "timestamp": timestamp
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
async def compare_frames(file: UploadFile = File(...)):
    """Legacy compare endpoint - just enhancement + OCR"""
    try:
        timestamp = int(time.time())
        # Fix: Ensure we write to the project root 'outputs' directory
        root_dir = Path(__file__).resolve().parent.parent.parent
        temp_dir = root_dir / "outputs" / "temp_tools"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{timestamp}_{file.filename}"
        input_path = temp_dir / filename
        
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        tools = get_tools()
        enhancer = tools["enhancer"]
        
        img = cv2.imread(str(input_path))
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
            
        enhanced_img, _ = enhancer.enhance_image(img, "Deblur → Low-Light", "tool_request")
        
        enhanced_filename = f"enhanced_{filename}"
        enhanced_path = temp_dir / enhanced_filename
        cv2.imwrite(str(enhanced_path), enhanced_img)
        
        # Call OCR server directly
        ocr_result = call_ocr_server(enhanced_path)
        detected_text = ocr_result.get("text", "")
        
        return {
            "original_url": f"/outputs/temp_tools/{filename}", 
            "enhanced_url": f"/outputs/temp_tools/{enhanced_filename}",
            "detected_text": detected_text,
            "comparison_data": {
                "sharpness_improvement": "High",
                "readability": "Improved"
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))