"""
Chandra OCR HTTP Server

Runs inside the OCR virtual environment (chandra_ocr_venv) and exposes
a simple HTTP API so other environments can use ChandraOCR without
subprocess overhead.

Endpoints:
- POST /ocr
  Body: {"image_path": "C:\\path\\to\\image.jpg"}
  Returns: ChandraOCR.extract_text(...) result as JSON.
"""

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any

from modules.chandra_ocr_text_extraction import ChandraOCR


app = FastAPI(title="Chandra OCR Server")

# Load the OCR model ONCE at server startup
print("=" * 80)
print("[SERVER] Starting Chandra OCR server and loading model...")
print("=" * 80)

# Adjust model_name / device as per your hardware
ocr_engine = ChandraOCR(
    model_name="Qwen/Qwen2-VL-2B-Instruct",  # smaller & faster than 7B
    device="cuda",                           # or "auto" / "cpu"
    use_8bit=True,
)


class OCRRequest(BaseModel):
    image_path: str
    prompt: Optional[str] = None
    max_new_tokens: Optional[int] = 512


@app.post("/ocr")
def ocr_endpoint(req: OCRRequest) -> Dict[str, Any]:
    """
    Run OCR on a single image_path and return JSON with:
    - text
    - time_sec
    - image_name
    - dimensions
    - etc. (whatever ChandraOCR.extract_text returns)
    """
    prompt = req.prompt or (
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

    result = ocr_engine.extract_text(
        image=req.image_path,
        prompt=prompt,
        max_new_tokens=req.max_new_tokens or 512,
        verbose=False,
    )
    # Ensure we always return a dict FastAPI can serialize
    return result


@app.on_event("shutdown")
def shutdown_event():
    """Cleanup OCR engine on server shutdown."""
    print("[SERVER] Shutting down Chandra OCR server, cleaning up...")
    try:
        ocr_engine.cleanup()
    except Exception:
        pass


if __name__ == "__main__":
    # Run the server on localhost:8001
    uvicorn.run(
        "chandra_ocr_server:app",
        host="127.0.0.1",
        port=8001,
        reload=False,
        workers=1,
    )
