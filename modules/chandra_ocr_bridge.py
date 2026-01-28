"""
Chandra OCR Bridge - Client for HTTP Server
"""

import requests
import json
import time
from pathlib import Path
from typing import Dict, List, Union
import numpy as np
from PIL import Image

class ChandraOCRBridge:
    """
    Bridge to Chandra OCR via HTTP Server.
    Assumes the OCR server is running on localhost:8001.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8001",
        # Arguments below are kept for API compatibility but unused
        ocr_env_path: str = None,
        model_name: str = None,
    ):
        self.server_url = server_url
        self.ocr_url = f"{self.server_url}/ocr"
        
        # Check if server is reachable
        try:
            # We can't easily check status without a ping endpoint, 
            # but we assume it's up or will fail gracefully later.
            self.available = True
            print(f"✅ Chandra OCR Bridge initialized (Server: {self.server_url})")
        except Exception as e:
            print(f"⚠️ OCR Server might be down: {e}")
            self.available = False

    def extract_text(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
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
        max_new_tokens: int = 256,
        verbose: bool = True,
    ) -> Dict:
        """
        Extract text from image by calling OCR HTTP Server.
        """
        if not self.available:
            # Try to reconnect?
            self.available = True

        # Handle image input
        temp_file_created = False
        image_path = None
        
        try:
            if isinstance(image, (str, Path)):
                image_path = str(image)
                # Verify exists
                if not Path(image_path).exists():
                     return {"error": f"Image not found: {image_path}", "text": "", "time_sec": 0}
            elif isinstance(image, (np.ndarray, Image.Image)):
                # Save to a temporary path that the server can access?
                # configured to run locally, so server can access local files if paths are absolute.
                # Ideally, we should send the image bytes, but the current server implementation 
                # (chandra_ocr_server.py) expects a 'image_path'.
                # So we must save it to disk.
                temp_path = Path(f"temp_ocr_client_{time.time()}.jpg").absolute()
                if isinstance(image, np.ndarray):
                    Image.fromarray(image).save(temp_path)
                else:
                    image.save(temp_path)
                image_path = str(temp_path)
                temp_file_created = True

            payload = {
                "image_path": str(Path(image_path).absolute()), # Send absolute path
                "prompt": prompt,
                "max_new_tokens": max_new_tokens
            }

            if verbose:
                print(f"🔄 Requesting OCR from {self.ocr_url}...")

            start_time = time.time()
            response = requests.post(self.ocr_url, json=payload, timeout=300)
            response.raise_for_status()
            
            result = response.json()
            
            # Ensure time_sec is present
            if "time_sec" not in result:
                result["time_sec"] = time.time() - start_time

            return result

        except Exception as e:
            if verbose:
                print(f"❌ OCR Client Error: {e}")
            return {
                "text": "",
                "error": str(e),
                "time_sec": 0.0,
                "status": "failed"
            }
        finally:
            if temp_file_created and image_path and Path(image_path).exists():
                try:
                    Path(image_path).unlink()
                except:
                    pass

    def extract_text_batch(
        self,
        images: List[Union[str, Path]],
        output_dir: str = "outputs/ocr",
        verbose: bool = True,
    ) -> List[Dict]:
        """
        Extract text from batch (sequentially via HTTP).
        """
        results = []
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        
        for idx, img in enumerate(images):
            if verbose:
                print(f"Processing OCR {idx+1}/{len(images)}: {img}")
            
            res = self.extract_text(img, verbose=False)
            results.append(res)
            
            # Save individual result
            out_path = output_root / f"ocr_result_{idx:04d}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
                
        return results

    def cleanup(self):
        pass

# Backward compatibility
ChandraOCR = ChandraOCRBridge
