import threading
import time
import json
import traceback
from pathlib import Path
from datetime import datetime
from sqlalchemy.orm import Session
import sys

# Add parent directory to sys.path to allow importing from root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from database import SessionLocal, Job
from main import WagonInspectionPipeline, get_config

class PipelineRunner:
    def __init__(self):
        self.active_jobs = {}

    def start_job(self, job_id: int):
        thread = threading.Thread(target=self._run_job, args=(job_id,))
        thread.start()

    def _update_job(self, db: Session, job_id: int, **kwargs):
        try:
            job = db.query(Job).filter(Job.id == job_id).first()
            if job:
                for k, v in kwargs.items():
                    setattr(job, k, v)
                db.commit()
                db.refresh(job)
        except Exception as e:
            print(f"Error updating job {job_id}: {e}")

    def _update_live_count(self, db: Session, job_id: int, key: str, value: int):
        """Update a specific live count value, merging with existing counts"""
        try:
            # Use a fresh session to avoid threading issues
            fresh_db = SessionLocal()
            try:
                job = fresh_db.query(Job).filter(Job.id == job_id).first()
                if job:
                    # Get current counts and update
                    current_counts = dict(job.live_counts) if job.live_counts else {}
                    current_counts[key] = value
                    # Force SQLAlchemy to recognize the change by assigning a new dict
                    job.live_counts = current_counts.copy()
                    fresh_db.commit()
            finally:
                fresh_db.close()
        except Exception as e:
            print(f"Error updating live count {key} for job {job_id}: {e}")

    def _run_job(self, job_id: int):
        db = SessionLocal()
        try:
            job = db.query(Job).filter(Job.id == job_id).first()
            if not job:
                return

            self._update_job(db, job_id, status="processing", current_stage="Initializing", progress=0.0)

            # --- SETUP CONFIG ---
            config = get_config()
            config.stage1.direction = job.direction
            
            # Create unique output directory for this job
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Fix: Ensure we write to the project root 'outputs' directory, not backend/outputs
            # Since we are likely running from 'backend' dir, go up one level
            root_dir = Path(__file__).resolve().parent.parent
            job_output_dir = root_dir / "outputs" / f"job_{job_id}_{timestamp}"
            job_output_dir.mkdir(parents=True, exist_ok=True)
            config.output_root = job_output_dir
            
            # Update job with output_dir
            self._update_job(db, job_id, output_dir=str(job_output_dir))

            # Select pipeline based on camera_angle
            if job.camera_angle == "TOP":
                # ============================================================
                # TOP CAMERA PIPELINE (5 stages, no doors/OCR)
                # ============================================================
                pipeline_start_time = time.time()  # Track processing time
                from main_top_pipeline import TopInspectionPipeline
                pipeline = TopInspectionPipeline(config=config, verbose=True)
                
                # Create callback for real-time wagon count updates
                def on_wagon_counted_callback(count):
                    self._update_live_count(db, job_id, "wagons_counted", count)
                
                # --- STAGE 1 (TOP) ---
                self._update_job(db, job_id, current_stage="Stage 1: Top Detection & Counting", progress=10.0)
                wagon_data = pipeline.run_stage1(
                    video_path=job.video_path, 
                    output_video_path=str(job_output_dir / "stage1_annotated.mp4"),
                    on_wagon_counted=on_wagon_counted_callback  # Real-time updates!
                )
                
                wagons_count = len(wagon_data)
                self._update_job(db, job_id, total_wagons=wagons_count, progress=25.0)
                self._update_live_count(db, job_id, "wagons_counted", wagons_count)
                
                if wagons_count == 0:
                    self._update_job(db, job_id, status="failed", error_message="No wagons detected")
                    return
                
                # --- STAGE 2 (TOP) ---
                self._update_job(db, job_id, current_stage="Stage 2: Quality Assessment", progress=40.0)
                
                def on_frame_assessed_callback(count):
                    self._update_live_count(db, job_id, "frames_assessed", count)
                    
                pipeline.run_stage2(on_frame_assessed=on_frame_assessed_callback)
                self._update_live_count(db, job_id, "frames_assessed", wagons_count)
                
                # --- STAGE 3 (TOP) ---
                self._update_job(db, job_id, current_stage="Stage 3: Image Enhancement", progress=55.0)
                
                def on_frame_enhanced_callback(count):
                    self._update_live_count(db, job_id, "frames_enhanced", count)
                    
                pipeline.run_stage3(on_frame_enhanced=on_frame_enhanced_callback)
                self._update_live_count(db, job_id, "frames_enhanced", wagons_count)
                
                # --- STAGE 4 (TOP) ---
                self._update_job(db, job_id, current_stage="Stage 4: Top Damage Detection", progress=75.0)
                
                def on_top_damage_callback(count):
                    # count here represents cumulative defects found
                    self._update_live_count(db, job_id, "defects_found", count)

                damage_results = pipeline.run_stage4(on_damage_processed=on_top_damage_callback)
                
                # Update final defect count
                defects_count = 0
                if damage_results:
                     # Sum 'total_detections' from each wagon result
                    defects_count = sum(r.get("total_detections", 0) for r in damage_results.values())
                    
                self._update_job(db, job_id, defects_count=defects_count)
                self._update_live_count(db, job_id, "defects_found", defects_count)
                
                # --- STAGE 5 (TOP) = Aggregation ---
                self._update_job(db, job_id, current_stage="Stage 5: Aggregation", progress=90.0)
                total_processing_time = time.time() - pipeline_start_time
                final_results = pipeline.run_aggregation(damage_results, total_processing_time)
                
            else:
                # ============================================================
                # SIDE CAMERA PIPELINE (6 stages with doors/OCR)
                # ============================================================
                from main import WagonInspectionPipeline
                pipeline = WagonInspectionPipeline(config=config, verbose=True)
                
                # Create callback for real-time wagon count updates
                def on_wagon_counted_callback(count):
                    self._update_live_count(db, job_id, "wagons_counted", count)
                
                # --- STAGE 1 (SIDE) ---
                self._update_job(db, job_id, current_stage="Stage 1: Detection & Counting", progress=10.0)
                wagon_data = pipeline.run_stage1(
                    video_path=job.video_path, 
                    output_video_path=str(job_output_dir / "stage1_annotated.mp4"),
                    on_wagon_counted=on_wagon_counted_callback  # Real-time updates!
                )
            
                wagons_count = len(wagon_data)
                self._update_job(db, job_id, total_wagons=wagons_count, progress=20.0)
                self._update_live_count(db, job_id, "wagons_counted", wagons_count)
                
                if wagons_count == 0:
                    self._update_job(db, job_id, status="failed", error_message="No wagons detected")
                    return

                # --- STAGE 2 ---
                self._update_job(db, job_id, current_stage="Stage 2: Quality Assessment", progress=30.0)
                
                def on_frame_assessed_callback(count):
                    self._update_live_count(db, job_id, "frames_assessed", count)
                    
                pipeline.run_stage2(on_frame_assessed=on_frame_assessed_callback)
                self._update_live_count(db, job_id, "frames_assessed", wagons_count)
                
                # --- STAGE 3 ---
                self._update_job(db, job_id, current_stage="Stage 3: Image Enhancement", progress=40.0)
                
                def on_frame_enhanced_callback(count):
                    self._update_live_count(db, job_id, "frames_enhanced", count)
                    
                pipeline.run_stage3(on_frame_enhanced=on_frame_enhanced_callback)
                self._update_live_count(db, job_id, "frames_enhanced", wagons_count)
                
                # --- STAGE 4 ---
                self._update_job(db, job_id, current_stage="Stage 4: Door Detection", progress=55.0)
                
                # Create callback for real-time door count updates
                def on_door_counted_callback(count):
                    self._update_live_count(db, job_id, "doors_detected", count)
                    
                door_results = pipeline.run_stage4(on_door_counted=on_door_counted_callback)
                
                # Update final count
                doors_count = sum(r.get("total_doors_detected", 0) for r in door_results.values()) if door_results else 0
                self._update_live_count(db, job_id, "doors_detected", doors_count)
                
                # --- STAGE 5 ---
                self._update_job(db, job_id, current_stage="Stage 5: Damage Detection", progress=70.0)
                
                def on_damage_processed_callback(count):
                    # count here represents cumulative defects found
                    self._update_live_count(db, job_id, "defects_found", count)
                    
                damage_results = pipeline.run_stage5(on_damage_processed=on_damage_processed_callback)
                
                # Count final defects to ensure sync
                defects_count = sum(
                    1 for r in damage_results.values() if r.get("total_detections", 0) > 0
                ) if damage_results else 0
                
                # Update final job counts
                self._update_job(db, job_id, defects_count=defects_count)
                # Note: defects_count in job model might be wagons with defects vs total defects. 
                # Let's trust the run_stage5 logic which returns dict of wagons.
                # However, live count 'defects_found' tracks total defects.
                # We should update final live count with total defects too.
                total_defects_count = sum(r.get("total_detections", 0) for r in damage_results.values()) if damage_results else 0
                self._update_live_count(db, job_id, "defects_found", total_defects_count)

                # --- STAGE 6 ---
                self._update_job(db, job_id, current_stage="Stage 6: OCR Extraction", progress=85.0)
                
                def on_ocr_processed_callback(count):
                    # count here represents crops processed
                    self._update_live_count(db, job_id, "ocr_crops_processed", count)
                    
                ocr_results = pipeline.run_stage6(on_ocr_processed=on_ocr_processed_callback)
                
                # Count successful OCR extractions
                try:
                    if isinstance(ocr_results, list):
                        ocr_success_count = sum(1 for r in ocr_results if isinstance(r, dict) and r.get("crop_texts") and len(r.get("crop_texts", [])) > 0)
                    elif isinstance(ocr_results, dict):
                        ocr_success_count = sum(1 for r in ocr_results.values() if isinstance(r, dict) and r.get("crop_texts") and len(r.get("crop_texts", [])) > 0)
                    else:
                        ocr_success_count = 0
                except Exception as e:
                    print(f"Error counting OCR results: {e}")
                    ocr_success_count = 0
                self._update_live_count(db, job_id, "numbers_extracted", ocr_success_count)
                
                # --- AGGREGATION ---
                self._update_job(db, job_id, current_stage="Aggregation", progress=95.0)
                final_results = pipeline.run_aggregation(door_results, damage_results, ocr_results)
            
            # Save final status (common for both pipelines)
            json_path = job_output_dir / "final_results" / f"final_results_{timestamp}.json"
            # Since run_aggregation already saves it, we just need to find it.
            # But the timestamp inside run_aggregation might differ slightly if seconds ticked over.
            # Actually pipeline.run_aggregation uses datetime.now() inside. 
            # Let's hope logic is consistent. 
            # Helper: find the json file
            try:
                found_jsons = list((job_output_dir / "final_results").glob("*.json"))
                if found_jsons:
                    result_json_path = str(found_jsons[0])
                    self._update_job(db, job_id, result_json_path=result_json_path)
            except:
                pass

            self._update_job(db, job_id, status="completed", progress=100.0, completed_at=datetime.now())

        except Exception as e:
            traceback.print_exc()
            self._update_job(db, job_id, status="failed", error_message=str(e))
        finally:
            db.close()

pipeline_runner = PipelineRunner()