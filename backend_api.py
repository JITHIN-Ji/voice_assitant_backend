# backend_api.py

import os
import json
import uuid
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import logging

# Ensure these imports are correct based on your project structure
# Assuming backend_api.py is in the root, and app/ is a sibling directory.
# You might need to adjust your PYTHONPATH or ensure your IDE recognizes 'app' as a package.
# Example for running: `uvicorn backend_api:app --host 0.0.0.0 --port 5000` from project root
from app.pipeline.core import MedicalAudioProcessor
from app.agent.config import set_session_id, logger, GEMINI_API_KEY
from app.agent.core import process_medicines, process_appointment

# Load environment variables
load_dotenv()

# --- FastAPI App Setup ---
app = FastAPI(
    title="Medical Audio Processor API",
    description="API for processing medical audio, generating SOAP notes, and executing treatment plans.",
    version="1.0.0"
)

# Configure CORS for local development with React Native
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development. In production, restrict this.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = 'recordings_backend'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Global MedicalAudioProcessor Instance ---
# This ensures models are loaded only once when the FastAPI app starts
logger.info("Initializing MedicalAudioProcessor for backend...")
processor = MedicalAudioProcessor(UPLOAD_FOLDER)
try:
    processor.load_models(whisper_model_name="medium.en", device="cpu") # Use "cpu" for broader compatibility
    logger.info("MedicalAudioProcessor models loaded successfully for backend.")
except Exception as e:
    logger.error(f"Failed to load MedicalAudioProcessor models for backend: {e}")
    raise RuntimeError(f"Failed to load required ML models: {e}") from e


# --- API Endpoints ---

@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "Medical Audio Processor Backend is running!"}

@app.post("/process_audio")
async def process_audio_api(
    audio: UploadFile = File(...),
    section_text: str = Form(""),
    session_id: str = Form(None)
):
    """
    Processes an uploaded audio file to generate a medical transcript and SOAP summary.
    Optionally, provides reference text for NER evaluation.
    """
    # Use provided session_id if available, else generate a new one
    session_id = set_session_id(session_id or str(uuid.uuid4())[:8])
    logger.info(f"[{session_id}] Received request for audio processing. Audio filename: {audio.filename}")

    if not audio.filename:
        logger.error(f"[{session_id}] No audio file provided.")
        raise HTTPException(status_code=400, detail="No audio file provided.")

    # Save the uploaded file temporarily
    # Using tempfile for better cleanup, though in this example, UPLOAD_FOLDER is still used for processor context
    # It's good practice to save to a known location, and then use processor logic.
    try:
        # Create a temporary file to save the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, dir=UPLOAD_FOLDER, suffix=os.path.splitext(audio.filename)[1]) as temp_audio_file:
            contents = await audio.read()
            temp_audio_file.write(contents)
            filepath = temp_audio_file.name
        logger.info(f"[{session_id}] Audio file saved to {filepath}")

        # Re-use the existing pipeline logic
        wav_path = processor.ensure_wav(filepath)
        transcript = processor.transcribe_file(wav_path)
        
        if not transcript:
            logger.error(f"[{session_id}] Transcription failed for {audio.filename}.")
            raise HTTPException(status_code=500, detail="Failed to transcribe audio.")

        gemini_summary_raw = processor.query_gemini(transcript)
        
        if not gemini_summary_raw:
            logger.error(f"[{session_id}] Gemini summary generation failed for {audio.filename}.")
            raise HTTPException(status_code=500, detail="Failed to generate summary.")

        soap_sections = gemini_summary_raw

        response_data = {
            "transcript": transcript,
            "soap_sections": soap_sections,
            "audio_file_name": audio.filename
        }

        # Optional: Include NER metrics if section_text is provided
        if section_text and section_text.strip():
            try:
                section_entities = set(processor.extract_entities(section_text))
                # Gemini summary itself might contain entities, so we extract from its JSON representation
                summary_entities = set(processor.extract_entities(json.dumps(soap_sections)))
                metrics = processor.calculate_ner_metrics(section_entities, summary_entities)
                response_data["ner_metrics"] = metrics
                response_data["reference_entities"] = list(section_entities)
                response_data["generated_entities"] = list(summary_entities)
                logger.info(f"[{session_id}] NER metrics calculated.")
            except Exception as e:
                logger.warning(f"[{session_id}] NER metric calculation failed: {e}")
                response_data["ner_metrics_error"] = str(e)
        
        logger.info(f"[{session_id}] Audio processing successful. Sending response.")
        return JSONResponse(content=response_data, status_code=200)

    except Exception as e:
        logger.error(f"[{session_id}] Error during audio processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        # Clean up the temporary audio file
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"[{session_id}] Cleaned up temporary audio file: {filepath}")


@app.post("/approve_plan")
async def approve_plan_api(payload: dict):
    """
    Approves the extracted medical plan and executes agent actions
    like processing medicines and scheduling appointments.
    """
    # Reuse client-provided session_id if present to correlate logs across requests
    client_session_id = payload.get('session_id') if isinstance(payload, dict) else None
    session_id = set_session_id(client_session_id or str(uuid.uuid4())[:8])
    logger.info(f"[{session_id}] Received request for plan approval.")

    plan_section = payload.get('plan_section')
    user_email = payload.get('user_email', 'default_patient@example.com') # Fallback email
    send_email = bool(payload.get('send_email', True))

    if not plan_section or plan_section.strip().lower() == "n/a":
        logger.warning(f"[{session_id}] No valid plan section provided for approval.")
        return JSONResponse(content={"status": "warning", "message": "No valid plan section provided for approval."}, status_code=200)

    results = {}
    try:
        # Process Medicines
        logger.info(f"[{session_id}] Processing medicines...")
        medicine_res = process_medicines(plan_section)
        results['medicine_processing'] = medicine_res

        # Process Appointment - Generate content first (send_email flag controls sending)
        logger.info(f"[{session_id}] Generating appointment email content...")
        appointment_preview_res = process_appointment(plan_section, user_email, send_email=False)
        results['appointment_preview'] = appointment_preview_res

        if appointment_preview_res["status"] == "success" and "email_content" in appointment_preview_res:
            if send_email:
                # If email content was generated and sending requested, now actually send it
                logger.info(f"[{session_id}] Sending appointment email...")
                appointment_send_res = process_appointment(plan_section, user_email, send_email=True)
                results['appointment_sending'] = appointment_send_res
                if appointment_send_res["status"] == "success":
                    results['message'] = "Plan approved and actions executed (including appointment email)."
                    logger.info(f"[{session_id}] Plan approved and actions executed successfully.")
                    return JSONResponse(content=results, status_code=200)
                else:
                    results['message'] = "Plan approved, but appointment email sending failed."
                    logger.error(f"[{session_id}] Appointment email sending failed: {appointment_send_res.get('error')}")
                    # Still return 200 if other parts succeeded, but indicate partial failure
                    return JSONResponse(content=results, status_code=200)
            else:
                results['message'] = "Plan approved. Email content generated for review; sending not requested."
                logger.info(f"[{session_id}] Plan approved; email content generated, not sent.")
                return JSONResponse(content=results, status_code=200)
        else:
            results['message'] = "Plan approved, but no appointment found or email generation failed."
            logger.info(f"[{session_id}] No appointment found or email generation failed.")
            return JSONResponse(content=results, status_code=200)

    except Exception as e:
        logger.error(f"[{session_id}] Error during plan approval: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server error during plan approval: {str(e)}")