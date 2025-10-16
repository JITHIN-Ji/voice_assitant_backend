
import os
import logging
from faster_whisper import WhisperModel
from pydub import AudioSegment
from app.agent.config import logger 

def load_whisper_model(model_name: str = "medium.en", device: str = "cpu") -> WhisperModel:
    """Load faster-whisper model"""
    try:
        compute_type = "int8" if device == "cpu" else "float16"
        logger.info(f"Loading Whisper model '{model_name}' on {device} (compute_type={compute_type})...")
        whisper_model = WhisperModel(model_name, device=device, compute_type=compute_type)
        logger.info("Whisper model loaded successfully")
        return whisper_model
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        raise

def ensure_wav(audio_path: str) -> str:
    """Convert MP3/M4A/FLAC to WAV if needed"""
    if audio_path.lower().endswith((".mp3", ".m4a", ".flac")):
        temp_dir = os.path.dirname(audio_path)
        wav_path = os.path.join(temp_dir, f"{os.path.splitext(os.path.basename(audio_path))[0]}.wav")

        if os.path.exists(wav_path):
            return wav_path

        try:
            if audio_path.lower().endswith(".mp3"):
                AudioSegment.from_mp3(audio_path).export(wav_path, format="wav")
            elif audio_path.lower().endswith(".m4a"):
                AudioSegment.from_file(audio_path, format="m4a").export(wav_path, format="wav")
            elif audio_path.lower().endswith(".flac"):
                AudioSegment.from_file(audio_path, format="flac").export(wav_path, format="wav")
            
            logger.info(f"Converted {os.path.basename(audio_path)} to {os.path.basename(wav_path)}")
            return wav_path
        except Exception as e:
            logger.warning(f"Failed to convert {audio_path} to WAV: {e}. Returning original path.")
            return audio_path
    return audio_path

def transcribe_file(whisper_model: WhisperModel, audio_path: str, beam_size: int = 5) -> str:
    """Transcribe audio file with Whisper"""
    if not whisper_model:
        logger.error("Whisper model not loaded. Cannot transcribe.")
        return ""
    try:
        segments, _ = whisper_model.transcribe(audio_path, word_timestamps=False, beam_size=beam_size)
        text_parts = []
        for seg in segments:
            if hasattr(seg, "text"):
                text_parts.append(seg.text.strip())
            else:
                text_parts.append(str(seg).strip())
        text = " ".join([p for p in text_parts if p])
        return " ".join(text.split())
    except Exception as e:
        logger.error(f"Transcription failed for {audio_path}: {e}")
        return ""