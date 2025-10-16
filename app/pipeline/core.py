
import logging
from typing import List, Tuple, Dict, Set, Optional
from faster_whisper import WhisperModel # Import only for type hinting
import spacy # Import only for type hinting
from app.pipeline.audio_utils import load_whisper_model, ensure_wav, transcribe_file

from app.pipeline.gemini_llm import query_gemini_summary
from app.pipeline.nlp_utils import load_ner_model, extract_entities, calculate_ner_metrics


class MedicalAudioProcessor:
    def __init__(self, audio_dir: str = "recordings")-> None:
        self.audio_dir = audio_dir
        self.nlp: spacy.Language = None
        self.whisper_model: WhisperModel = None
        
    def load_models(self, whisper_model_name: str = "medium.en", device: str = "cpu") -> None:
        """Load Whisper and scispaCy NER models"""
        self.whisper_model = load_whisper_model(whisper_model_name, device)
        self.nlp = load_ner_model()

    def ensure_wav(self, audio_path: str) -> str:
        return ensure_wav(audio_path)

    def transcribe_file(self, audio_path: str, beam_size: int = 5) -> str:
        return transcribe_file(self.whisper_model, audio_path, beam_size)

    def query_gemini(self, transcript: str) -> str:
        return query_gemini_summary(transcript)

    def extract_entities(self, text: str) -> list: 
        return extract_entities(self.nlp, text)

    def calculate_ner_metrics(self, reference_entities: set, system_entities: set) -> dict: 
        return calculate_ner_metrics(reference_entities, system_entities)