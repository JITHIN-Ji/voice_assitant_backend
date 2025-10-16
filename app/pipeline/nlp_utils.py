
import logging
import spacy
import pandas as pd
from typing import Dict, List, Tuple
from app.agent.config import logger 
from typing import Dict, List, Tuple, Set

def load_ner_model() -> spacy.Language:
    """Load scispaCy NER model"""
    try:
        nlp = spacy.load("en_ner_bc5cdr_md")
        logger.info("scispaCy model loaded successfully")
        return nlp
    except OSError:
        logger.error("scispaCy model not found. Install with: pip install scispacy && python -m spacy download en_ner_bc5cdr_md")
        raise

def extract_entities(nlp_model: spacy.Language, text: str) -> List[Tuple[str, str]]:
    """Extract entities using scispaCy"""
    if not nlp_model:
        logger.error("scispaCy NER model not loaded. Cannot extract entities.")
        return []
    if not text or pd.isna(text):
        return []
    
    doc = nlp_model(text)
    return [(ent.text.lower().strip(), ent.label_) for ent in doc.ents if ent.text.strip()]

def calculate_ner_metrics(reference_entities: set, system_entities: set) -> Dict[str, float]:
    """Calculate NER evaluation metrics"""
    TP = len(reference_entities & system_entities)
    FP = len(system_entities - reference_entities)
    FN = len(reference_entities - system_entities)
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'Precision': round(precision, 3),
        'Recall': round(recall, 3),
        'F1_Score': round(f1, 3)
    }