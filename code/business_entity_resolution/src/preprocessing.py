"""
Smart Normalization and Preprocessing module for Entity Resolution.
Handles business names, addresses, and country attributes while keeping original fields intact.
"""
import re
from typing import Dict, List, Tuple
from src.config import LEGAL_SUFFIXES

# Regex pre-compilations for speed
PUNCT_RE = re.compile(r'[^a-zA-Z0-9\s]')
SPACE_RE = re.compile(r'\s+')
DIGIT_RE = re.compile(r'\b\d+\b')
POSTAL_RE = re.compile(r'\b\d{5,6}\b') # US 5-digit zip or India 6-digit PIN

ADDRESS_ABBREVIATIONS = {
    'rd': 'road',
    'st': 'street',
    'ave': 'avenue',
    'blvd': 'boulevard',
    'dr': 'drive',
    'ln': 'lane',
    'ct': 'court',
    'pl': 'place',
    'pkwy': 'parkway',
    'ste': 'suite',
    'apt': 'apartment',
    'no': 'number',
    'fl': 'floor',
    'bldg': 'building',
    'hwy': 'highway',
    'expwy': 'expressway',
    'opp': 'opposite',
    'nr': 'near',
}

def normalize_country(country: str) -> str:
    """Normalize country string to uppercase trimmed format."""
    if not country:
        return ""
    return str(country).strip().upper()

def normalize_name(name: str) -> Tuple[str, List[str], str]:
    """
    Normalizes business name.
    Returns:
      norm_name: cleaned string without legal suffixes
      tokens: list of cleaned non-legal tokens
      first_sig_token: first significant token (>2 chars, non-legal)
    """
    if not name:
        return "", [], ""
    
    clean = PUNCT_RE.sub(" ", str(name).lower())
    clean = SPACE_RE.sub(" ", clean).strip()
    
    tokens = clean.split()
    sig_tokens = [t for t in tokens if t not in LEGAL_SUFFIXES and len(t) > 1]
    
    norm_name = " ".join(sig_tokens) if sig_tokens else clean
    first_sig = sig_tokens[0] if sig_tokens else (tokens[0] if tokens else "")
    
    return norm_name, sig_tokens, first_sig

def normalize_address(address: str) -> Tuple[str, List[str], List[str], str]:
    """
    Normalizes address text.
    Returns:
      norm_addr: cleaned address with expanded abbreviations
      tokens: list of tokenized words
      digits: list of extracted numeric tokens (house numbers, unit numbers)
      postal_code: extracted 5/6 digit postal/PIN code if present
    """
    if not address or str(address).lower() == 'nan':
        return "", [], [], ""
    
    raw = str(address).lower()
    
    # Extract postal/PIN code before heavy cleaning
    postal_matches = POSTAL_RE.findall(raw)
    postal_code = postal_matches[-1] if postal_matches else ""
    
    clean = PUNCT_RE.sub(" ", raw)
    clean = SPACE_RE.sub(" ", clean).strip()
    
    tokens = clean.split()
    standardized_tokens = [ADDRESS_ABBREVIATIONS.get(t, t) for t in tokens]
    
    norm_addr = " ".join(standardized_tokens)
    digits = DIGIT_RE.findall(clean)
    
    return norm_addr, standardized_tokens, digits, postal_code
