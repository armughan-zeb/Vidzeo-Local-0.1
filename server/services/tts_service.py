# =============================================================================
# TTS SERVICE - Kokoro Text-to-Speech
# =============================================================================

import numpy as np
import soundfile as sf
from pathlib import Path
import sys
import os

# Add config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import VOICES, TEMP_DIR

# Global state
_pipeline = None
_available = False


def init_tts():
    """Initialize Kokoro TTS pipeline"""
    global _pipeline, _available
    
    if _pipeline is not None:
        return _available
    
    try:
        from kokoro import KPipeline
        _pipeline = KPipeline(lang_code='a')
        _available = True
        print("✅ Kokoro TTS loaded")
    except Exception as e:
        print(f"❌ Kokoro TTS failed: {e}")
        _pipeline = None
        _available = False
    
    return _available


def is_available():
    """Check if TTS is available"""
    return _available


def get_voices():
    """Get list of available voices"""
    return list(VOICES.keys())


def get_voice_id(voice_name):
    """Get voice ID from display name"""
    return VOICES.get(voice_name, 'af_sky')


def generate_audio(script: str, voice: str = 'Adam (Male US)', speed: float = 1.0) -> tuple:
    """
    Generate audio from script using Kokoro TTS.
    
    Args:
        script: Text to synthesize
        voice: Voice display name
        speed: Speech speed (0.5-2.0)
    
    Returns:
        tuple: (audio_path, duration_seconds)
    """
    global _pipeline
    
    if not _available or _pipeline is None:
        raise RuntimeError("TTS not available. Run init_tts() first.")
    
    voice_id = get_voice_id(voice)
    
    # Generate audio segments
    audio_parts = []
    for _, _, audio in _pipeline(script, voice=voice_id, speed=speed, split_pattern=r'\n+'):
        if audio is not None and len(audio) > 0:
            audio_parts.append(audio)
    
    if not audio_parts:
        raise ValueError("No audio generated from script")
    
    # Concatenate all parts
    full_audio = np.concatenate(audio_parts)
    
    # Save to file
    import hashlib
    import time
    audio_hash = hashlib.md5(f"{script[:50]}{time.time()}".encode()).hexdigest()[:8]
    audio_path = TEMP_DIR / f"audio_{audio_hash}.wav"
    
    sf.write(str(audio_path), full_audio, 24000)
    
    duration = len(full_audio) / 24000
    
    return str(audio_path), duration


def preview_voice(voice: str, text: str = "Hello! This is a preview of my voice.") -> str:
    """
    Generate a short voice preview.
    
    Args:
        voice: Voice display name
        text: Preview text
    
    Returns:
        str: Path to preview audio file
    """
    global _pipeline
    
    if not _available or _pipeline is None:
        raise RuntimeError("TTS not available")
    
    voice_id = get_voice_id(voice)
    
    # Generate preview
    audio_parts = []
    for _, _, audio in _pipeline(text, voice=voice_id, speed=1.0, split_pattern=r'\n+'):
        if audio is not None and len(audio) > 0:
            audio_parts.append(audio)
    
    if not audio_parts:
        raise ValueError("Preview generation failed")
    
    full_audio = np.concatenate(audio_parts)
    
    # Save preview
    preview_path = TEMP_DIR / f"preview_{voice_id}.wav"
    sf.write(str(preview_path), full_audio, 24000)
    
    return str(preview_path)
