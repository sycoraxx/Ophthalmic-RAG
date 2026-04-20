"""
src.speech — Fast ASR module for voice-based queries.
Uses faster-whisper (CTranslate2) with large-v3-turbo for low-latency,
Indian-accent-optimized speech recognition.
"""

from src.speech.speech_recognizer import SpeechRecognizer, TranscriptionResult

__all__ = ["SpeechRecognizer", "TranscriptionResult"]
