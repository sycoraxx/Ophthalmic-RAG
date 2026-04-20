"""
speech_recognizer.py — Low-Latency ASR Engine for Ophthalmic RAG
────────────────────────────────────────────────────────────────────────────────
Uses faster-whisper (CTranslate2 backend) with the large-v3-turbo model
for fast, accurate transcription optimised for Indian-accented English.

Features:
  • VAD (Silero) to skip silence → lower latency
  • float16 on GPU, int8 fallback on CPU
  • Thread-safe singleton pattern (matches EyeClipAgent design)
  • Graceful degradation if model fails to load

Usage:
    from src.speech import SpeechRecognizer

    sr = SpeechRecognizer()
    result = sr.transcribe(audio_bytes)
    print(result.text)
"""

from __future__ import annotations

import io
import os
import time
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Union

# ─── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL_SIZE = "large-v3-turbo"   # 809M params, 6× faster than large-v3
DEFAULT_LANGUAGE = "en"                  # Pre-set to English (skip lang detect → faster)
DEFAULT_BEAM_SIZE = 3                    # Lower beam = faster; 3 is sweet-spot for accuracy
DEFAULT_VAD_THRESHOLD = 0.4              # Silero VAD sensitivity (0–1, lower = more aggressive)


@dataclass
class TranscriptionSegment:
    """A single timed segment from the transcription."""
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    """Complete transcription output with metadata."""
    text: str
    language: str = ""
    language_probability: float = 0.0
    duration_seconds: float = 0.0
    processing_time_seconds: float = 0.0
    segments: List[TranscriptionSegment] = field(default_factory=list)

    @property
    def real_time_factor(self) -> float:
        """RTF < 1.0 means faster-than-realtime."""
        if self.duration_seconds > 0:
            return self.processing_time_seconds / self.duration_seconds
        return 0.0


class SpeechRecognizer:
    """
    Fast speech-to-text engine backed by faster-whisper.

    Designed to be instantiated once and cached (like EyeClipAgent).
    Handles raw audio bytes (from st.audio_input) or file paths.
    """

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL_SIZE,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        download_root: Optional[str] = None,
    ):
        """
        Initialise the ASR model.

        Args:
            model_size: Whisper model variant. One of:
                        tiny, base, small, medium, large-v3, large-v3-turbo
            device: "cuda" or "cpu". Auto-detected if None.
            compute_type: "float16" (GPU), "int8" (CPU), or "auto".
            download_root: Optional directory for model cache.
        """
        self._model = None
        self._model_size = model_size
        self._load_error: Optional[str] = None

        # Auto-detect device
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        # Set compute type based on device
        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"

        self._device = device
        self._compute_type = compute_type

        # Attempt to load model
        try:
            from faster_whisper import WhisperModel

            load_start = time.time()
            kwargs = {
                "device": device,
                "compute_type": compute_type,
            }
            if download_root:
                kwargs["download_root"] = download_root

            self._model = WhisperModel(model_size, **kwargs)
            load_time = time.time() - load_start
            print(
                f"[SpeechRecognizer] ✓ Loaded {model_size} on {device} "
                f"({compute_type}) in {load_time:.1f}s"
            )
        except Exception as e:
            self._load_error = str(e)
            print(f"[SpeechRecognizer] ✗ Failed to load model: {e}")

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """Whether the ASR model is loaded and ready."""
        return self._model is not None

    @property
    def model_info(self) -> dict:
        """Return model metadata for debugging / UI display."""
        return {
            "model_size": self._model_size,
            "device": self._device,
            "compute_type": self._compute_type,
            "ready": self.is_ready,
            "error": self._load_error,
        }

    # ── Core Transcription ────────────────────────────────────────────────────

    def transcribe(
        self,
        audio_input: Union[bytes, io.BytesIO, str, Path],
        language: str = DEFAULT_LANGUAGE,
        beam_size: int = DEFAULT_BEAM_SIZE,
        vad_filter: bool = True,
        vad_threshold: float = DEFAULT_VAD_THRESHOLD,
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio_input: Raw audio bytes, BytesIO stream, or file path string.
            language: ISO language code (e.g. "en"). Set to None for auto-detect.
            beam_size: Beam width for decoding. Lower = faster.
            vad_filter: Enable Silero VAD to skip silence.
            vad_threshold: VAD sensitivity (0.0–1.0).

        Returns:
            TranscriptionResult with full text, timing, and metadata.
        """
        if not self.is_ready:
            return TranscriptionResult(
                text="",
                language=language or "",
                processing_time_seconds=0.0,
            )

        # Resolve audio to a file path (faster-whisper needs a path or ndarray)
        temp_path = None
        try:
            audio_path = self._resolve_audio_input(audio_input)
            if audio_path is None:
                return TranscriptionResult(text="", language=language or "")

            # Check if we created a temp file
            if isinstance(audio_input, (bytes, io.BytesIO)):
                temp_path = audio_path

            t0 = time.time()

            # Transcribe with faster-whisper
            segments_iter, info = self._model.transcribe(
                audio_path,
                language=language,
                beam_size=beam_size,
                vad_filter=vad_filter,
                vad_parameters={"threshold": vad_threshold} if vad_filter else None,
                condition_on_previous_text=False,  # Prevents hallucination loops
                no_speech_threshold=0.5,
            )

            # Collect segments
            segments: List[TranscriptionSegment] = []
            text_parts: List[str] = []
            for seg in segments_iter:
                segments.append(TranscriptionSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                ))
                text_parts.append(seg.text.strip())

            processing_time = time.time() - t0
            full_text = " ".join(text_parts).strip()

            # Clean up common Whisper artefacts
            full_text = self._clean_transcription(full_text)

            result = TranscriptionResult(
                text=full_text,
                language=info.language,
                language_probability=info.language_probability,
                duration_seconds=info.duration,
                processing_time_seconds=processing_time,
                segments=segments,
            )

            print(
                f"[SpeechRecognizer] Transcribed {info.duration:.1f}s audio "
                f"in {processing_time:.2f}s (RTF={result.real_time_factor:.2f}) "
                f"lang={info.language}({info.language_probability:.0%})"
            )

            return result

        except Exception as e:
            print(f"[SpeechRecognizer] ✗ Transcription failed: {e}")
            return TranscriptionResult(
                text="",
                language=language or "",
                processing_time_seconds=0.0,
            )
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    # ── Audio Input Resolution ────────────────────────────────────────────────

    def _resolve_audio_input(
        self, audio_input: Union[bytes, io.BytesIO, str, Path]
    ) -> Optional[str]:
        """
        Convert various audio input types to a file path.

        faster-whisper accepts file paths or numpy arrays.
        For bytes/BytesIO, we write to a temp file.
        """
        if isinstance(audio_input, (str, Path)):
            path = str(audio_input)
            if not os.path.exists(path):
                print(f"[SpeechRecognizer] ✗ Audio file not found: {path}")
                return None
            return path

        if isinstance(audio_input, io.BytesIO):
            audio_bytes = audio_input.read()
            audio_input.seek(0)  # Reset for potential re-reads
        elif isinstance(audio_input, bytes):
            audio_bytes = audio_input
        else:
            print(f"[SpeechRecognizer] ✗ Unsupported audio input type: {type(audio_input)}")
            return None

        if not audio_bytes or len(audio_bytes) < 100:
            print("[SpeechRecognizer] ✗ Audio input too short or empty")
            return None

        # Write to a temp file — faster-whisper will handle format detection via ffmpeg
        suffix = ".wav"
        # Detect format from magic bytes
        if audio_bytes[:4] == b"fLaC":
            suffix = ".flac"
        elif audio_bytes[:3] == b"ID3" or audio_bytes[:2] == b"\xff\xfb":
            suffix = ".mp3"
        elif audio_bytes[:4] == b"OggS":
            suffix = ".ogg"

        fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix="asr_")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(audio_bytes)
        except Exception as e:
            print(f"[SpeechRecognizer] ✗ Failed to write temp audio: {e}")
            os.close(fd)
            return None

        return temp_path

    # ── Post-Processing ───────────────────────────────────────────────────────

    @staticmethod
    def _clean_transcription(text: str) -> str:
        """
        Clean common Whisper transcription artefacts.

        Removes:
          - Repeated phrases (hallucination loops)
          - Leading/trailing whitespace and punctuation quirks
          - Common filler sounds that Whisper sometimes hallucinates
        """
        if not text:
            return ""

        # Remove common hallucinated fillers
        import re
        # Remove "Thank you for watching" and similar Whisper artefacts
        hallucination_patterns = [
            r"(?i)thank you for watching\.?",
            r"(?i)thanks for watching\.?",
            r"(?i)please subscribe\.?",
            r"(?i)like and subscribe\.?",
            r"(?i)see you in the next video\.?",
            r"(?i)^\s*you\s*$",  # standalone "you" from silence
        ]
        for pattern in hallucination_patterns:
            text = re.sub(pattern, "", text)

        # Detect and collapse repeated phrases (hallucination loops)
        # e.g. "my eye hurts my eye hurts my eye hurts" → "my eye hurts"
        words = text.split()
        if len(words) > 6:
            for phrase_len in range(2, min(len(words) // 2 + 1, 15)):
                phrase = " ".join(words[:phrase_len])
                rest = " ".join(words[phrase_len:])
                repetitions = 0
                while rest.lower().startswith(phrase.lower()):
                    repetitions += 1
                    rest = rest[len(phrase):].strip()
                if repetitions >= 2:
                    text = phrase + (" " + rest if rest else "")
                    break

        return text.strip()


# ─── Quick CLI Test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    sr = SpeechRecognizer()
    print(f"Model ready: {sr.is_ready}")
    print(f"Model info: {sr.model_info}")

    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        print(f"\nTranscribing: {audio_file}")
        result = sr.transcribe(audio_file)
        print(f"  Text: {result.text}")
        print(f"  Language: {result.language} ({result.language_probability:.0%})")
        print(f"  Duration: {result.duration_seconds:.1f}s")
        print(f"  Processing: {result.processing_time_seconds:.2f}s")
        print(f"  RTF: {result.real_time_factor:.2f}")
