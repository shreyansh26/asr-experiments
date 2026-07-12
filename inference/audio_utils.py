from io import BytesIO
from math import sqrt
from pathlib import Path
import re
import struct
import wave


ASR_TEXT_START = "<asr_text>"
ASR_TEXT_END = "</asr_text>"


def audio_length_seconds(audio_path: Path) -> float:
    with wave.open(str(audio_path), "rb") as audio_file:
        return audio_file.getnframes() / audio_file.getframerate()


def audio_bytes_length_seconds(audio_bytes: bytes) -> float:
    with wave.open(BytesIO(audio_bytes), "rb") as audio_file:
        return audio_file.getnframes() / audio_file.getframerate()


def clipped_audio_bytes(audio_path: Path, duration_s: float) -> bytes:
    clipped_file = BytesIO()
    with wave.open(str(audio_path), "rb") as source:
        num_frames = int(duration_s * source.getframerate())
        frames = source.readframes(num_frames)
        with wave.open(clipped_file, "wb") as target:
            target.setparams(source.getparams())
            target.writeframes(frames)
    clipped_file.seek(0)
    return clipped_file.read()


def audio_rms(audio_bytes: bytes) -> int:
    with wave.open(BytesIO(audio_bytes), "rb") as audio_file:
        frames = audio_file.readframes(audio_file.getnframes())
        sample_width = audio_file.getsampwidth()
    if not frames:
        return 0
    if sample_width == 1:
        samples = [sample - 128 for sample in frames]
    elif sample_width == 2:
        sample_count = len(frames) // sample_width
        samples = struct.unpack(f"<{sample_count}h", frames[: sample_count * sample_width])
    elif sample_width == 4:
        sample_count = len(frames) // sample_width
        samples = struct.unpack(f"<{sample_count}i", frames[: sample_count * sample_width])
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")
    return int(sqrt(sum(sample * sample for sample in samples) / len(samples)))


def extract_stream_text(event: object) -> str:
    delta = getattr(event, "delta", None)
    if isinstance(delta, str):
        return delta

    text = getattr(event, "text", None)
    if isinstance(text, str):
        return text

    if not hasattr(event, "model_dump"):
        return ""

    data = event.model_dump()
    choices = data.get("choices") or []
    if not choices:
        return ""

    choice = choices[0]
    delta = choice.get("delta") or {}
    content = delta.get("content")
    if isinstance(content, str):
        return content

    text = choice.get("text")
    if isinstance(text, str):
        return text

    return ""


def clean_transcription_text(text: str) -> str:
    text = re.sub(r"\blanguage [^<]*<asr_text>", "", text)
    text = text.replace(ASR_TEXT_START, "")
    if ASR_TEXT_END in text:
        text = text.split(ASR_TEXT_END, 1)[0]
    return text
