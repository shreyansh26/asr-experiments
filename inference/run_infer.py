import argparse
from io import BytesIO
from math import sqrt
from pathlib import Path
import re
import struct
from time import perf_counter
from typing import NamedTuple
import wave

from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "prepared_data"
ASR_TEXT_START = "<asr_text>"
ASR_TEXT_END = "</asr_text>"


class PreparedAudio(NamedTuple):
    audio_path: Path
    audio_bytes: bytes
    audio_seconds: float


class InferenceResult(NamedTuple):
    text: str
    latency_s: float
    ttft_s: float | None = None
    audio_seconds: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", nargs="?", help="Path to a local audio file")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--base-url", default="http://localhost:8090/v1")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--num-files", type=int, default=None)
    parser.add_argument("--uniform-audio-length", type=float, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--no-speech-rms-threshold", type=int, default=1)
    parser.add_argument("--print-text", dest="print_text", action="store_true")
    parser.add_argument("--no-print-text", dest="print_text", action="store_false")
    parser.set_defaults(print_text=None)
    return parser.parse_args()


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


def prepare_audio(
    audio_path: Path,
    uniform_audio_length: float | None,
    no_speech_rms_threshold: int,
) -> PreparedAudio | None:
    if uniform_audio_length is None:
        audio_bytes = audio_path.read_bytes()
    else:
        audio_bytes = clipped_audio_bytes(audio_path, uniform_audio_length)
        if audio_rms(audio_bytes) <= no_speech_rms_threshold:
            return None
    return PreparedAudio(
        audio_path=audio_path,
        audio_bytes=audio_bytes,
        audio_seconds=audio_bytes_length_seconds(audio_bytes),
    )


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


def transcribe_audio(
    client: OpenAI,
    prepared_audio: PreparedAudio,
    *,
    model: str,
    stream: bool,
    timeout_seconds: float,
    max_tokens: int,
) -> InferenceResult:
    with BytesIO(prepared_audio.audio_bytes) as audio_file:
        audio_file.name = prepared_audio.audio_path.name
        if stream:
            stream_start = perf_counter()
            stream_response = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                stream=True,
                extra_body={"max_completion_tokens": max_tokens},
                timeout=timeout_seconds,
            )

            ttft_s = None
            text_parts = []
            final_text = None
            for event in stream_response:
                delta = extract_stream_text(event)
                if delta:
                    text_parts.append(delta)
                    if ttft_s is None and clean_transcription_text("".join(text_parts)):
                        ttft_s = perf_counter() - stream_start

                event_type = getattr(event, "type", None)
                if event_type == "transcript.text.done":
                    event_text = getattr(event, "text", None)
                    if event_text:
                        final_text = event_text

            latency_s = perf_counter() - stream_start
            text = final_text if final_text is not None else "".join(text_parts)
            return InferenceResult(
                text=clean_transcription_text(text),
                latency_s=latency_s,
                ttft_s=ttft_s,
                audio_seconds=prepared_audio.audio_seconds,
            )

        start_time = perf_counter()
        transcription = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            extra_body={"max_completion_tokens": max_tokens},
            timeout=timeout_seconds,
        )
        latency_s = perf_counter() - start_time

    usage = getattr(transcription, "usage", None)
    return InferenceResult(
        text=transcription.text,
        latency_s=latency_s,
        audio_seconds=getattr(usage, "seconds", None) or prepared_audio.audio_seconds,
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    args = parse_args()
    if args.num_files is not None and args.num_files < 0:
        raise ValueError("--num-files must be >= 0")
    if args.uniform_audio_length is not None and args.uniform_audio_length <= 0:
        raise ValueError("--uniform-audio-length must be > 0")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be > 0")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be >= 1")
    if args.no_speech_rms_threshold < 0:
        raise ValueError("--no-speech-rms-threshold must be >= 0")

    if args.audio_path is None:
        input_root = args.input_root.expanduser().resolve()
        if not input_root.is_dir():
            raise FileNotFoundError(f"Input root not found: {input_root}")
        audio_paths = sorted(input_root.rglob("*.wav"))
        display_path = lambda path: path.relative_to(input_root)
    else:
        audio_path = Path(args.audio_path).expanduser().resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio_paths = [audio_path]
        display_path = lambda path: path

    all_audio_count = len(audio_paths)
    if args.uniform_audio_length is not None:
        audio_paths = [
            path
            for path in audio_paths
            if audio_length_seconds(path) > args.uniform_audio_length
        ]
        print(
            f"Uniform audio length: {args.uniform_audio_length:g}s "
            f"({len(audio_paths)}/{all_audio_count} files longer than this)"
        )

    if args.num_files is not None:
        audio_paths = audio_paths[: args.num_files]

    print_text = args.print_text
    if print_text is None:
        print_text = args.audio_path is not None

    print(f"Preparing {len(audio_paths)} audio files")
    prepared_results = [
        prepare_audio(
            path,
            args.uniform_audio_length,
            args.no_speech_rms_threshold,
        )
        for path in audio_paths
    ]
    prepared_audio = [audio for audio in prepared_results if audio is not None]
    skipped_no_speech = len(prepared_results) - len(prepared_audio)
    if skipped_no_speech:
        print(
            "Skipped "
            f"{skipped_no_speech} clipped no-speech payloads "
            f"(rms <= {args.no_speech_rms_threshold})"
        )

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    latencies_s: list[float] = []
    ttfts_s: list[float] = []
    audio_seconds_total = 0.0
    completed = 0
    failed = 0
    total = len(prepared_audio)

    inference_start_time = perf_counter()
    for index, audio in enumerate(prepared_audio, start=1):
        try:
            result = transcribe_audio(
                client,
                audio,
                model=args.model,
                stream=args.stream,
                timeout_seconds=args.timeout_seconds,
                max_tokens=args.max_tokens,
            )
        except Exception as exc:
            failed += 1
            print(
                f"[{index}/{total}] failed: {display_path(audio.audio_path)} "
                f"error={type(exc).__name__}: {exc}"
            )
            continue

        completed += 1
        latencies_s.append(result.latency_s)
        if result.ttft_s is not None:
            ttfts_s.append(result.ttft_s)
        if result.audio_seconds is not None:
            audio_seconds_total += result.audio_seconds

        labels = [f"latency={result.latency_s:.3f}s"]
        if result.ttft_s is not None:
            labels.append(f"ttft={result.ttft_s:.3f}s")
        if result.audio_seconds is not None:
            labels.append(f"audio_seconds={result.audio_seconds:g}")
        print(
            f"[{index}/{total}] done: "
            f"{display_path(audio.audio_path)} {' '.join(labels)}"
        )
        if print_text:
            print(result.text)

    inference_wall_s = perf_counter() - inference_start_time
    print(
        "Done. "
        f"completed={completed} "
        f"failed={failed} "
        f"skipped_no_speech={skipped_no_speech}"
    )
    if completed:
        files_per_second = completed / inference_wall_s if inference_wall_s else 0.0
        audio_seconds_per_second = (
            audio_seconds_total / inference_wall_s if inference_wall_s else 0.0
        )
        print(
            "Inference metrics: "
            f"wall_time={inference_wall_s:.3f}s "
            f"throughput={files_per_second:.3f} files/s "
            f"audio_throughput={audio_seconds_per_second:.3f} audio_s/s"
        )
        print(f"Latency: avg={mean(latencies_s):.3f}s")
        if ttfts_s:
            print(f"TTFT: avg={mean(ttfts_s):.3f}s")
        else:
            print("TTFT: avg=n/a")


if __name__ == "__main__":
    main()
