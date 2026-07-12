import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from math import ceil
from pathlib import Path
from queue import Queue
from threading import Thread
from time import perf_counter
from typing import NamedTuple

from audio_utils import audio_bytes_length_seconds
from audio_utils import audio_length_seconds
from audio_utils import audio_rms
from audio_utils import clean_transcription_text
from audio_utils import clipped_audio_bytes
from audio_utils import extract_stream_text
from openai import APITimeoutError, OpenAI


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "prepared_data"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "predicted"
STOP_WRITER = object()
WARMUP_AUDIO_COUNT = 20


class WriteJob(NamedTuple):
    output_path: Path
    text: str


class InferenceJob(NamedTuple):
    audio_path: Path
    output_path: Path


class PreparedInferenceJob(NamedTuple):
    audio_path: Path
    output_path: Path
    audio_bytes: bytes
    audio_seconds: float


class InferenceResult(NamedTuple):
    status: str
    latency_s: float | None = None
    ttft_s: float | None = None
    audio_seconds: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--base-url", default="http://localhost:8090/v1")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--num-files", type=int, default=None)
    parser.add_argument("--uniform-audio-length", type=float, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--no-speech-rms-threshold", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def format_seconds(value: float) -> str:
    return f"{value:g}".replace(".", "_")


def prediction_path(audio_path: Path, input_root: Path, output_root: Path) -> Path:
    relative_path = audio_path.relative_to(input_root)
    return output_root / relative_path.with_suffix(".txt")


def prepare_inference_job(
    job: InferenceJob,
    uniform_audio_length: float | None,
    no_speech_rms_threshold: int,
) -> PreparedInferenceJob | None:
    if uniform_audio_length is None:
        audio_bytes = job.audio_path.read_bytes()
    else:
        audio_bytes = clipped_audio_bytes(job.audio_path, uniform_audio_length)
        if audio_rms(audio_bytes) <= no_speech_rms_threshold:
            return None
    return PreparedInferenceJob(
        audio_path=job.audio_path,
        output_path=job.output_path,
        audio_bytes=audio_bytes,
        audio_seconds=audio_bytes_length_seconds(audio_bytes),
    )


def transcribe_audio(
    job: PreparedInferenceJob,
    *,
    model: str,
    base_url: str,
    stream: bool,
    timeout_seconds: float,
    max_tokens: int,
    write_queue: Queue,
    write_output: bool = True,
) -> InferenceResult:
    client = OpenAI(base_url=base_url, api_key="EMPTY")

    with BytesIO(job.audio_bytes) as audio_file:
        audio_file.name = job.audio_path.name
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
            if write_output:
                write_queue.put(WriteJob(job.output_path, clean_transcription_text(text)))
            return InferenceResult(
                status="queued" if write_output else "done",
                latency_s=latency_s,
                ttft_s=ttft_s,
                audio_seconds=job.audio_seconds,
            )

        start_time = perf_counter()
        transcription = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            extra_body={"max_completion_tokens": max_tokens},
            timeout=timeout_seconds,
        )
        latency_s = perf_counter() - start_time

    if write_output:
        write_queue.put(WriteJob(job.output_path, transcription.text))
    usage = getattr(transcription, "usage", None)
    audio_seconds = getattr(usage, "seconds", None)
    return InferenceResult(
        status="queued" if write_output else "done",
        latency_s=latency_s,
        audio_seconds=audio_seconds or job.audio_seconds,
    )


def write_predictions(write_queue: Queue, errors: list[tuple[Path | None, Exception]]) -> None:
    while True:
        job = write_queue.get()
        try:
            if job is STOP_WRITER:
                return
            job.output_path.parent.mkdir(parents=True, exist_ok=True)
            job.output_path.write_text(job.text.strip() + "\n", encoding="utf-8")
        except Exception as exc:
            path = job.output_path if isinstance(job, WriteJob) else None
            errors.append((path, exc))
        finally:
            write_queue.task_done()


def percentile(values: list[float], percentile_value: int) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = ceil((percentile_value / 100) * len(sorted_values)) - 1
    return sorted_values[max(0, min(index, len(sorted_values) - 1))]


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
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
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    if args.output_root is not None:
        output_root = args.output_root.expanduser().resolve()
    elif args.uniform_audio_length is None:
        output_root = DEFAULT_OUTPUT_ROOT
    else:
        suffix = format_seconds(args.uniform_audio_length)
        output_root = DEFAULT_OUTPUT_ROOT.parent / f"predicted_uniform_audio_length_{suffix}s"

    all_audio_paths = sorted(input_root.rglob("*.wav"))
    if args.uniform_audio_length is not None:
        eligible_audio_paths = [
            path
            for path in all_audio_paths
            if audio_length_seconds(path) > args.uniform_audio_length
        ]
        print(
            f"Uniform audio length: {args.uniform_audio_length:g}s "
            f"({len(eligible_audio_paths)}/{len(all_audio_paths)} files longer than this)"
        )
    else:
        eligible_audio_paths = all_audio_paths

    skipped_existing = 0
    skipped_no_speech = 0
    prepared_warmup_jobs: list[PreparedInferenceJob] = []
    prepared_jobs: list[PreparedInferenceJob] = []

    print(f"Found {len(all_audio_paths)} wav files under {input_root}")
    if args.num_files is None:
        print(
            f"Preparing usable audio payloads from {len(eligible_audio_paths)} "
            f"eligible files (warmup={WARMUP_AUDIO_COUNT} measured=all remaining)"
        )
    else:
        print(
            f"Preparing {WARMUP_AUDIO_COUNT + args.num_files} usable audio payloads "
            f"from {len(eligible_audio_paths)} eligible files "
            f"(warmup={WARMUP_AUDIO_COUNT} measured={args.num_files})"
        )
    print(f"Writing predictions under {output_root}")

    prepare_start_time = perf_counter()
    for audio_path in eligible_audio_paths:
        output_path = prediction_path(audio_path, input_root, output_root)
        if (
            len(prepared_warmup_jobs) >= WARMUP_AUDIO_COUNT
            and output_path.exists()
            and not args.overwrite
        ):
            skipped_existing += 1
            continue
        prepared = prepare_inference_job(
            InferenceJob(audio_path, output_path),
            args.uniform_audio_length,
            args.no_speech_rms_threshold,
        )
        if prepared is None:
            skipped_no_speech += 1
            continue
        if len(prepared_warmup_jobs) < WARMUP_AUDIO_COUNT:
            prepared_warmup_jobs.append(prepared)
            continue
        prepared_jobs.append(prepared)
        if args.num_files is not None and len(prepared_jobs) >= args.num_files:
            break

    prepare_wall_s = perf_counter() - prepare_start_time
    if len(prepared_warmup_jobs) < WARMUP_AUDIO_COUNT:
        raise ValueError(
            f"Need {WARMUP_AUDIO_COUNT} usable warmup audio payloads, "
            f"found {len(prepared_warmup_jobs)} after filtering"
        )
    if args.num_files is not None and len(prepared_jobs) < args.num_files:
        raise ValueError(
            f"Need {args.num_files} usable benchmark audio payloads after "
            f"{WARMUP_AUDIO_COUNT} warmup payloads, found {len(prepared_jobs)}"
        )

    print(
        f"Prepared {len(prepared_warmup_jobs) + len(prepared_jobs)} audio payloads "
        f"(warmup={len(prepared_warmup_jobs)} measured={len(prepared_jobs)}) "
        f"in {prepare_wall_s:.3f}s before vLLM submission"
    )
    if skipped_existing:
        print(f"Skipping {skipped_existing} files with existing predictions")
    if skipped_no_speech:
        print(
            "Skipped "
            f"{skipped_no_speech} clipped no-speech payloads "
            f"during preparation (rms <= {args.no_speech_rms_threshold})"
        )

    total = len(prepared_jobs)
    print(
        f"Submitting {total} measured files to vLLM "
        f"(timeout={args.timeout_seconds:g}s max_tokens={args.max_tokens})"
    )

    warmup_completed = 0
    warmup_failed = 0
    warmup_timed_out = 0
    queued = 0
    failed = 0
    timed_out = 0
    latencies_s: list[float] = []
    ttfts_s: list[float] = []
    audio_seconds_total = 0.0
    write_queue: Queue = Queue()
    writer_errors: list[tuple[Path | None, Exception]] = []
    writer_thread = Thread(
        target=write_predictions,
        args=(write_queue, writer_errors),
        daemon=True,
    )
    writer_thread.start()

    try:
        if prepared_warmup_jobs:
            print(f"Warming up with {len(prepared_warmup_jobs)} audio files")
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                warmup_futures = {
                    executor.submit(
                        transcribe_audio,
                        job,
                        model=args.model,
                        base_url=args.base_url,
                        stream=args.stream,
                        timeout_seconds=args.timeout_seconds,
                        max_tokens=args.max_tokens,
                        write_queue=write_queue,
                        write_output=False,
                    ): job
                    for job in prepared_warmup_jobs
                }

                for index, future in enumerate(as_completed(warmup_futures), start=1):
                    job = warmup_futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        warmup_failed += 1
                        if isinstance(exc, APITimeoutError):
                            warmup_timed_out += 1
                        print(
                            f"[warmup {index}/{len(prepared_warmup_jobs)}] failed: "
                            f"{job.audio_path.relative_to(input_root)} "
                            f"error={type(exc).__name__}: {exc}"
                        )
                        continue
                    warmup_completed += 1
                    print(
                        f"[warmup {index}/{len(prepared_warmup_jobs)}] "
                        f"{result.status}: {job.audio_path.relative_to(input_root)}"
                    )

        inference_start_time = perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    transcribe_audio,
                    job,
                    model=args.model,
                    base_url=args.base_url,
                    stream=args.stream,
                    timeout_seconds=args.timeout_seconds,
                    max_tokens=args.max_tokens,
                    write_queue=write_queue,
                ): job
                for job in prepared_jobs
            }

            for index, future in enumerate(as_completed(futures), start=1):
                job = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    failed += 1
                    if isinstance(exc, APITimeoutError):
                        timed_out += 1
                    print(
                        f"[{index}/{total}] failed: "
                        f"{job.audio_path.relative_to(input_root)} "
                        f"error={type(exc).__name__}: {exc}"
                    )
                    continue
                if result.status == "queued":
                    queued += 1
                    if result.latency_s is not None:
                        latencies_s.append(result.latency_s)
                    if result.ttft_s is not None:
                        ttfts_s.append(result.ttft_s)
                    if result.audio_seconds is not None:
                        audio_seconds_total += result.audio_seconds
                    labels = []
                    if result.latency_s is not None:
                        labels.append(f"latency={result.latency_s:.3f}s")
                    if result.ttft_s is not None:
                        labels.append(f"ttft={result.ttft_s:.3f}s")
                    latency_label = f" {' '.join(labels)}" if labels else ""
                else:
                    latency_label = ""
                print(
                    f"[{index}/{total}] {result.status}: "
                    f"{job.audio_path.relative_to(input_root)}{latency_label}"
                )
        inference_wall_s = perf_counter() - inference_start_time

        write_queue.join()
    finally:
        write_queue.put(STOP_WRITER)
        writer_thread.join()

    if writer_errors:
        path, exc = writer_errors[0]
        raise RuntimeError(f"Failed to write prediction {path}: {exc}") from exc

    print(
        "Done. "
        f"warmup_completed={warmup_completed} "
        f"warmup_failed={warmup_failed} "
        f"warmup_timed_out={warmup_timed_out} "
        f"queued_for_write={queued} "
        f"skipped_existing={skipped_existing} "
        f"skipped_no_speech={skipped_no_speech} "
        f"failed={failed} "
        f"timed_out={timed_out}"
    )
    if queued:
        files_per_second = queued / inference_wall_s if inference_wall_s else 0.0
        audio_seconds_per_second = (
            audio_seconds_total / inference_wall_s if inference_wall_s else 0.0
        )
        print(
            "Inference metrics: "
            f"wall_time={inference_wall_s:.3f}s "
            f"throughput={files_per_second:.3f} files/s "
            f"audio_throughput={audio_seconds_per_second:.3f} audio_s/s"
        )
        print(
            "Latency: "
            f"p50={percentile(latencies_s, 50):.3f}s "
            f"p90={percentile(latencies_s, 90):.3f}s "
            f"p99={percentile(latencies_s, 99):.3f}s"
        )
        if ttfts_s:
            print(
                "TTFT: "
                f"p50={percentile(ttfts_s, 50):.3f}s "
                f"p90={percentile(ttfts_s, 90):.3f}s "
                f"p99={percentile(ttfts_s, 99):.3f}s"
            )
        else:
            print("TTFT: p50=n/a p90=n/a p99=n/a")


if __name__ == "__main__":
    main()
