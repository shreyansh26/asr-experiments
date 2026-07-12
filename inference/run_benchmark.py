import argparse
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = INFERENCE_DIR / "results"
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "prepared_data"
DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_BASE_URL = "http://localhost:8090/v1"
DEFAULT_NO_SPEECH_RMS_THRESHOLD = 1
DEFAULT_WORKERS = 1
DEFAULT_SEQUENTIAL_OUTPUT_ROOT = REPO_ROOT / "data" / "sequential_predicted"
DEFAULT_BATCHED_OUTPUT_ROOT = REPO_ROOT / "data" / "batched_predicted"

CSV_COLUMNS = [
    "timestamp_utc",
    "mode",
    "stream",
    "workers",
    "overwrite",
    "input_root",
    "output_root",
    "model",
    "base_url",
    "uniform_audio_length",
    "num_files",
    "no_speech_rms_threshold",
    "warmup_completed",
    "warmup_failed",
    "warmup_timed_out",
    "completed",
    "queued_for_write",
    "failed",
    "timed_out",
    "existing_not_written",
    "skipped_no_speech",
    "wall_time_s",
    "throughput_files_s",
    "audio_throughput_audio_s_s",
    "latency_avg_s",
    "latency_p50_s",
    "latency_p95_s",
    "latency_p99_s",
    "ttft_avg_s",
    "ttft_p50_s",
    "ttft_p95_s",
    "ttft_p99_s",
    "command",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("sequential", "batched"), required=True)
    parser.add_argument("--no-stream", dest="stream", action="store_false")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--uniform-audio-length", type=float, default=None)
    parser.add_argument("--num-files", type=int, default=None)
    parser.add_argument(
        "--no-speech-rms-threshold",
        type=int,
        default=DEFAULT_NO_SPEECH_RMS_THRESHOLD,
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.set_defaults(stream=True)
    return parser.parse_args()


def format_seconds(value: float) -> str:
    return f"{value:g}".replace(".", "_")


def default_output_root(mode: str, uniform_audio_length: float | None) -> Path:
    if mode == "sequential":
        base = DEFAULT_SEQUENTIAL_OUTPUT_ROOT
        prefix = "sequential"
    else:
        base = DEFAULT_BATCHED_OUTPUT_ROOT
        prefix = "batched"

    if uniform_audio_length is None:
        return base
    suffix = format_seconds(uniform_audio_length)
    return base.parent / f"{prefix}_predicted_uniform_audio_length_{suffix}s"


def build_command(args: argparse.Namespace, output_root: Path) -> list[str]:
    if args.mode == "sequential":
        if args.workers != DEFAULT_WORKERS:
            raise ValueError("--workers can only be used with --mode batched")
        script = INFERENCE_DIR / "run_infer.py"
    else:
        script = INFERENCE_DIR / "run_infer_batched.py"

    command = [
        sys.executable,
        "-u",
        str(script),
        "--input-root",
        str(args.input_root),
        "--output-root",
        str(output_root),
        "--model",
        args.model,
        "--base-url",
        args.base_url,
        "--no-speech-rms-threshold",
        str(args.no_speech_rms_threshold),
    ]

    if args.stream:
        command.append("--stream")
    if args.uniform_audio_length is not None:
        command.extend(["--uniform-audio-length", str(args.uniform_audio_length)])
    if args.num_files is not None:
        command.extend(["--num-files", str(args.num_files)])
    if args.overwrite:
        command.append("--overwrite")
    if args.mode == "batched":
        command.extend(["--workers", str(args.workers)])

    return command


def strip_seconds(value: str) -> str:
    return value[:-1] if value.endswith("s") and value != "n/a" else value


def parse_key_values(line: str) -> dict[str, str]:
    return {
        key: strip_seconds(value)
        for key, value in re.findall(r"([a-zA-Z0-9_]+)=([^ ]+)", line)
    }


def parse_metrics(output: str) -> dict[str, str]:
    row: dict[str, str] = {}
    for line in output.splitlines():
        if line.startswith("Done. "):
            row.update(parse_key_values(line))
        elif line.startswith("Inference metrics: "):
            metrics = parse_key_values(line)
            if "wall_time" in metrics:
                row["wall_time_s"] = metrics["wall_time"]
            if "throughput" in metrics:
                row["throughput_files_s"] = metrics["throughput"]
            if "audio_throughput" in metrics:
                row["audio_throughput_audio_s_s"] = metrics["audio_throughput"]
        elif line.startswith("Latency: "):
            metrics = parse_key_values(line)
            if "avg" in metrics:
                row["latency_avg_s"] = metrics["avg"]
            for percentile in ("p50", "p95", "p99"):
                if percentile in metrics:
                    row[f"latency_{percentile}_s"] = metrics[percentile]
        elif line.startswith("TTFT: "):
            metrics = parse_key_values(line)
            if "avg" in metrics:
                row["ttft_avg_s"] = metrics["avg"]
            for percentile in ("p50", "p95", "p99"):
                if percentile in metrics:
                    row[f"ttft_{percentile}_s"] = metrics[percentile]
    return row


def migrate_csv_header(csv_path: Path) -> None:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return

    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames == CSV_COLUMNS:
            return
        rows = list(reader)

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def append_csv(mode: str, row: dict[str, str]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"{mode}.csv"
    migrate_csv_header(csv_path)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})
    return csv_path


def run_command(command: list[str]) -> str:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        output_lines.append(line)

    return_code = process.wait()
    output = "".join(output_lines)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command, output=output)
    return output


def main() -> None:
    args = parse_args()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else default_output_root(args.mode, args.uniform_audio_length)
    )
    input_root = args.input_root.expanduser().resolve()
    try:
        command = build_command(args, output_root)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    output = run_command(command)
    row = {column: "" for column in CSV_COLUMNS}
    row.update(parse_metrics(output))
    row.update(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "stream": str(args.stream),
            "workers": str(args.workers) if args.mode == "batched" else "",
            "overwrite": str(args.overwrite),
            "input_root": str(input_root),
            "output_root": str(output_root),
            "model": args.model,
            "base_url": args.base_url,
            "uniform_audio_length": (
                "" if args.uniform_audio_length is None else str(args.uniform_audio_length)
            ),
            "num_files": "" if args.num_files is None else str(args.num_files),
            "no_speech_rms_threshold": str(args.no_speech_rms_threshold),
            "command": " ".join(command),
        }
    )
    csv_path = append_csv(args.mode, row)
    print(f"Updated metrics CSV: {csv_path}")


if __name__ == "__main__":
    main()
