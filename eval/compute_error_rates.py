import argparse
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor
import math
import os
from pathlib import Path
from typing import NamedTuple
from typing import Sequence

from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REF_ROOT = REPO_ROOT / "data" / "prepared_data"
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)


class ScoringTask(NamedTuple):
    relative_path: Path
    prediction_path: Path
    ref_path: Path


class ScoringResult(NamedTuple):
    relative_path: Path
    cer_edits: int
    cer_ref_chars: int
    wer_edits: int
    wer_ref_words: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prediction_root",
        type=Path,
        help="Directory containing predicted .txt files.",
    )
    parser.add_argument(
        "--ref-root",
        type=Path,
        default=DEFAULT_REF_ROOT,
        help=f"Directory containing ground-truth .txt files. Default: {DEFAULT_REF_ROOT}",
    )
    parser.add_argument(
        "--per-file",
        action="store_true",
        help="Print one tab-separated metric row per matched prediction file.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel scoring workers. Default: {DEFAULT_WORKERS}",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def edit_distance(source: Sequence[str], target: Sequence[str]) -> int:
    if len(source) < len(target):
        source, target = target, source

    previous = list(range(len(target) + 1))
    for source_index, source_item in enumerate(source, start=1):
        current = [source_index]
        for target_index, target_item in enumerate(target, start=1):
            substitution_cost = 0 if source_item == target_item else 1
            current.append(
                min(
                    previous[target_index] + 1,
                    current[target_index - 1] + 1,
                    previous[target_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def error_rate(edits: int, ref_count: int) -> float:
    if ref_count == 0:
        return math.nan
    return edits / ref_count


def format_rate(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.6f} ({value * 100:.2f}%)"


def score_file(task: ScoringTask) -> ScoringResult:
    prediction_text = normalize_text(task.prediction_path.read_text(encoding="utf-8"))
    ref_text = normalize_text(task.ref_path.read_text(encoding="utf-8"))
    prediction_words = prediction_text.split()
    ref_words = ref_text.split()

    return ScoringResult(
        relative_path=task.relative_path,
        cer_edits=edit_distance(prediction_text, ref_text),
        cer_ref_chars=len(ref_text),
        wer_edits=edit_distance(prediction_words, ref_words),
        wer_ref_words=len(ref_words),
    )


def main() -> None:
    args = parse_args()
    prediction_root = args.prediction_root.expanduser().resolve()
    ref_root = args.ref_root.expanduser().resolve()
    if args.workers < 1:
        raise ValueError(f"--workers must be >= 1, got {args.workers}")

    if not prediction_root.is_dir():
        raise NotADirectoryError(f"Prediction directory not found: {prediction_root}")
    if not ref_root.is_dir():
        raise NotADirectoryError(f"Reference directory not found: {ref_root}")

    prediction_paths = sorted(prediction_root.rglob("*.txt"))
    tasks: list[ScoringTask] = []
    missing_refs: list[Path] = []
    cer_edits = 0
    cer_ref_chars = 0
    wer_edits = 0
    wer_ref_words = 0

    if args.per_file:
        print("path\tcer\twer\tchar_edits\tref_chars\tword_edits\tref_words")

    for prediction_path in prediction_paths:
        relative_path = prediction_path.relative_to(prediction_root)
        ref_path = ref_root / relative_path
        if not ref_path.is_file():
            missing_refs.append(relative_path)
            continue
        tasks.append(ScoringTask(relative_path, prediction_path, ref_path))

    executor_context = (
        nullcontext(None)
        if args.workers == 1
        else ProcessPoolExecutor(max_workers=args.workers)
    )
    with executor_context as executor:
        results = map(score_file, tasks) if executor is None else executor.map(score_file, tasks)
        for result in tqdm(results, total=len(tasks), desc="Scoring", unit="file"):
            cer_edits += result.cer_edits
            cer_ref_chars += result.cer_ref_chars
            wer_edits += result.wer_edits
            wer_ref_words += result.wer_ref_words

            if args.per_file:
                print(
                    f"{result.relative_path}\t"
                    f"{error_rate(result.cer_edits, result.cer_ref_chars):.6f}\t"
                    f"{error_rate(result.wer_edits, result.wer_ref_words):.6f}\t"
                    f"{result.cer_edits}\t{result.cer_ref_chars}\t"
                    f"{result.wer_edits}\t{result.wer_ref_words}"
                )

    if not tasks:
        raise RuntimeError(
            f"No prediction files matched references. "
            f"prediction_files={len(prediction_paths)} missing_refs={len(missing_refs)}"
        )

    print(f"Reference root: {ref_root}")
    print(f"Prediction root: {prediction_root}")
    print(f"Predicted text files: {len(prediction_paths)}")
    print(f"Matched files: {len(tasks)}")
    print(f"Missing references: {len(missing_refs)}")
    print(f"CER: {format_rate(error_rate(cer_edits, cer_ref_chars))} [{cer_edits}/{cer_ref_chars}]")
    print(f"WER: {format_rate(error_rate(wer_edits, wer_ref_words))} [{wer_edits}/{wer_ref_words}]")


if __name__ == "__main__":
    main()
