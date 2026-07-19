import argparse
import json
import math
import random
from importlib import metadata
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "prepared_data"
DEFAULT_OUTPUT = REPO_ROOT / "inference" / "results" / "fp8_static_scales.json"
FP8_E4M3_MAX = 448.0
FP8_MIN_SCALE = 1.0 / (FP8_E4M3_MAX * 512.0)
TARGET_SAMPLE_RATE = 16_000
REQUIRED_TRANSFORMERS_VERSION = "4.57.6"


class PerTensorScaleRecorder:
    def __init__(self, model: torch.nn.Module) -> None:
        self._stats: dict[str, dict[str, object]] = {}
        self._handles = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and not name.endswith("lm_head"):
                self._handles.append(
                    module.register_forward_pre_hook(self._make_hook(name))
                )

    def _make_hook(self, name: str):
        def record(_module: torch.nn.Module, args: tuple[object, ...]) -> None:
            if not args or not isinstance(args[0], torch.Tensor):
                return
            activation = args[0]
            if not activation.is_floating_point():
                return

            observed_absmax = activation.detach().abs().amax().float()
            shape = tuple(activation.shape)
            stats = self._stats.get(name)
            if stats is None:
                self._stats[name] = {
                    "absmax": observed_absmax,
                    "calls": 1,
                    "num_values": activation.numel(),
                    "shapes": {shape},
                }
                return

            stats["absmax"] = torch.maximum(stats["absmax"], observed_absmax)
            stats["calls"] += 1
            stats["num_values"] += activation.numel()
            stats["shapes"].add(shape)

        return record

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def export(self, scale_margin: float) -> dict[str, dict[str, object]]:
        exported = {}
        for name, stats in sorted(self._stats.items()):
            absmax = float(stats["absmax"].item())
            if not math.isfinite(absmax):
                raise ValueError(f"Non-finite activation maximum observed for {name}")
            scale = max(absmax * scale_margin / FP8_E4M3_MAX, FP8_MIN_SCALE)
            exported[name] = {
                "absmax": absmax,
                "scale": scale,
                "calls": stats["calls"],
                "num_values": stats["num_values"],
                "input_shapes": [list(shape) for shape in sorted(stats["shapes"])],
            }
        return exported


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record one calibration absmax and FP8 E4M3 activation scale per "
            "Qwen3-ASR Linear input."
        )
    )
    parser.add_argument("audio_paths", nargs="*", type=Path)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-files", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-audio-seconds", type=float, default=50.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scale-margin",
        type=float,
        default=1.05,
        help="Multiply each observed absmax by this headroom factor.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_files < 1:
        raise ValueError("--num-files must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.max_audio_seconds <= 0:
        raise ValueError("--max-audio-seconds must be > 0")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be >= 1")
    if args.scale_margin < 1.0:
        raise ValueError("--scale-margin must be >= 1.0")


def select_audio_paths(args: argparse.Namespace) -> list[Path]:
    if args.audio_paths:
        paths = [path.expanduser().resolve() for path in args.audio_paths]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Audio file not found: {missing[0]}")
    else:
        input_root = args.input_root.expanduser().resolve()
        if not input_root.is_dir():
            raise FileNotFoundError(f"Input root not found: {input_root}")
        paths = sorted(input_root.rglob("*.wav"))

    random.Random(args.seed).shuffle(paths)
    selected = paths[: args.num_files]
    if not selected:
        raise ValueError("No WAV files selected for calibration")
    if args.batch_size > 1:
        selected.sort(
            key=lambda path: min(sf.info(path).duration, args.max_audio_seconds)
        )
    return selected


def load_audio(path: Path, max_audio_seconds: float) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    max_samples = round(max_audio_seconds * sample_rate)
    audio = audio[:max_samples]
    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    if sample_rate != TARGET_SAMPLE_RATE:
        divisor = gcd(sample_rate, TARGET_SAMPLE_RATE)
        audio = resample_poly(
            audio,
            TARGET_SAMPLE_RATE // divisor,
            sample_rate // divisor,
        ).astype(np.float32, copy=False)
    return audio


def vllm_module_name(name: str) -> str:
    prefix, separator, suffix = name.rpartition(".")
    fused_suffixes = {
        "q_proj": "qkv_proj",
        "k_proj": "qkv_proj",
        "v_proj": "qkv_proj",
        "gate_proj": "gate_up_proj",
        "up_proj": "gate_up_proj",
    }
    suffix = fused_suffixes.get(suffix, suffix)
    return f"{prefix}{separator}{suffix}"


def fuse_vllm_scales(
    module_scales: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    fused = {}
    for name, stats in module_scales.items():
        fused_name = vllm_module_name(name)
        existing = fused.get(fused_name)
        if existing is None:
            fused[fused_name] = {
                "absmax": stats["absmax"],
                "scale": stats["scale"],
                "source_modules": [name],
            }
            continue
        existing["absmax"] = max(existing["absmax"], stats["absmax"])
        existing["scale"] = max(existing["scale"], stats["scale"])
        existing["source_modules"].append(name)
    return dict(sorted(fused.items()))


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.manual_seed(args.seed)

    transformers_version = metadata.version("transformers")
    try:
        qwen_asr_version = metadata.version("qwen-asr")
    except metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "qwen-asr is required for the checkpoint's thinker.* model layout. "
            "Run this script through the provided isolated uv command."
        ) from error
    if transformers_version != REQUIRED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"Qwen/Qwen3-ASR-1.7B uses the Transformers "
            f"{REQUIRED_TRANSFORMERS_VERSION} thinker.* checkpoint layout, but "
            f"this environment has Transformers {transformers_version}. Run this "
            "calibration script in a uv overlay with the required version."
        )

    audio_paths = select_audio_paths(args)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Calibration currently requires a CUDA device")

    from qwen_asr import Qwen3ASRModel

    print(f"Loading BF16 calibration model on {device}")
    asr_model = Qwen3ASRModel.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation=args.attn_implementation,
        max_inference_batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    model = asr_model.model
    model.eval()

    recorder = PerTensorScaleRecorder(model)
    completed_paths = []
    try:
        for batch_start in range(0, len(audio_paths), args.batch_size):
            batch_paths = audio_paths[batch_start : batch_start + args.batch_size]
            batch_audio = [
                load_audio(path, args.max_audio_seconds) for path in batch_paths
            ]
            print(
                f"Running batch {batch_start // args.batch_size + 1} "
                f"size={len(batch_paths)}"
            )
            with torch.inference_mode():
                transcriptions = asr_model.transcribe(
                    audio=[
                        (audio, TARGET_SAMPLE_RATE) for audio in batch_audio
                    ],
                    language=None,
                )
            if len(transcriptions) != len(batch_paths):
                raise RuntimeError(
                    f"Expected {len(batch_paths)} transcriptions, "
                    f"received {len(transcriptions)}"
                )
            for offset, (audio_path, audio, transcription) in enumerate(
                zip(batch_paths, batch_audio, transcriptions),
                start=1,
            ):
                completed_paths.append(audio_path)
                index = batch_start + offset
                print(
                    f"[{index}/{len(audio_paths)}] {display_path(audio_path)} "
                    f"samples={audio.size} output_chars={len(transcription.text)}"
                )
    finally:
        recorder.close()

    module_scales = recorder.export(args.scale_margin)
    if not module_scales:
        raise RuntimeError("No Linear activation statistics were recorded")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "format": "qwen3_asr_fp8_static_activation_scales",
        "version": 1,
        "model": args.model,
        "revision": args.revision,
        "calibration_backend": "qwen_asr_transformers_bfloat16_generate",
        "aggregation": "per_module_max_abs_over_all_inputs_and_calls",
        "fp8_dtype": "float8_e4m3fn",
        "fp8_max": FP8_E4M3_MAX,
        "minimum_scale": FP8_MIN_SCALE,
        "scale_margin": args.scale_margin,
        "batch_size": args.batch_size,
        "max_audio_seconds": args.max_audio_seconds,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "qwen_asr_version": qwen_asr_version,
        "audio_files": [display_path(path) for path in completed_paths],
        "modules": module_scales,
        "vllm_fused_modules": fuse_vllm_scales(module_scales),
    }
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"Recorded {len(module_scales)} Linear modules")
    print(f"Recorded {len(artifact['vllm_fused_modules'])} vLLM module aliases")
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
