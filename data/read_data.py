import json
import os
import re
import shutil
from pathlib import Path


CHANNEL_SUFFIX_RE = re.compile(r"_(channel_\d+)$")
CHANNEL_NAME_RE = re.compile(r"channel[_-]?(\d+)")
SAFE_PATH_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_path_part(value: str) -> str:
    value = SAFE_PATH_RE.sub("_", value.strip()).strip("._-")
    return value or "unknown"


def read_manifest_records(path: Path):
    try:
        with path.open("r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        with path.open("r") as f:
            for line_no, line in enumerate(f, start=1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSONL in {path} at line {line_no}") from exc
        return

    if isinstance(data, list):
        yield from data
    else:
        yield data


def split_sample_and_channel(audio_path: Path, manifest_channel: str) -> tuple[str, str]:
    match = CHANNEL_SUFFIX_RE.search(audio_path.stem)
    if match:
        return audio_path.stem[: match.start()], match.group(1)

    channel_match = CHANNEL_NAME_RE.search(manifest_channel)
    channel = f"channel_{channel_match.group(1)}" if channel_match else manifest_channel.rstrip("_")
    return audio_path.stem, safe_path_part(channel)


def place_audio(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "auto":
        try:
            os.link(src, dst)
        except OSError:
            dst.symlink_to(src)
    else:
        dst.symlink_to(src)


def write_ground_truth(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.write_text(text.rstrip() + "\n")


def prepare_dataset(
    mapping_path: Path,
    output_dir: Path,
    mode: str = "symlink",
    overwrite: bool = False,
) -> None:
    with mapping_path.open("r") as f:
        mappings = json.load(f)

    stats = {
        "written": 0,
        "missing_manifests": 0,
        "missing_audio": 0,
        "bad_records": 0,
    }

    for title, channels in mappings.items():
        dataset_dir = output_dir / safe_path_part(title)

        for manifest_channel, manifest_file in channels.items():
            manifest_file = Path(manifest_file)
            if not manifest_file.exists():
                stats["missing_manifests"] += 1
                print(f"Missing manifest for {title}/{manifest_channel}: {manifest_file}")
                continue

            for record in read_manifest_records(manifest_file):
                audio_file = Path(record.get("audio_filepath", ""))
                text = record.get("text")
                if not audio_file or text is None:
                    stats["bad_records"] += 1
                    print(f"Bad record in {manifest_file}: missing audio_filepath or text")
                    continue
                if not audio_file.exists():
                    stats["missing_audio"] += 1
                    print(f"Missing audio for {title}/{manifest_channel}: {audio_file}")
                    continue

                sample_id, channel = split_sample_and_channel(audio_file, manifest_channel)
                sample_dir = dataset_dir / safe_path_part(sample_id)
                sample_dir.mkdir(parents=True, exist_ok=True)

                audio_dst = sample_dir / f"{safe_path_part(channel)}{audio_file.suffix}"
                text_dst = sample_dir / f"{safe_path_part(channel)}.txt"

                place_audio(audio_file, audio_dst, mode=mode, overwrite=overwrite)
                write_ground_truth(text_dst, text, overwrite=overwrite)
                stats["written"] += 1

    print(
        f"Wrote {stats['written']} audio/text pairs to {output_dir} "
        f"({stats['missing_manifests']} missing manifests, "
        f"{stats['missing_audio']} missing audio files, "
        f"{stats['bad_records']} bad records)."
    )


if __name__ == "__main__":
    prepare_dataset(
        mapping_path=Path(__file__).with_name("manifest.json"),
        output_dir=Path(__file__).with_name("prepared_data"),
        mode="copy",
        overwrite=True,
    )