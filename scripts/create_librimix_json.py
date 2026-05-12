#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import soundfile as sf
from tqdm import tqdm


def wav_length(path):
    info = sf.info(str(path))
    return int(info.frames)


def collect_wavs(directory):
    return {path.name: path.resolve() for path in sorted(directory.rglob("*.wav"))}


def write_json(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(items, f, indent=4)


def main():
    parser = argparse.ArgumentParser(
        description="Create mixture/source JSON manifests for Libri2Mix/WHAM-style training."
    )
    parser.add_argument(
        "--split_dir",
        required=True,
        help="Split directory, e.g. DataPreProcess/Libri2Mix/train-100.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Where to write JSON files. Defaults to --split_dir.",
    )
    parser.add_argument(
        "--mix_dir",
        default="mix_clean",
        help="Mixture wav directory name. Use mix_both for noisy WHAM/LibriMix.",
    )
    parser.add_argument(
        "--mix_json",
        default=None,
        help="Output mixture JSON filename. Defaults to <mix_dir>.json.",
    )
    parser.add_argument("--s1_dir", default="s1")
    parser.add_argument("--s2_dir", default="s2")
    args = parser.parse_args()

    split_dir = Path(args.split_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else split_dir

    mix_dir = split_dir / args.mix_dir
    s1_dir = split_dir / args.s1_dir
    s2_dir = split_dir / args.s2_dir

    for directory in [mix_dir, s1_dir, s2_dir]:
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

    mix_wavs = collect_wavs(mix_dir)
    s1_wavs = collect_wavs(s1_dir)
    s2_wavs = collect_wavs(s2_dir)

    common_names = sorted(set(mix_wavs) & set(s1_wavs) & set(s2_wavs))
    if not common_names:
        raise RuntimeError(
            "No matching wav filenames found across mix/s1/s2 directories. "
            "Check --mix_dir, --s1_dir, and --s2_dir."
        )

    missing_s1 = sorted(set(mix_wavs) - set(s1_wavs))
    missing_s2 = sorted(set(mix_wavs) - set(s2_wavs))
    if missing_s1 or missing_s2:
        print(f"Warning: {len(missing_s1)} mix files missing in s1")
        print(f"Warning: {len(missing_s2)} mix files missing in s2")

    mix_infos = []
    s1_infos = []
    s2_infos = []

    for name in tqdm(common_names, desc=f"Creating JSON for {split_dir.name}"):
        mix_path = mix_wavs[name]
        s1_path = s1_wavs[name]
        s2_path = s2_wavs[name]

        mix_len = wav_length(mix_path)
        s1_len = wav_length(s1_path)
        s2_len = wav_length(s2_path)

        if not (mix_len == s1_len == s2_len):
            print(
                f"Warning: length mismatch for {name}: "
                f"mix={mix_len}, s1={s1_len}, s2={s2_len}"
            )

        mix_infos.append([str(mix_path), mix_len])
        s1_infos.append([str(s1_path), s1_len])
        s2_infos.append([str(s2_path), s2_len])

    mix_json = args.mix_json or f"{args.mix_dir}.json"

    write_json(out_dir / mix_json, mix_infos)
    write_json(out_dir / "s1.json", s1_infos)
    write_json(out_dir / "s2.json", s2_infos)

    print(f"Wrote {len(common_names)} examples to: {out_dir}")
    print(f"- {out_dir / mix_json}")
    print(f"- {out_dir / 's1.json'}")
    print(f"- {out_dir / 's2.json'}")


if __name__ == "__main__":
    main()
