#!/usr/bin/env python3
"""
Prepare a Joyce recording for phonological analysis.

Concatenates the per-side raw audio files for a given recording and produces
both archival (FLAC, native rate) and analysis (16 kHz mono WAV) artifacts.

**No restoration/DSP is applied.** Conservative classical DSP (60 Hz
high-pass + spectral gating via noisereduce; ffmpeg adeclick + afftdn)
was tested empirically on the 1929 ALP: per-token alignment of the
wav2vec2-CTC phoneme transcription against the I.8 espeak G2P showed no
meaningful difference between DSP-processed and raw audio (mean ED 2.013
vs 2.005, 1039 vs 1032 tokens aligned). The audible difference (volume)
doesn't carry into the phoneme outputs. Neural enhancers were avoided to
prevent phonetic hallucination.

Supported recordings:
  joyce-1929-alp     — 2 FLACs (archive.org Great 78 Project), concat.
  joyce-1924-aeolus  — 1 MP3 (HMV / John F. Taylor speech), decode via ffmpeg.

Inputs:  data/audio/<recording>/raw/<files>
Outputs: data/audio/<recording>/restored/full.flac           (native rate, archival)
         data/audio/<recording>/analysis/full_16khz_mono.wav (downsampled mono for ML)

Usage:
    python scripts/preprocessing/audio/restore_audio.py
    python scripts/preprocessing/audio/restore_audio.py --recording joyce-1924-aeolus
"""

import argparse
import subprocess
import sys
import tempfile
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

REPO_ROOT = Path(__file__).resolve().parents[3]

# Per-recording config. Each recording has a glob for source files and a flag
# for whether to concatenate (single-file recordings skip concat).
RECORDINGS: dict[str, dict] = {
    "joyce-1929-alp": {
        "glob": "*.flac",
        "concat": True,
        "archival_subtype": "PCM_24",
    },
    "joyce-1924-aeolus": {
        "glob": "*.mp3",
        "concat": False,
        "archival_subtype": "PCM_16",
    },
}


def resample_to_16k_mono(audio: np.ndarray, sr: int) -> np.ndarray:
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    if sr == 16000:
        return mono
    g = gcd(sr, 16000)
    return resample_poly(mono, 16000 // g, sr // g)


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    """Read FLAC/WAV via soundfile; decode MP3 (and other formats) via ffmpeg."""
    suffix = path.suffix.lower()
    if suffix in (".flac", ".wav", ".aiff", ".aif", ".ogg"):
        audio, sr = sf.read(str(path), always_2d=True)
        return audio, sr

    # ffmpeg → temporary WAV → soundfile. Preserves channels and sample rate.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(path),
            "-f", "wav", "-acodec", "pcm_s16le",
            tmp.name,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            sys.stderr.write(f"ffmpeg failed for {path}:\n{result.stderr}\n")
            sys.exit(1)
        audio, sr = sf.read(tmp.name, always_2d=True)
    return audio, sr


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore + downsample a Joyce recording")
    parser.add_argument(
        "--recording",
        default="joyce-1929-alp",
        choices=sorted(RECORDINGS),
        help="Recording name (subdirectory under data/audio/)",
    )
    args = parser.parse_args()
    cfg = RECORDINGS[args.recording]

    rec_root = REPO_ROOT / "data/audio" / args.recording
    raw_dir = rec_root / "raw"
    restored_dir = rec_root / "restored"
    analysis_dir = rec_root / "analysis"

    raw_files = sorted(raw_dir.glob(cfg["glob"]))
    if not raw_files:
        sys.stderr.write(f"No {cfg['glob']} files in {raw_dir}\n")
        sys.exit(1)
    if not cfg["concat"] and len(raw_files) > 1:
        sys.stderr.write(
            f"Recording {args.recording!r} expects 1 source file but found {len(raw_files)}:\n"
        )
        for f in raw_files:
            sys.stderr.write(f"  {f.name}\n")
        sys.exit(1)

    print(f"Recording: {args.recording}")
    print(f"Found {len(raw_files)} raw file(s):")
    for f in raw_files:
        print(f"  {f.name}")

    audio_parts: list[np.ndarray] = []
    sr_seen: int | None = None
    for f in raw_files:
        audio, sr = _read_audio(f)
        if sr_seen is None:
            sr_seen = sr
        elif sr_seen != sr:
            sys.stderr.write(f"Sample rate mismatch: {sr} vs {sr_seen}\n")
            sys.exit(1)
        audio_parts.append(audio)
    audio = np.concatenate(audio_parts, axis=0) if cfg["concat"] else audio_parts[0]
    sr = sr_seen
    n_ch = audio.shape[1]
    print(
        f"\nLoaded {audio.shape[0]:,} samples @ {sr} Hz "
        f"({audio.shape[0] / sr:.1f}s, {n_ch} channels)"
    )

    restored_dir.mkdir(parents=True, exist_ok=True)
    restored_path = restored_dir / "full.flac"
    sf.write(str(restored_path), audio.astype(np.float32), sr, subtype=cfg["archival_subtype"])
    print(f"Wrote {restored_path.relative_to(REPO_ROOT)} (archival, no DSP)")

    analysis_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = analysis_dir / "full_16khz_mono.wav"
    mono16k = resample_to_16k_mono(audio, sr)
    sf.write(str(analysis_path), mono16k.astype(np.float32), 16000, subtype="PCM_16")
    print(f"Wrote {analysis_path.relative_to(REPO_ROOT)} (16 kHz mono, for ML)")


if __name__ == "__main__":
    main()
