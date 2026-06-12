#!/usr/bin/env python3
"""
Phoneme-level transcription of the restored 1929 Joyce ALP recording.

Runs facebook/wav2vec2-lv-60-espeak-cv-ft (a wav2vec2-CTC model fine-tuned to
emit espeak-flavored IPA phonemes) over data/audio/joyce-1929-alp/analysis/
full_16khz_mono.wav and writes timestamped phoneme records.

This is the raw audio-derived IPA artifact: what the model heard Joyce say,
on the same phoneme symbol set as our espeak-driven text→IPA pipeline.
Downstream alignment to the orthographic text of I.8 is a separate step.

Decoding: CTC argmax per output frame, collapsing repeated tokens and blanks
(standard CTC decoding). Frame stride is 320 samples / 16 kHz = 20 ms, giving
50 Hz timestamp resolution. Chunked at 30 s to keep CPU memory bounded.

Output (data/audio/joyce-1929-alp/alignment/i8_phonemes.jsonl):
  {"phoneme": "m", "start_s": 0.12, "end_s": 0.14}
  {"phoneme": "ɪ", "start_s": 0.14, "end_s": 0.16}
  ...

Model: facebook/wav2vec2-lv-60-espeak-cv-ft (~1.2 GB on first run; cached after).
Inference cost: ~10–20 min CPU for ~9 min of audio.

Usage:
    python scripts/preprocessing/audio/transcribe_audio_phonemes.py
    python scripts/preprocessing/audio/transcribe_audio_phonemes.py \\
        --input  data/audio/.../raw_16khz_mono.wav \\
        --output data/audio/.../i8_phonemes_raw.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = REPO_ROOT / "data/audio/joyce-1929-alp/analysis/full_16khz_mono.wav"
DEFAULT_OUTPUT = REPO_ROOT / "data/audio/joyce-1929-alp/alignment/i8_phonemes.jsonl"

MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
CHUNK_S = 30
FRAME_STRIDE_SAMPLES = 320  # wav2vec2 conv stack stride at 16 kHz


def transcribe(
    model: Wav2Vec2ForCTC,
    processor: Wav2Vec2Processor,
    audio: np.ndarray,
    sr: int,
    chunk_s: int,
) -> list[tuple[str, float, float]]:
    """Return [(phoneme, start_s, end_s), ...] across the full audio."""
    samples_per_chunk = chunk_s * sr
    frame_stride_s = FRAME_STRIDE_SAMPLES / sr
    blank_id = model.config.pad_token_id
    id_to_phoneme = {i: p for p, i in processor.tokenizer.get_vocab().items()}

    results: list[tuple[str, float, float]] = []
    n_chunks = (len(audio) + samples_per_chunk - 1) // samples_per_chunk

    for chunk_idx, chunk_start in enumerate(range(0, len(audio), samples_per_chunk)):
        chunk = audio[chunk_start : chunk_start + samples_per_chunk]
        chunk_start_s = chunk_start / sr

        inputs = processor(chunk, return_tensors="pt", sampling_rate=sr)
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        ids = torch.argmax(logits, dim=-1)[0].tolist()

        prev_id: int | None = None
        seg_start_frame = 0
        for frame_i, tid in enumerate(ids):
            if tid != prev_id:
                if prev_id is not None and prev_id != blank_id:
                    ph = id_to_phoneme.get(prev_id, "")
                    if ph and not ph.startswith("<"):
                        results.append((
                            ph,
                            chunk_start_s + seg_start_frame * frame_stride_s,
                            chunk_start_s + frame_i * frame_stride_s,
                        ))
                seg_start_frame = frame_i
                prev_id = tid
        if prev_id is not None and prev_id != blank_id:
            ph = id_to_phoneme.get(prev_id, "")
            if ph and not ph.startswith("<"):
                results.append((
                    ph,
                    chunk_start_s + seg_start_frame * frame_stride_s,
                    chunk_start_s + len(ids) * frame_stride_s,
                ))

        print(
            f"  Chunk {chunk_idx + 1}/{n_chunks}  "
            f"({chunk_start_s:.0f}s–{(chunk_start + len(chunk)) / sr:.0f}s)  "
            f"phonemes so far: {len(results)}",
            flush=True,
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else None)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="16 kHz mono WAV to transcribe")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="JSONL output path")
    args = parser.parse_args()

    audio_path: Path = args.input.resolve()
    out_path: Path = args.output.resolve()

    if not audio_path.exists():
        sys.stderr.write(f"Missing audio: {audio_path}\n")
        sys.exit(1)

    print(f"Loading model: {MODEL_ID}")
    print("  (first run downloads ~1.2 GB to HuggingFace cache)")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    model.eval()

    audio, sr = sf.read(str(audio_path))
    assert sr == 16000, f"Expected 16 kHz audio, got {sr}"
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    print(f"Loaded {len(audio) / sr:.1f}s of audio @ {sr} Hz from {audio_path.name}")

    print(f"\nTranscribing in {CHUNK_S}s chunks...")
    phonemes = transcribe(model, processor, audio, sr, CHUNK_S)
    print(f"\nTotal phonemes: {len(phonemes)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ph, s, e in phonemes:
            f.write(
                json.dumps(
                    {"phoneme": ph, "start_s": round(s, 4), "end_s": round(e, 4)},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
