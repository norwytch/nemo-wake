FROM python:3.11-slim

# System dependencies for the full preprocessing pipeline:
#   espeak-ng    — text→IPA G2P via phonemizer, and required at wav2vec2
#                  phoneme-tokenizer init time (Wav2Vec2PhonemeCTCTokenizer
#                  instantiates an espeak backend during load).
#   ffmpeg       — audio concat + downsample (and adeclick/afftdn filters
#                  we tested empirically; not used in the canonical pipeline
#                  but available for ad-hoc experiments).
#   libsndfile1  — runtime library for soundfile.
RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies cover all preprocessing: scrape, tokenizer training,
# IPA G2P, audio prep, FWEET parsing, wav2vec2 phoneme transcription,
# Smith-Waterman alignment. Heavy: torch (~150 MB) + transformers + biopython.
COPY scripts/requirements.txt scripts/requirements.txt
RUN pip install --no-cache-dir -r scripts/requirements.txt

# Source: preprocessing scripts + shared loaders + tracked annotation overlays.
COPY scripts/ scripts/
COPY shared/ shared/
COPY data/annotations/ data/annotations/
COPY data/manifest.json data/manifest.json

# Mount points for runtime data. All gitignored on the host side, populated
# at runtime via `docker run -v $(pwd)/data:/app/data -v $(pwd)/papers:/app/papers ...`.
#   /app/data/raw/         — scraped Trent JSONL + footnote JSONs
#   /app/data/audio/       — raw FLACs + restored/analysis
#   /app/data/ipa/         — per-language and joyce-derived IPA artifacts
#   /app/data/tokenizers/  — trained BPE/Unigram/Morfessor/FlatCat models
#   /app/data/pos/         — multi-method POS-hypothesis JSONL per token
#   /app/papers/           — FWEET_elucidation HTML (input to parse_fweet)
RUN mkdir -p data/raw data/audio data/ipa data/tokenizers data/pos papers

# Default: canonical first preprocessing step (Trent scrape).
# Override for other scripts, e.g.:
#   docker run wake-scraper python scripts/preprocessing/text/scrape_footnotes.py
#   docker run wake-scraper python scripts/preprocessing/text/parse_fweet.py
#   docker run wake-scraper python scripts/preprocessing/audio/restore_audio.py
#   docker run wake-scraper python scripts/preprocessing/audio/transcribe_audio_phonemes.py
CMD ["python", "scripts/preprocessing/text/scrape_trent.py"]
