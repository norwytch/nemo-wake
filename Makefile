.DEFAULT_GOAL := help

.PHONY: help install lint check-corpus build \
        scrape scrape-footnotes parse-fweet train-tokenizers transcribe-ipa lexical-stats \
        restore-audio transcribe-audio-phonemes align-audio extract-joyce-rules joyce-derived-ipa

# Full-preprocessing image (text + audio + IPA + tokenizers).
IMAGE := wake-preprocess

# Mounts:
#   data/raw         — scraped Trent JSONL + footnote JSONs
#   data/audio       — raw FLAC + restored + analysis WAV
#   data/ipa         — per-language and joyce-derived IPA artifacts
#   data/tokenizers  — trained BPE/Unigram/Morfessor/FlatCat models
#   papers           — FWEET_elucidation HTML (input to parse_fweet)
#   HF cache         — wav2vec2-lv-60-espeak-cv-ft (~1.2 GB) persisted across runs
#   stanza cache     — Stanza UD POS-tagger models (~3.6 GB across FWEET languages)
DOCKER_RUN := docker run --rm \
    -v $(CURDIR)/data/raw:/app/data/raw \
    -v $(CURDIR)/data/audio:/app/data/audio \
    -v $(CURDIR)/data/ipa:/app/data/ipa \
    -v $(CURDIR)/data/tokenizers:/app/data/tokenizers \
    -v $(CURDIR)/data/pos:/app/data/pos \
    -v $(CURDIR)/papers:/app/papers \
    -v $(HOME)/.cache/huggingface:/root/.cache/huggingface \
    -v $(HOME)/stanza_resources:/root/stanza_resources \
    $(IMAGE)

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*##"}; {printf "  %-28s %s\n", $$1, $$2}'

install: ## Install shared package and dev tools locally (run once per venv)
	pip install -e ".[dev]"

lint: ## Run ruff linter on all Python files
	ruff check .

check-corpus: ## Verify data/raw/ integrity against data/manifest.json
	python scripts/preprocessing/text/check_corpus.py

build: ## Build the full-preprocessing Docker image
	docker build -f scripts/Dockerfile -t $(IMAGE) .

# ---- text pipeline ---------------------------------------------------------

scrape: build ## Scrape Trent chapters into data/raw/ (skips existing)
	mkdir -p data/raw
	$(DOCKER_RUN) python scripts/preprocessing/text/scrape_trent.py

scrape-footnotes: build ## Scrape telelib footnote body text + apply FWEET corrections
	mkdir -p data/raw
	$(DOCKER_RUN) python scripts/preprocessing/text/scrape_footnotes.py

parse-fweet: build ## Parse FWEET HTML into data/raw/fweet_elucidations.jsonl
	mkdir -p data/raw
	$(DOCKER_RUN) python scripts/preprocessing/text/parse_fweet.py

train-tokenizers: build ## Train BPE / Unigram / Morfessor / FlatCat sweeps
	mkdir -p data/tokenizers
	$(DOCKER_RUN) python scripts/preprocessing/text/train_tokenizers.py

transcribe-ipa: build ## espeak-ng G2P per language → data/ipa/<lang>/
	mkdir -p data/ipa
	$(DOCKER_RUN) python scripts/preprocessing/text/transcribe_ipa.py

lexical-stats: build ## Corpus-level type/token/hapax/entropy stats
	$(DOCKER_RUN) python scripts/preprocessing/text/lexical_stats.py

pos-tag: build ## Multi-method POS hypotheses per token (Stanza, UD tags)
	mkdir -p data/pos
	$(DOCKER_RUN) python scripts/preprocessing/text/pos_tag.py

# ---- audio pipeline --------------------------------------------------------

restore-audio: build ## Concat raw FLACs → data/audio/joyce-1929-alp/restored + 16k mono analysis WAV
	mkdir -p data/audio
	$(DOCKER_RUN) python scripts/preprocessing/audio/restore_audio.py

transcribe-audio-phonemes: build ## wav2vec2-CTC phoneme transcription of Joyce 1929 ALP
	mkdir -p data/audio
	$(DOCKER_RUN) python scripts/preprocessing/audio/transcribe_audio_phonemes.py

align-audio: build ## Smith-Waterman align audio phonemes to text IPA
	$(DOCKER_RUN) python scripts/preprocessing/audio/align_audio_to_text.py

extract-joyce-rules: build ## Derive Joyce-specific phonological rules from alignment
	$(DOCKER_RUN) python scripts/preprocessing/audio/extract_joyce_rules.py

joyce-derived-ipa: build ## Apply Joyce rules over FWEET-multilingual IPA → data/ipa/joyce-derived/
	mkdir -p data/ipa
	$(DOCKER_RUN) python scripts/preprocessing/audio/generate_joyce_derived_ipa.py
