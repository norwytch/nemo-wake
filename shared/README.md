This repo is the home of any utilities that might be shared by multiple Wake projects. For now, it includes loaders for the Wake corpus, FWEET data, the annotation overlays, IPA transcriptions, and trained tokenizers — in `corpus.py`, `fweet.py`, `annotations.py`, `ipa.py`, and `tokenizers.py` respectively.

`corpus.py` loads the entire corpus from the `data/raw` directory, or loads by chapter, and can also iterate over lines filtered by book, chapter, or train/val/test split. It outputs `Line` and `Chapter` dataclasses carrying the canonical `page_line` coordinate (e.g. `"276.03"`), the line's `text`, and `left_margin` / `right_margin` fields (populated only for II.2). Also provides `load_footnotes(book, episode)` for the companion footnote body text and `line_split(page_line)` for the deterministic 80/10/10 train/val/test split.

`fweet.py` loads the parsed FWEET elucidations at `data/raw/fweet_elucidations.jsonl` (produced by `scripts/preprocessing/text/parse_fweet.py` from the HTML file at `papers/FWEET_elucidation`, generously provided by Raphael Slepon). The loader extracts source-language tags (with mappings to espeak-ng language codes), motifs, person references, source references, sigla, and the elucidation body text — all indexed by page.line. Note that using this utility inherently means assigning labels to some data, and so if you prefer an entirely unsupervised approach, this will violate that principle.

`annotations.py` loads the annotations for sigla in the corpus, and then returns the Unicode entry or description for the sigla. Note that for the music section in II.2, we do not load anything.

`ipa.py` loads the IPA transcriptions in `data/ipa/{language}/` (per-language subdirectories). Each record is one orthographic token with phoneme symbols listed under `tokens` for main text and `left_margin_tokens` / `right_margin_tokens` for II.2 marginalia. Transcriptions are at the phoneme level but tokenized at the word level — no morphological segmentation.

`tokenizers.py` contains three families of segmenters, each representing a distinct inductive bias on what a "subword unit" is:

- **BPE (byte-pair encoding):** greedy frequency-based merges. Pre-tokenized on `Whitespace` (not `ByteLevel`) so segmentation respects word boundaries. The Wake has ~72k unique types; vocab sizes at or above that range degenerate to whole-word memorization. We sweep over `(1000, 2000, 4000, 8000, 16000, 32000)`.
- **Unigram LM (Kudo 2018, SentencePiece-style):** probabilistic. Trained by EM to maximize corpus likelihood under a unigram distribution over subwords, then pruned to the target vocab. Closer to Morfessor's MDL spirit than BPE's frequency heuristic. Same library as BPE; same vocab-size sweep for a clean head-to-head.
- **Morfessor (Baseline + FlatCat):** MDL-based morphological segmentation.
    - **Baseline:** flat lexicon, unigram-based segmentation. Assumes morphs in a compound occur independently (bag-of-morphs) — evidently false for the Wake, where *passencore* is a sequence of source words (French *pas encore* + English *passenger*/*encore*) whose order is load-bearing. Baseline is the canonical reference point and documents this failure mode.
    - **FlatCat (Grönroos et al. 2014):** HMM over morph categories (prefix/stem/suffix/non-morpheme) with a flat lexicon. Adds sequential structure on top of Baseline. Works unsupervised but underperforms the unmaintained Categories-MAP variant in that regime; the flat lexicon makes it the right choice if we later hand-annotate a small set of Wake nonce words.



Not in scope (yet):
- **Segmental neural language models** (Kawakami & Dyer 2017/19): future work — character-level neural LMs with latent segment boundaries. The natural follow-on once we know what BPE / Unigram / Morfessor disagree about.
- **Tokenization-free models** (ByT5, CANINE, MEGABYTE): orthogonal — these sidestep segmentation entirely, useful for downstream tasks but not for the research question "what are the Wake's morphs."
