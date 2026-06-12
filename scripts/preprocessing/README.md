# Preprocessing Finnegans Wake

We must begin with the ground truth: the text of the Wake itself. The Wake uses non-standard orthography, morphology, and organization that is lost if work operates under standard assumptions of English structure using basic NLP tools. It therefore requires a level of deliberate pre-processing that standard English texts do not. This README details the decisions made in preprocessing our chosen Wake edition, open questions, and possible downstream effects of our preprocessing choices.

The Trent University edition is the standard edition for digital scholarship. It was scraped from the Wayback Machine, and then cross-checked:
  - for known errors via a [list maintained by FWEET.](http://www.fweet.org/pages/fw_typo.php).
  - for sigla via the [Hart Concordance.](http://www.rosenlake.net/fw/FWconcordance/)
  - against the [telelib copy](https://www.telelib.com/authors/J/JoyceJames/prose/finneganswake/index.html) to separate out footnotes.

TEI XML is the standard format for most digital humanities work, and there already exists a [TEI XML formatting of the Wake](https://github.com/open-editions/corpus-joyce-finnegans-wake-tei). However, for ease of use in future natural language processing tasks, we chose to render our corpus in JSONL, with one JSONL file per episode. For ease of future use in conventional Wake scholarship, we maintain the `book`, `episode`, `page_line` records within each chapter's JSONL recording, along with  text, and fields for footnotes and marginalia. This renders any NLP findings easily localizable via the traditional `page.line` notation. Text is preserved in Unicode and is UTF-8 encoded. 

## The Problem

The primary deviations from standard English processing that we observe in the literature include:
- Written words are not and do not contain consistent morphological boundaries in the Wake. Therefore, we do not employ word tokenization techniques, correct any spelling errors, or employ any spelling or lemmatization, as these techniques are based on assumptions about standard English which have been shown to provide no valuable analysis on the Wake [TODO: cite Bayesie, FinneGAN]. Deviations from conventional English spelling and morphology are semantically significant and must be preserved. The only exception made to the original text is the rejoining of words split across line break by a hyphen.
- High token to type ratio skews statistical analysis (Hart, 1963). [TODO: put standard token:type ratio, other lexical stats of the wake compared to standard English]
- Orthography includes mathematical and musical notation, some small drawings, and unique symbols of Joyce's creation called [sigla](http://www.rosenlake.net/fw/FWconcordance/), which standard text encoding tools are not equipped to handle. Mathematical notations and a few of the sigla have corresponding Unicode symbols; drawings, musical notation, and sigla are tracked in annotation overlay files in `data/annotations/`. For sigla without Unicode symbols, we have referred to them by the description found in the Hart concordance in each `annotations` overlay file.
- II.2 includes left/right marginalia and footnotes, intended to simultaneously mimic a child's schoolbook and scholarly commentary. The Trent edition preserves this three-column structure per line, with separate fields for left margin, main text, and right margin. Footnote references survive as `[N]` inline markers.
- Punctuation and capitalization is significant, and words carry different semantic meaning in lowercase versus uppercase, or relative to a punctuation mark. This is especially notable in the "thunder words." We therefore preserve punctuation and capitalization, contrary to convention, which strips both. (Drożdż, 2015) [TODO: figure out which edition of the wake these guys were using]

Note that some scholars spend their entire careers locating every dimension of Wakean linguistic deviance and annotating it. We cannot and do not attempt to address every deviance, including different font sizes, italics, bold type, etc, but instead leave such work to future scholars.

## Scripts

Preprocessing is split into two separate paths: `audio` and `text`. The `audio` dir is intended for use on primarily phonological tasks, and the `text` dir is intended for use on primarily morphological tasks.

[TODO: put dir tree path here]

### `audio`
This directory contains scripts that process audio readings of Joyce's own voice. Two Joyce recordings are used:

- **1929 Anna Livia Plurabelle** (`data/audio/joyce-1929-alp/`). Spans a section of Book I.8 (FW 213–216, the washerwomen's closing dialogue). Digitized from the original shellac disk by archive.org's Great 78 Project, available on the Internet Archive in two parts. The .flac files (`ALP_1.flac` and `ALP_2.flac`, both 96 kHz / 24-bit stereo) are stored in `raw/`. **This is the only recording of Joyce reading the Wake itself** and is used to derive phonological rules.
- **1924 Aeolus** (`data/audio/joyce-1924-aeolus/`). Joyce reading the John F. Taylor speech from *Ulysses* (Episode 7), recorded by HMV / Sylvia Beach. ~4 minutes, MP3. Not Wake text, so this recording is used **only as an out-of-distribution held-out** for the I.8-derived rules: do rules from one Joyce recording predict another? Expected text is at `data/raw/ulysses_aeolus_taylor.txt`.

The recording has non-speech noise common to digitized shellac records known as "hisses," "clicks," and "pops." Several open-source tools exist for addressing this noise: stock Audacity (still free and open-source under GPL v3, including its Click Removal and Noise Reduction effects), ffmpeg's `adeclick` and `afftdn` filters, and Python packages like `noisereduce` and `RNNoise`. Neural speech enhancers like DeepFilterNet and resemble-enhance are also available but were avoided here: trained on modern speech, they carry a high risk of phonetic hallucination on degraded historical audio, which is a methodology landmine for phonological analysis. Among classical-DSP options we tested two conservative pipelines (`noisereduce` spectral gating + 60 Hz high-pass; ffmpeg `adeclick` + `afftdn`) and found that neither produced an audible improvement on this specific recording, and that the audible difference (slight volume reduction) doesn't propagate into the wav2vec2-CTC phoneme outputs in any meaningful way. We empirically validated this decision: per-token Smith-Waterman alignment of wav2vec2 phoneme transcriptions against I.8 espeak G2P showed essentially identical alignment quality for DSP-processed vs raw audio (mean per-token edit distance 2.005 vs 2.013; 1032 vs 1039 tokens aligned). We therefore use the original, raw recording — `restore_audio.py` simply concatenates the two parts and resamples to 16 kHz mono for downstream ML, applying no DSP. We recommend that future scholars with access to professional audio restoration tools (iZotope RX, Click Repair) revisit this decision.

Nevertheless, the audio recording remains significant. Phonological patterns were important to Joyce, and we contend that the recording of Joyce is the authoritative source on the intended phonology of the Wake. Therefore, in order to do any phonological analysis of the Wake, we must take this recording as our baseline. 

Two options exist for the extension of the recording's phonological patterns across the rest of the text: clone the voice in the recording with known voice cloning and speech synthesis models (ElevenLabs, GANs, etc), and have that model run inference across the entire Wake; or, transcribe the recording directly into the International Phonetic Alphabet, observe phonological patterns both unique to Joyce and consistent with Hiberno-English, and then apply rules based on those patterns to the entire Wake on top of a grapheme-to-phoneme tool trained on standard English. Given the recording is an extremely small sample, we concluded the first option posed too great a risk of overfitting. Voice cloning also posed practical concerns: Joyce's estate is notoriously litigious, and having a voice clone of the author read one of his works seemed too gray of a legal area to risk. Additionally, as the authors of this project hold Joyce and his art in great esteem, we were also influenced by a desire not to create a Joyce homunculus. It is our greatest desire that future computational Joyce scholars share the same principle. Therefore, we chose to simply proceed with a direct IPA transcription -> rules derivation -> grapheme-to-phoneme pipeline. This takes place with the following scripts:

`restore_audio.py` - decodes and downsamples a Joyce recording. Takes a `--recording` arg (default `joyce-1929-alp`) selecting which recording to process. For the 1929 ALP, concatenates the two `raw/*.flac` files into a single 96 kHz / 24-bit stereo `restored/full.flac`. For the 1924 Aeolus, decodes the single `raw/*.mp3` via ffmpeg. Both then resample to 16 kHz mono and write `analysis/full_16khz_mono.wav` for downstream wav2vec2-CTC / MFA / Whisper-phoneme tooling. No DSP applied (see preceding paragraph for rationale and empirical validation). The name "restore" is preserved for compatibility but is a misnomer — it means "prepare for analysis."

```
python scripts/preprocessing/audio/restore_audio.py                            # 1929 ALP (default)
python scripts/preprocessing/audio/restore_audio.py --recording joyce-1924-aeolus
```

`transcribe_audio_phonemes.py` - takes in 16 kHz mono Joyce audio and transcribes it to IPA using `facebook/wav2vec2-lv-60-espeak-cv-ft`, a wav2vec2-Large variant fine-tuned on Common Voice 6.1 / LibriVox-60K with espeak-derived phoneme labels. The model is multilingual but its training data is heavily English-weighted, so phoneme predictions on degraded Hiberno-English audio are skewed toward standard English G2P expectations. Accepts `--input` / `--output` so it can be applied to either recording. Output: one record per phoneme with `start_s` / `end_s` timestamps at 20 ms resolution.

```
python scripts/preprocessing/audio/transcribe_audio_phonemes.py  # defaults to I.8
python scripts/preprocessing/audio/transcribe_audio_phonemes.py \
    --input  data/audio/joyce-1924-aeolus/analysis/full_16khz_mono.wav \
    --output data/audio/joyce-1924-aeolus/alignment/aeolus_phonemes.jsonl
```

`align_audio_to_text.py` - aligns an audio phoneme stream against an expected-text phoneme stream via Smith-Waterman local alignment. Two text sources are supported: the FW corpus (default — book/episode flags), or an external plain-text file via `--text-file`. Runs espeak G2P over the orthographic tokens to build expected IPA, then aligns. Output: one record per orthographic token with `aligned: bool`, observed IPA, expected IPA, edit distance, coverage, and timestamps for aligned tokens. For I.8, ~1,032 of 7,907 tokens fall within the ~9-minute recorded span. For Aeolus, 370 of 585 tokens align — Joyce starts roughly ⅓ into the Taylor speech text per the SW span.

```
python scripts/preprocessing/audio/align_audio_to_text.py                       # I.8 (default)
python scripts/preprocessing/audio/align_audio_to_text.py \
    --input  data/audio/joyce-1924-aeolus/alignment/aeolus_phonemes.jsonl \
    --output data/audio/joyce-1924-aeolus/alignment/aeolus_tokens_aligned.jsonl \
    --text-file data/raw/ulysses_aeolus_taylor.txt
```

`extract_joyce_rules.py` - reads the I.8 aligned tokens and emits a tiered ruleset of phoneme substitutions Joyce produces relative to espeak en-us. Tiers, most-specific to least:

1. **`phoneme_pair`**: `(expected, preceding, following)` triple → observed.
2. **`word_position`**: `(expected, {initial, medial, final, isolate})` → observed.
3. **`context_free`**: `expected` → observed.

At application time, the most-specific firing tier wins, falling back to identity if no tier fires. Context (preceding/following phoneme and position) is computed over the **expected** sequence, so the rules are usable outside I.8 where there is no audio. Thresholds: `MIN_SUPPORT=3` observations and `MIN_RATE=0.50` — a rule must be the majority observation, not just modal.

Evaluation reports the **phoneme error rate (PER)** — total edit distance over total expected phonemes — with a Wilson 95% CI, on the 20% held-out (val+test) split. Plus a **per-rule leave-one-out lift**: for each substitution rule, recompute held-out ED with that rule alone removed, and rank rules by Δ ED. Substitution rules with negative LOO lift (net-hurting on held-out) are filtered out before serialization. Output: `data/audio/joyce-1929-alp/rules/joyce_rules.json`.

Current ruleset (after filtering): 148 phoneme_pair / 62 word_position / 23 context_free rules. Substitution counts: 29 / 10 / 2. The data-driven rules independently surface canonical Hiberno-English features at appropriately specific tiers — most strikingly `ŋ|ɪ|#` → `n` (word-final -ING reduction in the *KIT*-preceded environment, the linguistically motivated context) is the top-helping rule on held-out. Context-free Joyce-specific rules at MIN_RATE=0.50 collapse to just two (down from 10 at 0.30), reflecting how much of the "data-driven" signal was previously borderline.

**Held-out evaluation on I.8 (val+test, 153 tokens, 555 expected phonemes):**

| | Total ED | PER | 95% CI |
|---|---|---|---|
| Identity (raw espeak) | 313 | 0.564 | [0.522, 0.605] |
| Tiered rules applied | 303 | 0.546 | [0.504, 0.587] |

Improvement: 10 phonemes (+3.19%). **95% CIs overlap → improvement is not statistically significant at α=0.05.** The filter caveat (filtering hurting rules uses the same data as evaluation) is documented in the rules JSON metadata.

`evaluate_rules_on_aeolus.py` - out-of-distribution evaluation of the I.8-trained rules on the 1924 Aeolus alignment. Same scoring functions as `extract_joyce_rules.py`. Output: `data/audio/joyce-1924-aeolus/rules/aeolus_eval.json` + stdout cross-recording comparison.

**Cross-recording finding:** the I.8-trained rules do not generalize.

| Set | PER | 95% CI | vs identity |
|---|---|---|---|
| I.8 held-out identity | 0.564 | [0.522, 0.605] | — |
| I.8 held-out rules | 0.546 | [0.504, 0.587] | +3.19% |
| Aeolus identity | 0.680 | [0.655, 0.704] | — |
| Aeolus rules | 0.681 | [0.655, 0.705] | −0.11% |

The 3% lift on I.8 held-out is within-recording. Applied to a different Joyce recording, the rules add essentially nothing (4 helping, 4 hurting, 32 neutral on Aeolus LOO). The Aeolus identity PER is also 12 points worse than I.8 identity (genre + audio-quality + alignment-coverage differences), so both recordings sit in a high-noise regime. The honest interpretation: **a single ~8-min recording is not enough data to derive generalizable Joyce phonology** through this pipeline. The literature-grounded Hiberno-English artifact (see below) is the defensible alternative baseline for phonological work.

`generate_joyce_derived_ipa.py` - takes the orthographic tokens of each chapter (whitespace-split, hyphen-rejoined across line breaks) and produces a unified per-token IPA artifact at `data/ipa/joyce-derived/book{NN}_ep{NN}.jsonl`. Each token record carries:

- `ipa.en-us-baseline`: naive espeak English G2P
- `ipa.en-us-hiberno-english`: English baseline with the literature-grounded Dublin HE ruleset applied (see Hiberno-English Rules section below). Present for every token.
- `ipa.en-us-joyce-audio-observed`: wav2vec2-CTC observation from Joyce's 1929 ALP recording. Present **only** for the ~1,032 aligned I.8 tokens.
- `ipa.en-us-joyce-rule-applied`: English baseline with the data-driven Joyce tiered ruleset applied. Present for **every token except** the aligned I.8 tokens (which have the audio observation instead). `joyce_ipa_source: "audio-observed"` or `"rule-applied"` records the provenance.
- `ipa.{lang}-baseline`: per-language espeak G2P for each non-English language FWEET tags on the token's page-line (e.g. `fr-fr-baseline` for `pas encore` annotations).
- `fweet_languages`: `{espeak_code: {confidence, source_form, edit_distance, gloss}}` — confidence is `high` / `medium` / `low` / `line-only` based on orthographic similarity between the token and the FWEET elucidation's source-form.
- `fweet_motifs`: list of motif labels active on the page-line (e.g. "Motif: HCE").
- `fweet_glosses`: human-readable list of all language elucidations on the page-line.

The split between `-audio-observed`, `-rule-applied`, and `-hiberno-english` is deliberate: the field name should not oversell what is in it. `-rule-applied` is "espeak with weakly-supported, non-generalizing rules"; `-hiberno-english` is "espeak with literature-grounded rules"; only `-audio-observed` is grounded in Joyce's actual voice (and only for I.8). Downstream consumers should choose the field appropriate to the claim being made.

For example, `passencore` (FW 003.04) resolves to `en-us-baseline: /pæsəŋkɔːɹ/`, `en-us-joyce-rule-applied: /pæsʌnkɔːɹ/` (Joyce rules apply `/ə/→/ʌ/` and `/ŋ/→/n/`), `en-us-hiberno-english: /pæsəŋkɔː/` (HE rules apply word-final r-deletion), `fr-fr-baseline: /pasɑ̃kɔʁ/` (FWEET tags French `pas encore` at high confidence, edit distance 1).

#### Hiberno-English rules (`data/annotations/hiberno_english_rules.json`)

A literature-grounded ruleset for Dublin Hiberno-English c.1900 (Joyce's native variety), hand-curated from Wells (1982) *Accents of English* Vol. 2 Ch. 5, Hickey (2007) *Irish English: History and Present-Day Forms*, and Hickey's Irish English Resource Centre. Same tiered schema as the data-driven Joyce rules — `apply_tiered_rules` in either of the two scripts above will apply it. Lives in `data/annotations/` rather than `data/audio/` because it is hand-curated annotation, not a derivation from the audio data.

Schema extension over the data-driven format: a rule's `predicted` field may be a string (1→1 substitution), a list of strings (1→N insertion), or an empty list / empty string (1→0 deletion). This is required for r-deletion and vowel breaking, which the data-driven format cannot express. The `apply_tiered_rules` function normalizes via a small `_as_list()` helper; existing scalar-valued joyce_rules.json continues to work unchanged.

Each rule carries `confidence`, `register` (`local Dublin` / `supraregional` / `both`), `feature`, `citation`, and `note` fields. Currently emitted rules (all hand-curated, all with `support`, `rate`, `total_observations`, `alternatives` set to null):

| Tier | Rule | Confidence | Feature |
|---|---|---|---|
| word_position | `ŋ\|final` → `n` | high | -ING reduction |
| word_position | `ɹ\|final` → ∅ | medium | R-deletion (local Dublin non-rhoticity) |
| word_position | `iː\|final` → `[iː, ə]` | low | FLEECE vowel breaking (local Dublin) |
| word_position | `uː\|final` → `[uː, ə]` | low | GOOSE vowel breaking (local Dublin) |
| context_free | `θ` → `t` | high | TH-fortition (voiceless) |
| context_free | `ð` → `d` | high | TH-fortition (voiced) |
| context_free | `ɔː` → `ɑː` | medium | THOUGHT unrounding |
| context_free | `aɪ` → `ʌɪ` | low | PRICE centering (older Dublin) |
| context_free | `aʊ` → `ʌʊ` | low | MOUTH centering (older Dublin) |

**Joyce idiolect caveat:** Hickey distinguishes "local Dublin English" (popular/vernacular) from "supraregional Irish English" (educated/standard). Joyce was Dublin middle-class (Rathgar, b.1882) and Jesuit-educated — his idiolect almost certainly mixed features from both registers. Rules tagged `local` are upper bounds on what Joyce likely did; `supraregional` rules apply more confidently. This caveat is documented in the JSON's `metadata.joyce_idiolect_caveat`.

Features in the JSON's `not_implemented` block — most are real Dublin HE features that don't fit our format: T-lenition (no espeak symbol for the slit-T [t̪]), MEAT/MEET distinction (lexical, not segmental), yod retention in /tj dj nj/ (orthographic), WHICH-WITCH /hw/ preservation (orthographic), NURSE lexical incidence (lexically variable), the vowel-quality changes that accompany r-deletion (e.g. [fʊːst] for *first*).

**Confidence disclaimer:** the rule set was compiled from secondary sources by a non-specialist; chapter-level citations are best-effort and should be page-verified before publication. The artifact has **not** been reviewed by a Hiberno-English phonology specialist.

### `text`

This directory contains scripts that process the Wake itself, and produce our working corpus.

`scrape_trent.py` - Fetches all 619 text pages (3–628, minus 7 inter-book blank pages) from Wayback Machine
archives of the Trent/Groden edition. Each HTML page is a table where every row is one
printed line, with an explicit `FP=NNN.NN` link giving the 1939 page.line coordinate.
Applies all 311 text substitutions and 35 word-placement corrections from the FWEET typo
list (`data/annotations/fweet_corrections.json`) during scraping.

Output: `data/raw/book{NN}_ep{NN}.jsonl` per chapter (17 files), one JSON record per line.
Also writes `data/raw/corrections_applied.json` logging every correction made.

```
python scripts/preprocessing/text/scrape_trent.py
python scripts/preprocessing/text/scrape_trent.py --force     # re-fetch existing files
python scripts/preprocessing/text/scrape_trent.py --delay 2.0 # seconds between requests (~25 min total)
```

JSONL record schema (one per line):
```json
{"book": 1, "episode": 3, "page": 48, "line": 1, "page_line": "048.01", "text": "...", "left_margin": null, "right_margin": null}
```

For II.2 (pages 260–308), `left_margin` and `right_margin` are populated per line with
whatever margin text appears in that row; null where the margin is blank at that position.

Footnote body text is not in the Trent HTML. Inline `[N]` markers survive in `text`.
Run `scrape_footnotes.py` to populate the companion `_footnotes.json` files.

- `scrape_footnotes.py` - Extracts footnote body text from telelib.com (the only machine-readable source that has
it) and writes companion footnote files. Also applies 20 FWEET corrections that specifically target
footnote lines (page_line containing `F`, e.g. `262.F08`). Per-footnote location is recovered via
a two-stage lookup: footnote N → page (by scanning `[N]` markers in the Trent main text) → per-page
ordinal (parsed from the FWEET correction's `(in footnote #N)` elaboration).

Output: `data/raw/book{NN}_ep{NN}_footnotes.json` per chapter.
```json
{
  "book": 2,
  "episode": 2,
  "source": "telelib.com",
  "footnotes": {"1": "...", "2": "..."},
  "footnote_pages": {"1": 260, "2": 260, ...},
  "corrections_applied": [{"page_line": "262.F08", "status": "applied", ...}]
}
```

```
python scripts/preprocessing/text/scrape_footnotes.py
python scripts/preprocessing/text/scrape_footnotes.py --force
```

Footnote body text is resolved against the `[N]` refs in the JSONL `text` field via
`shared/corpus.load_footnotes(book, episode)`.

- `check_corpus.py` - Validates `data/raw/` against `data/manifest.json`. Checks line counts per chapter.
Exit 0 = OK, exit 1 = failures.

```
python scripts/preprocessing/text/check_corpus.py
```
- `lexical_stats.py` - Computes corpus statistics using whitespace tokenization only — no lowercasing, no
punctuation stripping, consistent with the preprocessing decisions below. Includes
`text`, `left_margin`, and `right_margin` fields; excludes footnotes.

```
python scripts/preprocessing/text/lexical_stats.py
python scripts/preprocessing/text/lexical_stats.py --top 30
python scripts/preprocessing/text/lexical_stats.py --json stats.json
```

- `parse_fweet.py` - Parses the FWEET elucidation HTML dump at `papers/FWEET_elucidation` (provided by Raphael Slepon) into structured JSONL at `data/raw/fweet_elucidations.jsonl`. Each row of the HTML table is one elucidation, keyed by page.line. Extracts: source-language tags (e.g. `_F_` French, `_G_` German, `_L_` Latin, `_I_` Irish Gaelic), motifs (`_M,HCE_`, `_M,ALP_`, etc.), person references (`_P,JS_` for Swift, etc.), bibliographic source references (e.g. `<OTGen>` for Genesis), sigla (e.g. `*E*`, `*A*`), and cross-references to other page.lines. Produces 101,354 elucidations across 21,452 page.lines (essentially every line of FW has at least one annotation).

```
python scripts/preprocessing/text/parse_fweet.py
```

- `train_tokenizers.py` - Trains the three families of unsupervised segmenters described in `shared/README.md`: BPE (byte-pair encoding) and Unigram LM at six vocab sizes each (1000, 2000, 4000, 8000, 16000, 32000), plus Morfessor Baseline and FlatCat. All trained from scratch on the Wake itself; no English-pretrained models. Output: `data/tokenizers/bpe_{vocab}.json`, `unigram_{vocab}.json`, `morfessor.bin`, `flatcat.bin`. SuperBPE is also wired but requires a parallel venv with the upstream fork.

```
python scripts/preprocessing/text/train_tokenizers.py
python scripts/preprocessing/text/train_tokenizers.py --only unigram --vocab-sizes 4000 8000
```

- `transcribe_ipa.py` - Runs espeak-ng G2P over the corpus tokens for a single language (default `en-us`) and writes one record per line at `data/ipa/{language}/book{NN}_ep{NN}.jsonl`. This produces a single-language baseline IPA artifact, distinct from `generate_joyce_derived_ipa.py` (in the audio directory), which produces the unified multi-language artifact with Joyce-rule overlay and FWEET-driven language muxing. The two scripts serve different purposes and emit different schemas; `transcribe_ipa.py` is useful when you want pure per-language G2P without Joyce or FWEET context.

```
python scripts/preprocessing/text/transcribe_ipa.py
python scripts/preprocessing/text/transcribe_ipa.py --language fr-fr
```

- `pos_tag.py` - **Multi-method POS hypotheses per token.** The Wake breaks single-tag POS conventions because nonce/portmanteau tokens often function as a noun in one reading and a verb in another. Standard POS tagging on the Wake therefore can't be *evaluated* in the usual sense (no gold standard, no shared notion of "correct" for nonce forms). Rather than pick one tagger and call its output the answer, this script runs four independent methods and emits the disagreement set as the artifact. No method is treated as authoritative.

  Methods (Stanza, UD tagset throughout):
  1. **`surface`** — Stanza English POS on the FW line, pre-tokenized so corpus boundaries are preserved. Reliable for function words and common English; word-shape guesses on nonces.
  2. **`fweet_source_form`** — Stanza POS in the *source language* on each FWEET source-form (e.g. `passencore` → French `pas encore` tagged as fr ADV+ADV; `mishe` → Irish `mise` tagged as ga PRON, Hebrew `Moshe` tagged as he X).
  3. **`fweet_gloss`** — Stanza English POS on each FWEET English gloss, one POS per gloss word.
  4. **`morpheme`** — Stanza English POS on each morpheme, one hypothesis per `(segmenter, morph)`. Segmenters: BPE 4k, Unigram 4k, Morfessor Baseline, FlatCat. The most experimental method — off-the-shelf taggers aren't designed for isolated morphemes, so output is largely word-shape inference. The segmenters also disagree with *each other* (e.g. `passencore` → Morfessor/FlatCat preserve `encore` as a unit, BPE/Unigram shatter it into `en`+`core` or `en`+`co`+`re`); that meta-disagreement is itself signal.

  **FWEET orthographic-match filter:** FWEET elucidations are *line-level*, not token-level. Naively attaching every line's elucidations to every token on the line produces noise (every word on FW 003.01 inherits *riverranno*, *rêverons*, etc.). To avoid this, `pos_tag.py` mirrors `generate_joyce_derived_ipa.py`'s confidence-for logic: a FWEET source-form is only attached to a specific token when the normalized edit distance between token and source-form is < 0.25 (high-confidence orthographic match). This drops the source-form attachment rate to ~2% of tokens in English-heavy chapters like I.1, and higher in the multilingual chapters; in exchange, the attachments mean something specific.

  Caveats baked into the script docstring:
  - Surface tags on nonce forms are predictably low quality (word-shape heuristics, contextual fallbacks). Emitted anyway — that's the methodological point.
  - Morpheme POS is the most experimental method. Treat as low-confidence by construction.
  - Stanza models for any FWEET language that fails to download log a warning and skip; the rest of the pipeline continues.

  Output: `data/pos/joyce-pos-hypotheses/book{NN}_ep{NN}.jsonl`, one record per token:
  ```json
  {
    "orth": "passencore",
    "page_line": "003.04",
    "hypotheses": [
      {"method": "surface", "language": "en", "input": "passencore", "pos": ["NOUN"]},
      {"method": "fweet_source_form", "language": "fr-fr", "input": "pas encore", "pos": ["ADV", "ADV"]},
      {"method": "fweet_source_form", "language": "fr-fr", "input": "passe encore", "pos": ["VERB", "ADV"]},
      {"method": "fweet_gloss", "language": "en", "input": "not yet (Motif: Not yet)...", "pos": ["PART", "ADV", "NOUN", "PART", "NUM"]},
      {"method": "morpheme", "segmenter": "morfessor", "language": "en", "input": "pass", "pos": ["VERB"]},
      {"method": "morpheme", "segmenter": "morfessor", "language": "en", "input": "encore", "pos": ["VERB"]},
      {"method": "morpheme", "segmenter": "bpe-4000", "language": "en", "input": "pass", "pos": ["VERB"]},
      {"method": "morpheme", "segmenter": "bpe-4000", "language": "en", "input": "en", "pos": ["PROPN"]},
      {"method": "morpheme", "segmenter": "bpe-4000", "language": "en", "input": "core", "pos": ["NOUN"]}
    ]
  }
  ```

  Stanza models: ~22 languages, ~3.6 GB total in `~/stanza_resources/`. Mounted into the Docker image at `/root/stanza_resources/` (same pattern as the HuggingFace wav2vec2 cache). First run downloads everything; subsequent runs are model-load only. Full-corpus run is ~30–40 min on the i9-9880H once models are cached.

```
python scripts/preprocessing/text/pos_tag.py                          # full corpus
python scripts/preprocessing/text/pos_tag.py --book 1 --episode 1     # one chapter (smoke)
```

  **Corpus-level results (all 17 chapters, 214,875 tokens):**

  | | Count | % of tokens |
  |---|---|---|
  | Total tokens | 214,875 | — |
  | Tokens with high-confidence FWEET source-form match | 4,747 | 2.21% |
  | Total FWEET source-form hypotheses (some tokens have ≥2) | 5,000 | — |
  | Tokens with attached FWEET gloss | 4,742 | 2.21% |
  | **Tokens with multi-method POS disagreement (≥2 distinct tags)** | **117,828** | **54.8%** |

  The 54.8% disagreement rate is the load-bearing finding: more than half the corpus has at least two methods producing different POS tags for the same token, which is the empirical justification for the multi-hypothesis design over committing to any single tagger.

  Top FWEET-source languages by unique high-confidence source-forms: la 794 / de 781 / fr 545 / it 478 / da 303 / nl 277. Per-chapter FWEET match rate ranges from 1.4% (I.2) to 2.9% (II.1) — tracks the multilingualism gradient in the Wake.

  Total artifact: 166 MB across 17 per-chapter JSONLs.

---

## Special Elements

### Mathematical and Greek Notation (in corpus)

| Symbol | Unicode | Name | Location |
|--------|---------|------|----------|
| ∞ | U+221E | INFINITY | II.2 line 284.11 |
| ∴ | U+2234 | THEREFORE | II.2 line 292.11 |
| ∵ | U+2235 | BECAUSE | II.2 line 292.12 |
| ουκ ελβον πολιν | U+03Bx | Greek letters | II.2 left_margin 269.24 |

### Sigla and Other Special Elements (annotated, not in corpus)

All entries sourced from the Hart Concordance (rosenlake.net) Symbols/Drawings/Music/Math/Greek
pages, cross-referenced with McHugh (1976). Coordinates use canonical 1939 page.line format.

| Chapter | Page.Line | Description | File | Status |
|---------|-----------|-------------|------|--------|
| I.1 | 018.36 | F pointing down, F pointing up | `book01_ep01_sigla.json` | Located: 018.36 — char_offset TBD |
| I.2 | 036.17 | Ǝ (upside-down backwards E) | `book01_ep02_sigla.json` | Located: 036.17 — char_offset TBD |
| I.2 | 044.25 | Ballad of Persse O'Reilly (sheet music) | `book01_ep02_sigla.json` | Completely absent (drawn image) |
| I.5 | 119.17 | Ǝ (HCE) | `book01_ep05_sigla.json` | Located: 119.17 — char_offset TBD |
| I.5 | 119.18 | △ (ALP) | `book01_ep05_sigla.json` | Located: 119.18 — char_offset TBD |
| I.5 | 121.03 | upside-down backwards F | `book01_ep05_sigla.json` | Located: 121.03 — char_offset TBD |
| I.5 | 121.07 | upside-down F | `book01_ep05_sigla.json` | Located: 121.07 — char_offset TBD |
| I.5 | 124.08 | arrow (→) | `book01_ep05_sigla.json` | Confirmed in corpus: 124.08 |
| I.5 | 124.09 | lambda (Λ) | `book01_ep05_sigla.json` | Located: 124.09 — char_offset TBD |
| I.5 | 124.10 | equals (=) | `book01_ep05_sigla.json` | Confirmed in corpus: 124.10 |
| II.2 | 266.22 | F, backwards F (Ⅎ) | `book02_ep02_sigla.json` | Located: 266.22 — char_offset TBD |
| II.2 | 269.24 | Greek (ouk elabon polin) | `book02_ep02_sigla.json` | Confirmed in corpus |
| II.2 | 272.09 | B C A D (left margin music) | `book02_ep02_sigla.json` | Completely absent |
| II.2 | 284.11 | ∞ (infinity) | `book02_ep02_sigla.json` | Confirmed in corpus |
| II.2 | 292.11 | ∴ (therefore) | `book02_ep02_sigla.json` | Confirmed in corpus |
| II.2 | 292.12 | ∵ (because) | `book02_ep02_sigla.json` | Confirmed in corpus |
| II.2 | 293.12 | Vesica Piscis diagram | `book02_ep02_sigla.json` | Completely absent (drawn image) |
| II.2 | 299.F4 | Full siglum set (footnote 188) | `book02_ep02_sigla.json` | Sequence Ш,△,⊣,✕,□,∧,⌐ — all 7 referents confirmed; Unicode approx uncertain |
| II.2 | 308.F1 | nose thumbing (drawing) | `book02_ep02_sigla.json` | Completely absent (drawn image) |
| II.2 | 308.F2 | crossed spoons (drawing) | `book02_ep02_sigla.json` | Completely absent (drawn image) |

**Note on char_offset TBD:** `char_offset` values for stripped sigla will be computed after the Trent corpus is scraped, by locating the gap pattern (space-before-punctuation, double-comma, or no-gap) within the specific Trent line. These are much shorter search targets than the old telelib section offsets (a single line vs. up to 9,000 characters).

---

## Preprocessing Decisions Made

The following are settled and should not be revisited without updating this document:

1. **Source edition: Trent University / Groden digital edition (c.1999), via Wayback Machine.**
   This is the standard edition for digital scholarship and the only machine-readable FW edition
   with line-level 1939 Faber & Faber page.line coordinates. Telelib was evaluated and rejected
   as a primary source (paragraph-level only, no pagination). The open-editions TEI corpus was
   evaluated and rejected (pandoc conversion with no `<pb>`/`<lb>` encoding, no page.line data).
2. **Corpus format: JSONL, one file per chapter, one JSON record per line.**
   One `.jsonl` file per chapter (`data/raw/book{NN}_ep{NN}.jsonl`). Each record: `{book,
   episode, page, line, page_line, text, left_margin, right_margin}`. TEI XML was considered and
   rejected for NLP use; JSONL is streamable, grep-able, and HuggingFace-compatible.
3. **FWEET corrections applied during scraping.** 355 known errors in the Trent text from the
   [FWEET typo list](http://www.fweet.org/pages/fw_typo.php) were classified and saved to
   `data/annotations/fweet_corrections.json`. 311 simple text substitutions and 35 word-placement
   shifts are applied automatically by `scrape_trent.py`; 9 complex range fixes are stored for
   manual review.
4. **Citation coordinates: `page_line` field (`"NNN.NN"`) on every record.** This is the
   canonical 1939 Faber & Faber citation format used in all Wake scholarship (e.g., Hart 1963,
   McHugh 1980). All annotation overlays are keyed by `page_line`.
5. **Footnote references:** Preserved inline as `[N]` markers in `text`. Footnote body text is
   not in the Trent pages. Companion files `data/raw/book{NN}_ep{NN}_footnotes.json` (scraped
   from telelib.com via `scrape_footnotes.py`) hold the body text. Load via
   `shared/corpus.load_footnotes(book, episode)`, which returns `{}` if the file is absent.
6. **No Unicode normalization.** No lowercasing. No stemming. No punctuation stripping.
7. **Mathematical and Greek symbols are unique tokens.** ∞, ∴, ∵, and Greek letters in II.2
   must not be normalized, substituted, or stripped. They are structurally meaningful and
   intentionally chosen by Joyce.
8. **Printed sigla are annotated via `data/annotations/`, not injected into the corpus.**
   Drawn sigla were printed at 9 locations across I.1, I.2, I.5, and II.2 of the 1939 edition
   (confirmed by Hart Concordance Symbols page and McHugh 1976 pp. 133–134). They are stripped
   in the Trent corpus as they were in telelib. Tracked in per-chapter annotation overlay files
   keyed by page.line. Load annotations via `shared/annotations.py`.

---

## Corpus Statistics

Computed on the Trent corpus (21,479 lines, 17 chapters). Tokenization: whitespace-split, no lowercasing, no punctuation stripping. Line-break hyphens rejoined across consecutive lines (last token of a line ending in `-` is prepended to the first token of the next line; `--` em-dashes excluded). Includes `text`, `left_margin`, and `right_margin` fields; excludes footnotes.

| Statistic | Trent value | Notes |
|-----------|-------------|-------|
| Total tokens | 216,072 | Whitespace-split, line-break hyphens rejoined (typographic hyphen dropped) |
| Vocabulary (unique types) | 71,326 | cf. Hart (1963): 63,924 — difference reflects no-normalization policy |
| Type/token ratio | 0.3301 | High relative to standard English (~0.05–0.15) |
| Hapax legomena | 59,474 | 83.4% of vocabulary types appear exactly once |
| Shannon entropy | 12.1294 bits/token | Token-level; Drożdż et al. report 8.51 bits at **character** level — different measurement |

---

## Open Preprocessing Questions

These have not been decided. Each project should decide for itself unless a shared
preprocessing script is warranted:

- **Sentence segmentation:** How do you split the Wake into sentences? Standard
  sentence tokenizers fail. Options: punctuation-based heuristics, fixed-length windows,
  line-level units (the natural unit of the Trent corpus). Most projects will use BPE
  tokenizers operating at sub-word level rather than sentence-level segmentation. [ANSWER: sentence-level segmentation is for future projects, we will cross that bridge when we get to it, depending on the results of preprocessing for phonological and morphological tasks]
- **Language tagging:** FWEET has per-passage language glosses in 60+ languages. Should
  language tags be projected onto tokens? This is relevant to project 04 and 07. [ANSWER: we have already decided this when doing our IPA transcription. language of origin is relevant for both phonological and morphological analysis, so yes.]
- **Thunderword handling:** The ten 100-letter thunderwords span multiple printed lines.
  Projects should decide whether to treat them as atomic, split them, or exclude them. [ANSWER: atomic]
- **Portmanteau decomposition:** The Hart Concordance includes a syllabification guide and
  "Overtones" section listing 10,000 English words suggested by Joycean distortions. This
  could be the basis for a decomposition layer — but it is interpretive, not ground truth. [ANSWER: we will stick with NLP morphological segmentation analysus]


---

## Existing Computational Work

| Project | Method | Finding |
|---|---|---|
| JoyceNet (p-Mart) | 2-layer character LSTM | Produces coherent-seeming gibberish; boundary between Joyce and machine output is philosophically ambiguous |
| Count Bayesie (2015) | Character RNN | Same conclusion; character-level required |
| FinneGAN | GAN on audiobook | Learns phonetic/semantic patterns; no stable tokenization |
| Curry (2024) | TF-IDF + t-SNE + k-means | Mathematical structure "almost indistinguishable from a multifractal" despite apparent chaos |
| Drożdż et al. (2024) | Punctuation statistics | Decreasing hazard function, unique in literature |
| Drożdż et al. (2025) | Multifractal analysis | Translation-invariant structure across 5 languages |

## Digital Editions

| Resource | Notes |
|---|---|
| **Trent/Groden edition** (our source, via Wayback Machine) | 1939 F&F; line-level with page.line coords; II.2 three-column structure preserved |
| **telelib.com** (previous source) | 1939 F&F; paragraph-level only; no pagination; footnote text preserved |
| **James Joyce Digital Archive** (jjda.ie) | Genetic editions with all draft layers; useful for compositional intent |
| **open-editions/corpus-joyce-finnegans-wake-tei** (GitHub) | TEI XML; pandoc conversion; no `<pb>`/`<lb>` encoding — evaluated and rejected as corpus source |
| **Hart Concordance** (rosenlake.net) | 63,924 unique word forms with occurrence counts; Symbols/Math/Music/Greek pages used for annotations |
| **FWEET** (finneganswake.org) | Searchable elucidation database with 60+ language glosses per passage |

---

## Key References

- Hart, Clive. *A Concordance to Finnegans Wake*. University of Minnesota Press, 1963.
  Searchable at rosenlake.net/fw/FWconcordance/.
- Drożdż, Kwapień & Stanisz. "Statistics of punctuation in experimental literature —
  The remarkable case of Finnegans Wake." *Chaos* 34(8):083124, 2024.
- Drożdż, Kwapień & Stanisz. "Punctuation Patterns in Finnegans Wake by James Joyce Are
  Largely Translation-Invariant." *Entropy* 27(2):177, 2025.
- Curry, Brian James. "Mapping the Unmappable: The Hidden Mathematics of James Joyce's
  Finnegans Wake." Medium, 2024.
- Shen, Emily. "A Dream Before the Dawn of the Digital Age? Finnegans Wake, Media, and
  Communications." Harvard SEAS, 2020.
- "Neologizing in Finnegans Wake: Beyond a Typology of the Wakean Portmanteau." Academia.edu.
- McHugh, Roland. *The Sigla of Finnegans Wake*. University of Texas Press, 1976. Available on Archive.org.
- James Joyce Digital Archive: jjda.ie
- FWEET: finneganswake.org
- TEI corpus: github.com/open-editions/corpus-joyce-finnegans-wake-tei
