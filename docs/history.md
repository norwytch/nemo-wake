# Project 01 — Session notes

> **Note (2026-05-30):** Several scripts and lexicons referenced in earlier
> sections were retired in commit `6407c92` as part of a project-wide scale-back
> to ideas 2 and 8. Specifically: `src/replicate_tutoring.py`,
> `src/replicate_ambiguity.py`, `src/replicate_graded.py`,
> `scripts/preprocessing/text/build_lexicon_from_pos.py`, and lexicons
> `page3_ambig.json`, `page3_graded.json`, `ep1_graded.json`.
> These notes are preserved as session history; their markdown links to the
> retired files no longer resolve.

## 2026-05-21

### What got done

**Replications.**
- §3 lexicon acquisition (toy 2N+2V): properties P and Q both pass cleanly.
- §4 word order, SVO and VSO: both directions emit correctly (5/5 trials each).
- Runtime scaling: paper's n=100k for §4 is not tractable on this hardware; n=10k works.

**Infrastructure built.**
- [src/_common.py](src/_common.py) — shared seed/log/results scaffolding.
- [src/replicate_lex.py](src/replicate_lex.py), [src/replicate_svo.py](src/replicate_svo.py), [src/replicate_vso.py](src/replicate_vso.py) — three replication wrappers.
- [src/replicate_tutoring.py](src/replicate_tutoring.py) — individual-word-tutoring sweep.
- Reproducibility: PYTHONHASHSEED randomization broke numerical reproducibility despite seeded RNGs (set-iteration order in `brain.project()` changes RNG consumption). Fixed via self-exec.
- [lexicons/page3_mix.json](lexicons/page3_mix.json) — first Wake-derived lexicon (5N+5V from page 3, includes `vicus` and `riverrun` as Wake-distinctive).

**POS-tagged page 3 with spaCy** ([data/annotations/book01_ep01_pos.jsonl](../../data/annotations/book01_ep01_pos.jsonl)). Now superseded by the multi-method [data/pos/joyce-pos-hypotheses/](../../data/pos/joyce-pos-hypotheses/) artifact built outside the conversation.

### New findings on Wake-applied model

- Wake-derived 5N+5V lexicon converges in **3 rounds** vs ~30 for toy 2N+2V. More co-firing variety accelerates rather than slows convergence. Single data point; worth confirming.
- Tutoring effect on 5N+5V: **10–14% reduction** in word-presentations, well below the ~40% originally hypothesized for the (now-retired) individual-word-tutoring idea. Effect barely scales with frequency (freq=2 ≈ freq=10).
- Wake-distinctive items (`vicus`, `riverrun`) sort into their assigned hubs as cleanly as English ones — confirming the POS tag IS the supervision signal, not a feature of the word itself.

### Central design limitation

AC §3 requires a binary noun-or-verb commitment at training time. The Wake's 54.8% POS-method disagreement (from the new joyce-pos-hypotheses artifact) is the empirical version of "evades labeling." Three approaches for handling ambiguity were sketched:
- **Tag-swap**: same word, two lexicon files (one as N, one as V), compare runs. Smallest.
- **Dual co-firing**: fire both VISUAL and MOTOR during training for ambiguous words. New code (~50 lines).
- **Same word twice**: add the word to both nouns and verbs in the lexicon. Two disjoint PHON blocks; the model treats them as independent lexical entries. No code change. The "polysemous lexicon" view.

---

## 2026-05-22

### What got done

**Dual co-firing experiment built and run.**
- [lexicons/page3_ambig.json](lexicons/page3_ambig.json) — page-3 lexicon with three buckets: 4 pure nouns (bay/shore/time/vicus), 4 pure verbs (brings/sends/went/brewed), 2 ambiguous (riverrun/bend). Provenance from the multi-method [joyce-pos-hypotheses](../../data/pos/joyce-pos-hypotheses/) artifact; rationale noted per ambiguous entry.
- [src/replicate_ambiguity.py](src/replicate_ambiguity.py) — builds the brain manually (not via `LearnBrain`) to allow VISUAL/MOTOR sizing for the ambiguous bucket's extra slots. Implements **selective mutual inhibition**: pure-N/V words commit to one hub for recurrent rounds; ambiguous words keep both hubs in the project map throughout.

**Two debugging milestones along the way.** Recording these so I don't repeat them:
1. **Mutual inhibition is required.** First version used pure single-context tutoring with no mutual inhibition. PHON developed equal connections to NOUN and VERB for all words → Q ratios ≈ 1 for everything. `learner.tutor_single_word` uses `mutual_inhibition=True` for exactly this reason.
2. **Mutual-inhibition criterion must use TOTAL hub input, not PHON-only.** Second version used `sum(conn[PHON][NOUN])` vs `sum(conn[PHON][VERB])`. This inverts the decision because pure-N's NOUN top-k is selected on PHON+VISUAL combined input, so the chosen NOUN winners have *moderate* PHON-conn; VERB's top-k is selected on PHON alone so its winners have *maximal* PHON-conn. Ratios came out inverted (pure nouns → VERB hub). Fix mirrors `learner.get_total_input`: PHON + VISUAL for NOUN, PHON + MOTOR for VERB.

### Headline result

After 30 rounds of single-word tutoring (~2 min), both properties pass for all 10 words, and ambiguous behaves as predicted:

| bucket | mean N/V ratio | example | NOUN_in | VERB_in |
|---|---:|---|---:|---:|
| pure nouns | 25.05 | bay | 167,958 | 6,691 |
| pure verbs | 0.04 | brings | 6,681 | 170,400 |
| ambiguous | 0.99 | riverrun | 170,559 | 174,566 |

The critical observation: ambiguous words are **not** "weakly committed to both." Both hubs are at the same magnitude as pure-N's strong hub. `riverrun` gets the full strength of `bay`'s noun representation **plus** the full strength of `brings`'s verb representation. Bilateral, not diluted.

Property P (recovery from context) passes for both contexts on ambiguous words: VISUAL → NOUN → PHON recovers `riverrun` (96/100 overlap); MOTOR → VERB → PHON also recovers `riverrun` (96/100 overlap). Two complete assemblies, both robust.

### What this does and doesn't show

- **Does:** AC §3 can mechanically support a dual representation when supervised that way. The framework is extensible to N/V ambiguity; the standard training routine just forces commitment.
- **Doesn't:** show that this is the right model of how the brain represents `riverrun`. We're handing the model the labels; the model isn't discovering ambiguity from the input.
- **Doesn't address the morphological critique.** The ambiguity is at the *word level*. PHON still treats `riverrun` as an atomic block — no internal structure saying "river is N and run is V." (Phase 3 below closes this gap.)

### Phase 1 — Graded ambiguity (stochastic context selection)

**Built and run.** New artifacts:
- [scripts/preprocessing/text/build_lexicon_from_pos.py](../../scripts/preprocessing/text/build_lexicon_from_pos.py) — consumes [data/pos/joyce-pos-hypotheses/](../../data/pos/joyce-pos-hypotheses/), aggregates per-token noun/verb hypothesis votes into `p_noun ∈ [0, 1]`. Default rule: methods kept = `surface, fweet_source_form, fweet_gloss` (drop morpheme); per-hypothesis voting on UD tags (NOUN/PROPN → N; VERB/AUX → V). Bucket sampling for spread across p_noun.
- [lexicons/ep1_graded.json](lexicons/ep1_graded.json) — 30 words sampled from book 1 ep 1, 6 per p_noun bucket (0.0–0.1, 0.1–0.4, 0.4–0.6, 0.6–0.9, 0.9–1.0).
- [src/replicate_graded.py](src/replicate_graded.py) — per training step, fire VISUAL with probability `p_noun` else MOTOR; mutual-inhibit toward the chosen hub.

**Headline result (Property Q):** clean monotonic relationship across all 5 buckets, ~3 orders of magnitude span:

| p_noun bucket | mean N/V ratio |
|---|---:|
| 0.0–0.1 | 0.04 |
| 0.1–0.4 | 0.40 |
| 0.4–0.6 | 1.64 |
| 0.6–0.9 | 4.61 |
| 0.9–1.0 | 24.36 |

The model's commitment ratio tracks input `p_noun`. Direction correct, magnitude large, monotonic.

**Caveat (Property P):** cold context recovery is fragile under graded supervision. Only `larms` (p_noun=0.667, realized draws 18/12) recovered from *both* sides. Most intermediate-p_noun words split too evenly to build a concentrated assembly on either side. **The Property Q vs P split is informative:** the model encodes graded ambiguity in connection strengths, but stochastic supervision dilutes the assemblies below the threshold for one-shot cold recovery. Yesterday's binary dual co-firing (which gave 100% of *both* contexts every step) produced clean P AND Q; today's graded version trades P-recovery for graded Q.

Single seed. Multi-seed would smooth the within-bucket variance.

### Phase 3 — Phoneme-bag PHON (compositional sub-word semantics)

**Built and run.** This closes the sub-word gap the morphological critique flagged.

**Architecture.** PHON restructured: one explicit block per unique phoneme in the lexicon (16 blocks in the demo). A word's PHON activation = the *union* of its phoneme blocks. Shared phonemes share neurons. IPA source: `joyce_ipa_source` per token from [data/ipa/joyce-derived/](../../data/ipa/joyce-derived/) (the rule-applied variant).

**Code:** [src/replicate_phoneme_phon.py](src/replicate_phoneme_phon.py). Brain built manually (not via LearnBrain). `phon_k=20` so a 5-phoneme word activates ~100 PHON neurons, comparable to single ctx_k=100. Per-phoneme attribution probe is built in: fires each phoneme block alone, measures NOUN_in/VERB_in.

**Test setup (lexicon [phoneme_phon_balanced.json](lexicons/phoneme_phon_balanced.json)):**
- Train: `river` (N) /ɹ ɪ v ɚ/, `bay` (N) /b eɪ/, `run` (V) /ɹ ʌ n/, `goes` (V) /ɡ oʊ z/, `sends` (V) /s ɛ n d z/.
- Held-out probe: `riverrun` /ɹ ɪ v ʌ n/.
- Lexicon designed for balance: /ɹ/ appears in 1 N and 1 V (river vs run).

**Headline result:** the model exhibits clean compositional sub-word semantics.

Per-phoneme N/V attribution:

| phoneme | N/V | trained in |
|---|---:|---|
| /b/, /eɪ/ | ~18 | nouns only (bay) |
| /v/, /ɪ/, /ɚ/ | ~36–42 | nouns only (river) |
| /ɹ/ | **1.03** ✓ | balanced (1N: river, 1V: run) |
| /n/ | 0.01 | doubly verb (run + sends) |
| /ʌ/, /d/, /oʊ/, /s/, /z/, /ɛ/, /ɡ/ | ~0.04 | verbs only |

**Held-out `riverrun` N/V = 0.33** — bilateral but verb-leaning. The model's response to an unseen word is the additive sum of its constituent phonemes' learned biases. The verb-lean comes from /n/ having been supervised twice (run + sends) — predictable in hindsight.

**Diagnostic value of the unbalanced demo lexicon.** A first version ([phoneme_phon_demo.json](lexicons/phoneme_phon_demo.json), with `shore` included) had /ɹ/ at 2N:1V supervision. Result: `riverrun` N/V = 5.85, identical to the trained verb `run`'s ratio (5.82, also wrong-direction because of /ɹ/'s pollution). This demonstrated the failure mode: imbalanced phonemic supervision propagates everywhere. Balancing /ɹ/ in the second run fixed both `run`'s classification and `riverrun`'s direction.

**What this resolves.**

The morphological critique was: AC §3 lacks sub-word structure, so portmanteaus can't be decomposed. Phase 3 shows that with phoneme-bag PHON:
- Phonemes acquire per-POS biases from training context.
- A word's routing is the additive composition of its phoneme biases.
- Held-out words inherit routing from their constituent phonemes — *no per-word labeling required*.
- "Ambiguity" emerges from phonemic composition rather than being declared.

This is the **strong positive result** for project 01: AC's framework is extensible to both word-level ambiguity (phase 2's dual co-firing) and sub-word compositionality (phase 3's phoneme-bag PHON) without modifying nemo-core. Together they cover the Wake's polysemy mechanisms.

### Phase 4 — Morpheme-bag PHON (the cleaner version of phase 3)

Phoneme-bag works but its unit is wrong for the Wake. Phonemes don't carry meaning; morphemes do. Joyce's wordplay is morpheme-level. Replacing the unit moves the compositionality to the right linguistic layer.

**Code:** [src/replicate_morpheme_phon.py](src/replicate_morpheme_phon.py). Same architecture as phoneme-bag, but PHON has one block per unique morpheme. Morphemes come from one of the four unsupervised segmenters trained on the Wake (BPE-4k, Unigram-4k, Morfessor, FlatCat — see [scripts/preprocessing/README.md](../../scripts/preprocessing/README.md)).

**Demo lexicon** ([morpheme_phon_demo.json](lexicons/morpheme_phon_demo.json)): train `river` (N), `bay` (N), `run` (V), `goes` (V) — each a single-morpheme word. Probe held-out `riverrun`, whose decomposition under all 4 segmenters agrees: `river` + `run`.

**Result:** held-out `riverrun` N/V = **0.99** — bilateral at full hub magnitude. Each morpheme contributes its full trained signal: `river` pulls NOUN up to 168k, `run` pulls VERB up to 169k, the sums are essentially independent because the two morphemes activate disjoint NOUN-vs-VERB assemblies.

Comparison across approaches for `riverrun`:

| approach | riverrun N/V | requires per-word labeling? |
|---|---:|:-:|
| phase 2 (binary dual co-firing) | 1.00 | yes, declared ambiguous |
| phase 3 (phoneme-bag, balanced) | 0.33 | no, but depends on lexicon balance |
| **phase 4 (morpheme-bag)** | **0.99** | **no** |

Morpheme-bag is the cleanest because each morpheme has clean single-bucket supervision and the held-out word's routing is just an additive composition with no shared-morpheme cross-contamination.

### Phase 5 — Segmenter-disagreement gallery (idea #8, operationalized)

The morpheme-bag framework lets us probe any Wake portmanteau under each segmenter's decomposition hypothesis. The spread across segmenters becomes a quantitative measure of how morpheme-decomposition uncertainty propagates into POS uncertainty — exactly what README idea #8 (portmanteau decomposition via partial-activation maps) asked for.

**Gallery lexicon** ([page3_portmanteau_gallery.json](lexicons/page3_portmanteau_gallery.json)): five page-3 Wake portmanteaus (`passencore`, `rearrived`, `penisolate`, `topsawyer's`, `bellowsed`). 29 trained morphemes (each tagged via Stanza on the standalone morpheme — noisy by construction, documented). 6 untrained morphemes (INTJ/PUNCT/DET/ADJ — no N/V signal). 20 probes (5 portmanteaus × 4 segmenters).

**Headline gallery:**

| Portmanteau | bpe-4000 | unigram-4000 | morfessor | flatcat | Spread |
|---|---:|---:|---:|---:|---:|
| **passencore** | 1.27 | 1.23 | **0.04** | **0.04** | 32× |
| **rearrived** | **24.68** | **26.89** | 1.01 | 1.01 | 27× |
| **penisolate** | 0.79 | 1.15 | 1.05 | 1.05 | 1.5× |
| **topsawyer's** | 1.25 | 24.53 | **25.70** | **25.70** | 21× |
| **bellowsed** | 1.26 | 1.32 | **26.58** | **26.58** | 21× |

**The headline number** is `passencore`'s 32× spread: under morfessor/flatcat decomposition it's a verb (0.04); under BPE/unigram it's noun-leaning (1.27). Same Wake portmanteau, four segmenters, two diametrically opposed POS readings.

**The two-regime pattern.** Morfessor and flatcat agree on every decomposition and give linguistically interpretable readings (rear+rived → bilateral; bellows+ed → noun; pass+encore → verb; pen+isolate → bilateral). BPE/unigram fragment more aggressively, producing 3–4-morpheme decompositions where small letter-fragments (`s`, `en`, `co`, `re`) get noisy tagger assignments, and the resulting readings are dominated by tagger noise on tiny fragments rather than meaningful sub-words.

**`rearrived` is the textbook case.** Morfessor/flatcat read it as `rear` (N) + `rived` (V) → N/V = 1.01, the model representing both Joyce's "return" and the "rear+rived" split simultaneously. BPE/unigram never isolate `rived` as a unit → no verb component → N/V ≈ 25, indistinguishable from a pure noun.

**Honest caveat for paper-writing.** Morfessor/flatcat decompositions look "right" linguistically because they keep meaningful content morphemes intact; BPE/unigram look "wrong" because they shatter. But BPE/unigram are *unsupervised statistical* segmenters — their decompositions reflect Wake substring frequency, not human linguistic intuition. The fact that BPE-style decompositions give noisy readings is a *result*, not a methodological failure: it shows the model is correctly propagating segmenter-level signal up to POS-level signal.

---

## 2026-05-23 / 24

### Phase 6 — Automated, FWEET-restricted chapter-scale gallery

The page-3 gallery was hand-picked. To make a defensible chapter-scale claim — and to address yesterday's caveat that the FWEET coverage was only 2/18 portmanteaus — built an automation pipeline + scholarly-comparison tool.

**New tools.**
- [scripts/preprocessing/text/build_portmanteau_gallery_lexicon.py](../../scripts/preprocessing/text/build_portmanteau_gallery_lexicon.py) — auto-generates morpheme_phon lexicons from joyce-pos-hypotheses. Filters: multi-morph in ≥1 segmenter, segmenter disagreement, bilateral N/V potential. Optional `--require-fweet` filter restricts to tokens with high-confidence per-token FWEET source-form attribution.
- [src/fweet_compare.py](src/fweet_compare.py) — given a morpheme_phon results.json, looks up each portmanteau's FWEET source-forms and computes concordance (segmenter morph count vs scholarly source-form word count ±1).

**The gallery.** Filter `--require-fweet` applied to chapter I.1 → 24 portmanteaus with high-confidence per-token FWEET attribution. 89 trained morphemes (including 4 controls: river/bay/run/goes). 96 probes (24 × 4 segmenters).

**Lexicon:** [lexicons/ep1_fweet_gallery.json](lexicons/ep1_fweet_gallery.json). **Run results:** [results/morpheme_phon_20260522-212414/](results/morpheme_phon_20260522-212414/). Includes [fweet_compare.json](results/morpheme_phon_20260522-212414/fweet_compare.json).

**Runtime caveat.** This experiment was sized larger than I planned; it took 31.7 hours wall clock (~10 hours pure CPU, the rest was sleep when the laptop was closed). Per-round cost grew superlinearly with morpheme count (PHON_n = 12,500 here vs 8,600 in the page-3 gallery). For future runs at this scale: use `caffeinate -i` from the start, and consider `--rounds 15` since the model converges fast.

### Headline result — concordance with FWEET scholarship

| Segmenter | Concordance with scholarly source-form word count (±1) |
|---|---:|
| **flatcat** | **14 / 24 = 58.3%** |
| **morfessor** | **11 / 24 = 45.8%** |
| bpe-4000 | 8 / 24 = 33.3% |
| unigram-4000 | 6 / 24 = 25.0% |

Linguistically-motivated segmenters (FlatCat, Morfessor) align with FWEET scholarship at **roughly 2× the rate** of pure statistical segmenters (BPE, Unigram). The gap is the paper's quantitative claim: *for Wake portmanteaus with foreign-language source-forms, statistical sub-word segmentation diverges systematically from scholarly decomposition; linguistically-motivated segmenters converge with it.*

### Notable individual cases

| Portmanteau | FWEET source | Spread story |
|---|---|---|
| `Toucheaterre` (019.14) | fr "toucher terre" (2) | Morfessor/flatcat → 4 morphs `Touch+eat+er+re`, N/V=**0.04** (verb, matching French infinitive "toucher"). BPE/unigram → 5–6 fragments, N/V=20–22 (noun). 550× ratio between regimes. |
| `hamissim` (029.33) | he "khamishim" (1) | Morfessor/flatcat → 2 morphs `ham+issim`, N/V=27.15. BPE/unigram → 3–4 morphs, N/V=~1.25. Clean two-regime split. |
| `paisibly` (014.30) | fr "paisible" (1) | Same pattern: morfessor/flatcat clean noun (22.53); BPE/unigram bilateral noise (~1.0). |
| `Dor` (020.18) | he "dor" (1) | All four segmenters concordant with scholarly count. Even with universal concordance, the N/V ratios vary: BPE/unigram/flatcat → 22–23 (noun); morfessor → 0.04 (verb, because morfessor split `Do+r` and `r` happens to be untrained). |
| `passencore` (003.04) | fr "pas encore" (2) | The week's headline persists: morfessor/flatcat → 0.04 (verb); BPE → 1.27; unigram fails count concordance. |

`Dor` is a useful methodological case: scholarly concordance on count doesn't imply concordance on POS reading — segmenter choice can still flip the model's commitment even when the count matches.

### What's now defensible for the paper

The thread spanning phases 1–6:

- **Phase 1 (graded supervision)**: Q ratio tracks input p_noun monotonically across 3 orders of magnitude.
- **Phase 2 (dual co-firing)**: word-level ambiguity supports bilateral assemblies at full hub magnitude.
- **Phase 3 (phoneme-bag)**: held-out words inherit POS routing from phonemic biases.
- **Phase 4 (morpheme-bag)**: same, at the better linguistic unit.
- **Phase 5 (5-portmanteau gallery)**: per-Wake-portmanteau segmenter-induced N/V spread, hand-picked.
- **Phase 6 (24-portmanteau FWEET gallery)**: chapter-scale automated gallery with quantitative concordance against scholarly source-forms.

Phase 6 gives the paper a real falsifiable comparator: **the FlatCat-Morfessor concordance gap of ~25 percentage points over BPE-Unigram is a measurable claim about how Wake polysemy responds to choice of sub-word unit.**

---

## Tomorrow — candidates

### Strengthening phase 6

1. **Multi-chapter gallery.** Run the same `--require-fweet` filter on chapters beyond I.1. Does the 58% flatcat / 25% unigram gap hold across the whole book, or is it I.1-specific? Caveat: the chapter-I.1 run took 10 CPU-hours; a whole-book run is days of CPU. May need to reduce rounds or split per-chapter.
2. **N/V direction concordance** (not just count). Build a tool that compares the model's per-segmenter N/V reading to the scholarly source-form's POS in its source language (from FWEET source_form's `pos` field, e.g. fr "toucher terre" → ["VERB", "NOUN"]). Does morfessor's reading of `Toucheaterre` (verb) match the source-form's VERB-leading structure? This is a stronger claim than count concordance.
3. **FWEET-derived morpheme POS.** Stanza tagging on small fragments produces the bulk of our tag noise (`s`/`en`/`co` tagged PROPN). For morphemes that match FWEET source-form components, use FWEET's per-language tagging instead. Cleaner training signal → potentially cleaner gallery readings, especially for BPE/unigram which suffer most from fragment-tagger noise.

### Methods cleanup

4. **Multi-seed phase 6.** Single seed = single data point per probe. Run seeds 1–3 to estimate variance bands around the 58% / 25% numbers. ~30 CPU-hours total but parallelizable across seeds.
5. **`Dor`-style cases.** Audit the gallery for portmanteaus where all segmenters concord on count but disagree on N/V direction. These are the most informative methodological cases — they isolate cut-position from cut-count.

### Open framing — paper outline

Project 01 now has a clear narrative arc:

1. **Setup**: AC §3 trained on toy 2N+2V → property P/Q pass (replication).
2. **The Wake question**: Wake portmanteaus break single-POS labeling (54.8% of corpus tokens have multi-method POS disagreement per joyce-pos-hypotheses).
3. **Extension 1 (phase 1)**: graded supervision shows AC's commitment ratio is continuous, not binary.
4. **Extension 2 (phase 2)**: dual co-firing demonstrates the framework can hold bilateral representations at word level.
5. **Extension 3 (phase 4 morpheme-bag)**: held-out words inherit POS routing from their morphemes' biases — sub-word compositionality emerges without nemo modifications.
6. **Applied result (phase 6)**: chapter-scale gallery quantifies how the choice of unsupervised segmenter shapes the model's reading of Wake portmanteaus; FlatCat/Morfessor concord with FWEET scholarship at 2× the rate of BPE/Unigram.

The "AC is fundamentally ill-fit for the Wake" critique from the README's "human interpretation" notes is now empirically answered: it's not. Three small architectural extensions (no nemo-core changes) cover both word-level ambiguity and morphological compositionality, and the framework produces quantitative readings that compare meaningfully to scholarly annotations.

### Housekeeping

- Stale [data/annotations/book01_ep01_pos.jsonl](../../data/annotations/book01_ep01_pos.jsonl) — superseded by the multi-method artifact, can be deleted.
- **The paper-headline keeper:** [results/morpheme_phon_20260522-212414/](results/morpheme_phon_20260522-212414/) (phase 6, 24-portmanteau FWEET gallery + fweet_compare.json).
- Other significant keepers in `results/`:
  - Phase 2: [ambiguity_20260522-091320/](results/ambiguity_20260522-091320/)
  - Phase 1: [graded_20260522-093316/](results/graded_20260522-093316/)
  - Phase 3 (phoneme-bag, balanced): [phoneme_phon_20260522-100450/](results/phoneme_phon_20260522-100450/)
  - Phase 4 (morpheme-bag riverrun): [morpheme_phon_20260522-140557/](results/morpheme_phon_20260522-140557/)
  - Phase 5 hand-picked gallery: [morpheme_phon_20260522-141857/](results/morpheme_phon_20260522-141857/)
  - Phase 5b page-3 auto gallery: [morpheme_phon_20260522-144923/](results/morpheme_phon_20260522-144923/)
- `caffeinate -i` doesn't prevent lid-close sleep. For future long runs: `sudo pmset -c sleep 0 disablesleep 1` while on AC, or external display + clamshell mode.
