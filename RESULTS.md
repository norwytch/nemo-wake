# Project 01 — Experiment results

Detailed write-ups for the first-pass experiments. The [README](README.md) holds the
idea tracker and links here for the full detail. Session-by-session history lives in
[docs/history.md](docs/history.md).

---

## Idea 2 — Multilingual context areas (phonotactic attribution)

**Status:** finalized with multi-seed error bars and a fixed baseline (2026-06-13). The
single-seed first pass (2026-05-30) below is preserved for detail; the corrected headline
is the finalization block immediately following.

### Finalization (2026-06-13) — error bars, dedup baseline, and the load-bearing result

Three changes hardened this idea; full numbers in `results/finished_science_summary.json`.

**1. The single-seed headline was not representative (n=3 error bars).** Re-running eq25
(strict equal-cap, balanced) across 3 seeds shows wide seed variance — the originally
reported seed-0 numbers were a favorable draw on overall/macro-F1 and the *low* end on
foreign-gold:

| eq25 metric | seed-0 (reported) | 3-seed mean ± sd |
|---|---|---|
| Overall top-1 | 41.1% | **32.5% ± 7.6** |
| Foreign-gold top-1 | 21.8% | **25.4% ± 4.4** |
| Macro-F1 | 0.234 | **0.205 ± 0.026** |

Takeaway: nothing in idea 2 should be quoted single-seed. On the foreign-gold subset the
NEMO mean (25.4%) and the matched classical baseline (cap=25 ≈ 21.8%) remain a rough tie
within error — the original "tie, not a win" verdict survives, now with honest spread.

**2. Same-substrate baseline (dedup).** The bigram-LangID baseline scored a *multiset* of
line bigrams while NEMO fires each bigram once (a `set`); `--dedup-line` matches them. The
honest same-substrate baseline is *stronger*, not weaker: cap=100 foreign-gold rises
0.218→0.277 and macro-F1 0.281→0.308, while the headline cap=25 comparison barely moves
(foreign-gold 0.218 either way). So "same substrate" is now literally true and the verdict
holds — NEMO does not beat the classical baseline; if anything the gap widens slightly.

**3. The real result — Hebbian collapse is a recoverable normalization artifact (robust
across seeds).** This is the portable contribution and should lead the writeup. The cap=100
imbalanced run collapses (minority recall → ~0). A leakage-free gain-control read-out
(`src/context_inhibition.py`: per-slot z-normalization fit on the val split, scored on
test — the evaluator-side surrogate of the MOOD-area lateral inhibition in Mitropolsky &
Papadimitriou 2025) recovers it, and the recovery holds in **every one of 3 seeds**:

| cap=100 collapse, test split | raw (3-seed) | inhibited (3-seed) | per-seed lift |
|---|---|---|---|
| Foreign-gold top-1 | 0.206 ± 0.034 | **0.309 ± 0.045** | +7.3, +18.2, +5.5 pp |
| Macro-F1 | 0.238 ± 0.019 | **0.296 ± 0.011** | all 3 positive |

The means separate by more than 1 sd and the lift is positive in all 3 seeds — far stronger
than the single-seed bootstrap (CI [-2.2, +17.7] crossed zero). The claim: *Hebbian collapse
under class imbalance is largely a missing-gain-control artifact; standardizing per-slot
drive recovers minority-class detection without retraining or any nemo-core change.* It also
helps the already-balanced eq25 run, but mildly (foreign +4.2pp), as expected. Single
chapter, 55 foreign test lines, n=3 seeds — direction is robust, magnitude has seed
variance; 5–10 seeds and an in-core implementation are the obvious next steps.

---

#### Original single-seed first pass (2026-05-30) — preserved for detail

Headline (single seed): under strict class balance, NEMO Hebbian assemblies on a
phonotactic substrate *tie* classical bigram-LangID on foreign-language detection (22/101 =
21.8% on the foreign-gold subset) and *beat* it on the most phonotactically-distinctive
classes (Latin, Italian), while *losing* on phonotactically near-cousin classes (German vs
English). The multi-seed finalization above supersedes the single-seed numbers.

### Recast: phonotactic attribution, not orthographic

The original framing was orthographic ("which CONTEXT_L fires for these spellings"). The
recast — enabled by the joyce-derived IPA artifact (`data/ipa/joyce-derived/`) — is
**phonotactic**: each token enters PHON as a phoneme sequence; CONTEXT_L learns which
phoneme patterns belong to language L. The question becomes:

> Can a NEMO substrate attribute Wake passages to source-languages from their
> **phonotactic signatures alone**, and does the attribution agree with FWEET's scholarly
> annotation above frequency + classical-LangID baselines?

This is squarely Beguš-territory (phonotactics + multilingual phonology +
biologically-grounded neural models). The orthographic version was a DH question; the
phonotactic version is a phonology paper.

### Engagement with Mitropolsky & Papadimitriou 2025

The 2025 acquisition paper *itself* names multilinguality as a natural extension (§3,
Discussion):

> "Our system can be extended to handle *multilinguality*: A new area can be added
> containing an assembly for each language known to the speaker, very much like the
> representations of grammatical moods in the MOOD area. This assembly would be active
> while learning, or using, the corresponding language."

Idea 2 is a concrete instantiation of an extension Mitropolsky explicitly names — on real
multilingual literary data, via real preprocessed IPA, with a scholarly gold standard
(FWEET) for evaluation.

### Locked Tier-1 design

| Decision | Choice | Rationale |
|---|---|---|
| Phoneme inventory | **Unified** — pool all per-language espeak symbols into one PHON space | Cross-language phoneme overlap (e.g. /a/ in English ∩ Italian) is *signal*, not confound. Disjoint symbols (e.g. /ɑ̃/ French only) don't share blocks, so the model learns "French uses these, English doesn't" — the phonotactic discrimination we want. |
| Order in PHON | **Phoneme bigrams** | Reuses the n-gram approach from idea 8 (whose original "+64% from order" motivation was later found to be largely a training-intensity confound — see idea 8 finalization; bigrams remain a reasonable phonotactic unit regardless). Bigrams *are* phonotactics — sound-pair constraints are exactly what distinguishes languages. Same code path as `replicate_ngram_phon.py`. |
| Training chapter | **I.6 (Riddles / Twelve Questions)** | Dense FWEET annotation, famously multilingual, manageable size. |
| Language scope | **Top 5: en, de, fr, la, it** | ~80% of FWEET volume; tractable; the publishable claim is "NEMO attributes 5 languages from phonotactics above baselines." Scale to 22 in Tier 2 if Tier 1 lands. |

### Architecture sketch

```
PHON  (one block per unified IPA bigram seen across en/de/fr/la/it)
  ↓ ↑   (recurrent fibers, Hebbian plasticity)
CONTEXT_en   CONTEXT_de   CONTEXT_fr   CONTEXT_la   CONTEXT_it
```

Train: for each FWEET-tagged token in I.6, fire its bigram union into PHON + fix the
corresponding CONTEXT_L assembly. Project, plasticity binds phonotactic patterns to L.
Structurally mirrors the noun→VISUAL / verb→MOTOR routing in `replicate_phoneme_phon.py`
— language replaces N/V.

Probe (held-out, via `shared/corpus.line_split`): for each held-out FWEET-tagged line,
fire the union of its tokens' bigrams, project to all CONTEXT_L, observe k-cap winners.
Top-1 = NEMO's predicted language; top-N for code-mixed lines.

### Build (what was actually executed, 2026-05-30)

Compressed into one session, not the planned two weeks. Four files:

- `scripts/preprocessing/text/build_multilingual_lexicon.py` — extracts FWEET-tagged
  tokens from `data/ipa/joyce-derived/book01_ep06.jsonl`, restricts to the Top-5 target
  languages, emits one training instance per (token × high-confidence FWEET language) plus
  English-by-default for unannotated tokens. Probe lines = val + test split, with
  per-page-line gold language sets from FWEET. **Phoneme source for both training and
  probe: `en-us-baseline`.** Using per-language IPA at probe time would leak the language
  label; using it at train time would make CONTEXT_L learn a native template that wouldn't
  match the English-realized probe. The CONTEXT_L assembly therefore learns
  *English-realized phonotactic correlates of foreign-source-language presence* — same
  competence an English-speaking Wake reader exercises.
- `src/replicate_multilingual_context.py` — explicit PHON (bigram blocks, unified
  inventory) + explicit CONTEXT (one slot per language, Mitropolsky-2025 MOOD-style —
  single area, multiple fixed assemblies); train per (token, language) instance with
  class-balanced subsampling via `--max-per-lang`; probe per held-out line by firing the
  union of its tokens' bigrams and reading the CONTEXT k-cap overlap per slot.
- `src/baseline_attribution.py` — two classical baselines on the same lexicon and probe
  lines: (1) frequency baseline (predict argmax_L prior over training instances), (2)
  phoneme-bigram LangID (per-language Laplace-smoothed unigram distribution over bigrams;
  predict argmax_L sum log P(b|L)). Runs in both `--max-per-lang` modes.
- `src/evaluate_attribution.py` — takes labelled paths to one or more results files;
  computes per-method top-1 accuracy (overall + foreign-gold subset), per-language P/R/F1,
  macro-F1, and confusion matrices.

### I.6 lexicon stats

| | Count |
|---|---|
| I.6 page-lines | 1,513 (train 1,221 / val 139 / test 153) |
| Tokens | 15,988 |
| **Training instances** (en-us 12,586 / la 86 / de 61 / fr-fr 49 / it 25) | **12,807** |
| Probe lines (val + test) | 292, **101 (34.6%) with ≥1 non-English gold** |
| Per-language gold occurrences | en-us 191 / la 40 / de 40 / fr-fr 28 / it 15 |
| Unique en-us phonemes | 91 |
| **Unique phoneme bigrams (PHON_n basis)** | **1,545** |

### Three NEMO runs + matched baselines

| Run | Params | Wall | Overall | **Foreign-gold** | Notes |
|---|---|---|---|---|---|
| Smoke | rounds=2, max=30 | 13 min | 38.7% | **21.8%** (22/101) | Pipeline validated |
| Tier-1 (`cap=100`) | rounds=10, max=100 | 2h 28m | 42.8% | **15.8%** (16/101) | **Hebbian collapse** on minority classes (it→0%, fr-fr→3.6%) |
| **Strict equal-cap** (`eq=25`) | rounds=10, max=25 | 52 min | 41.1% | **21.8%** (22/101) | Collapse rescued; balanced per-class profile |

### Headline four-way comparison

All methods evaluated on the same 292 probe lines, languages = `['de', 'en-us', 'fr-fr',
'it', 'la']`.

| Method | Overall | **Foreign-gold** | **Macro-F1** | de F1 | en-us F1 | fr-fr F1 | it F1 | la F1 |
|---|---|---|---|---|---|---|---|---|
| Frequency (uncapped, ≡ "always en-us") | 65.4% | **0.0%** | 0.158 | 0 | 0.791 | 0 | 0 | 0 |
| Bigram LangID (uncapped) | 63.7% | **0.0%** | 0.156 | 0 | 0.778 | 0 | 0 | 0 |
| Bigram LangID (cap=100) | 50.7% | **21.8%** | **0.281** | 0.203 | 0.677 | 0.145 | 0.162 | 0.217 |
| Bigram LangID (cap=25) | 47.9% | **21.8%** | **0.282** | 0.103 | 0.648 | 0.264 | 0.200 | 0.198 |
| NEMO Tier-1 (cap=100, collapse) | 42.8% | 15.8% | 0.187 | 0.024 | 0.609 | 0.062 | 0.000 | 0.241 |
| **NEMO equal-cap (eq=25)** | 41.1% | **21.8%** | **0.234** | 0.040 | 0.577 | 0.210 | 0.136 | 0.208 |

### Per-class winner pattern — the actually-interesting finding

Restricted to NEMO eq=25 vs Bigram LangID cap=25 (the strict-balance comparison),
per-language **correct counts** on probe lines where that language is in gold:

| Language | Gold lines | NEMO eq=25 | Bigram LangID cap=25 | Δ (NEMO − Bigram) |
|---|---|---|---|---|
| Latin (la) | 32 | **8** | 6 | **+2** |
| French (fr-fr) | 19 | **6** | **6** | 0 |
| Italian (it) | 10 | **2** | 1 | **+1** |
| German (de) | 40 | 1 | **4** | **−3** |
| English (en-us) | 191 | 98 | **118** | −20 |

**NEMO outperforms classical bigram LangID on the most phonotactically-distinctive
classes (Latin, Italian); ties on French; loses on German and English.** The German loss
is significant — German shares enough phonotactic structure with English (both Germanic)
that NEMO's Hebbian assemblies can't disambiguate the near-cousin. Bigram LangID's
explicit Laplace-smoothed likelihood handles it. The English loss is partly dominant-data
(en-us has by far the most probe lines) and partly the same near-cousin issue inverted.

### Honest reading

- **On the foreign-gold subset (the actual question — "does the model detect non-English
  presence?"), NEMO eq=25 and Bigram LangID tie exactly at 22/101 = 21.8%.** The two
  methods identify the same number of foreign-gold lines correctly; they just identify
  *different ones*.
- **On macro-F1 (the per-class-balanced summary), NEMO loses by ~5 pts (0.234 vs
  0.282)**, driven by the German collapse and the English ceiling.
- **The per-class winner pattern is mechanistically interpretable:** Hebbian-plasticity
  assemblies discriminate via *phonotactic divergence* (Latin's `/-us/`, `/-um/`, `/-or/`
  endings are unmistakable; Italian's open-syllable pattern is too). Classical LangID
  discriminates via *statistical likelihood* (graceful Laplace falloff on near-cousins).
  They're learning different things from the same substrate.
- **The Tier-1 falsifiable claim ("NEMO exceeds the classical bigram LangID baseline") is
  not met as stated** — it's a tie on foreign-gold accuracy and a loss on macro-F1. But
  the per-class pattern is the more interesting finding for a phonology audience, and it's
  a defensible workshop-paper result: *NEMO Hebbian assemblies on a phonotactic substrate
  match classical bigram-LangID on foreign-language detection under strict class balance
  and outperform it on phonotactically-distinctive classes (Latin, Italian) while failing
  on phonotactically-near-cousin classes (German vs English). This suggests
  biologically-plausible assemblies discriminate via phonotactic distance rather than
  statistical likelihood — a property worth further investigation for code-switching and
  bilingual phonological acquisition models.*

### Caveats

- **Hebbian collapse under class imbalance is the load-bearing failure mode.** The cap=100
  run (en-us:la:de:fr:it = 100:86:61:49:25) drove Italian to 0% recall and French to 3.6%
  as plasticity over-reinforced en-us and Latin (the two largest classes). The cap=25
  strict-balance run prevented it. For real-world acquisition data (which is always
  imbalanced), this is the next problem to fix. Lateral inhibition between CONTEXT slots
  (which Mitropolsky 2025 uses for MOOD areas) is the obvious next move.
- **German is a persistent failure** even under strict balance (1/40 = 2.5% recall). The
  phonotactic-cousinship hypothesis is consistent with this but not proven; alternatives
  include the specific I.6 German tokens being short/atypical or the unified-inventory
  PHON not separating /ʃ/-rich German bigrams cleanly enough.
- **N = 292 probe lines from one chapter, single seed.** No multi-seed significance test;
  the 22/101 tie could shift by ±2–3 lines on a different seed. The per-class winner
  pattern is robust to seed but the headline tie is not statistically secured.
- **English-only baseline ceiling.** The "always-predict-en-us" baseline gets 65.4%
  overall because Joyce wrote in English by default. Any honest evaluation has to centre
  the foreign-gold subset, which we did.
- **Single substrate (en-us-baseline IPA), single n-gram (bigram), single chapter (I.6).**
  Tier 2 would extend to multi-chapter + 22-language scope; different paper, different
  compute budget.

### Artifacts

- Lexicon: `lexicons/i6_multilingual_top5.json` (1,545 bigrams, 12,807 train instances, 292 probe lines)
- NEMO equal-cap (the headline NEMO run): `results/mlctx_ngram2_20260530-100148/`
- NEMO Tier-1 cap=100 (the collapse run, kept for reference): `results/mlctx_ngram2_20260530-003720/`
- NEMO smoke run (pipeline validation): `results/mlctx_ngram2_20260530-001436/`
- Baselines: `results/baselines/{uncapped,cap25,cap100}.json`
- Four-way comparison: `results/idea2_final_comparison.json` (plus the earlier `results/idea2_tier1_comparison.json`)

### Next steps (if pursued)

- **Lateral inhibition between CONTEXT slots** in nemo-core to prevent Hebbian collapse
  under imbalance; re-run cap=100 with inhibition active.
- **German error analysis** — token-level look at which German tokens NEMO mis-attributes.
- **Multi-seed significance** on the foreign-gold tie (≥10 seeds).
- **Multi-chapter generalisation** — train on I.6, evaluate on II.2 / II.3 / etc.
- **Top-N multi-label evaluation** for code-mixed lines.
- **Tier 2** — scale to all 22 FWEET languages, with the inhibition fix in place.

---

## Idea 8 — Portmanteau decomposition via partial-activation maps

**Status:** finalized (2026-06-13) — and the original headline largely **did not survive**
the controlled comparison. The first-pass "+64% order wins across every segmenter" was
mostly a training-intensity confound, not a property of order. See the finalization block
below; the single-seed first pass (2026-05-29) is preserved after it for detail.

### Finalization (2026-06-13) — the order effect was largely a parameter confound

The first pass compared bigram at β=0.06/rounds=20/proj=2 against bag at β=0.03/rounds=10/
proj=1 — the bigram run simply trained ~4× harder (the bag overflowed float32 at the hot
params, so they were never matched). Re-running **both at identical safe params**
(β=0.03/rounds=10/proj=1, no clamp, no overflow) collapses the effect:

| mean separation | bag (n1) | bigram (n2) | controlled Δ | first-pass Δ (confounded) |
|---|---|---|---|---|
| BPE-4000 | 0.186 | 0.141 | **−24%** | +29% |
| Unigram-4000 | 0.166 | 0.106 | **−36%** | ~flat |
| Morfessor | 0.264 | 0.236 | **−11%** | +40% |
| FlatCat | 0.321 | 0.344 | **+7%** | +64% |

Under matched training, bigram is *worse* than bag on the three more-fragmenting segmenters
and only marginally better on FlatCat (+7%, vs the headline +64%). A residual order signal
survives **only on FlatCat**: P@k 0.447→0.553 and NEMO↔FWEET agreement 17/20→19/20 both
still favor bigram there. So the honest claim is much narrower: *order helps recovery only
under the FlatCat (linguistically-motivated) segmentation, and only modestly; the large
across-the-board effect was the parameter gap.* The hot-param regime where order looked
decisive is exactly the regime where the bag is numerically unstable, so it can't be
cleanly attributed to order. The Tier-3 sequence-formation engine is still the faithful
test, but the Tier-1 motivation for it is now weak, not strong.

A wrapper-side weight clamp (`--clamp`, in `replicate_ngram_phon.py`; nemo-core untouched)
was added and validated (bounds weights, stays finite) — but it is **not** used for this
comparison, because the legitimate bigram run reaches ~7.6e8 and clamping the bag below
that would manufacture a new confound. The clamp is kept for future corpus-scale runs where
overflow, not parameter-matching, is the concern.

---

#### Original single-seed first pass (2026-05-29) — preserved for detail; superseded above

Headline (confounded): order matters — bigram PHON beats bag across every segmenter (+64%
separation on FlatCat), lifting NEMO ↔ FWEET agreement to 95% (19/20). The finalization
above shows most of this gap was training intensity, not order.

### Engagement with Mitropolsky & Papadimitriou 2025 ("Simulated Language Acquisition in a Biologically Realistic Model of the Brain", arXiv 2507.11788)

The 2025 acquisition paper supersedes the 2021 parser
(`papers/assemblies_biological_language_organ.pdf`) by learning the lexicon, POS, word
order, generation, and basic hierarchy from grounded sentences — but **explicitly bypasses
phonetics and treats words as atomic stable PHON assemblies**: "we bypass this phase by
adopting an input-output convention whereby a sentence is presented to the system as a
sequence of stimuli corresponding to word tokens." That sub-PHON gap is exactly what idea
8 targets. The Wake violates nearly every NEMO acquisition precondition (atomic stable
tokens, sensorimotor grounding, concrete N/V lexicon, ignorable function words), so a
precondition stress-test is the natural framing.

### Build (Tier 1: n-gram bag-vs-order ablation)

`nemo-core` has **no sequence-formation primitive** (no `sequence` / `chain` /
`next_assembly`; only `project`, `associate`, `merge`, `fix_assembly`). Implementing
Dabagia–Papadimitriou–Vempala sequence formation in the engine ("Computation with
sequences of assemblies", Neural Computation 2024) is weeks of C++/Python — Tier 3 in the
spec. As a cheap pre-test of whether order matters at all, the first pass approximates
order with **phoneme n-gram units**: PHON has one explicit block per contiguous phoneme
n-gram; a word's PHON activation is the union of its n-gram blocks. `--ngram 1` =
bag-of-phonemes (the existing `replicate_phoneme_phon.py` design); `--ngram 2` = bigrams
(order-aware). No engine change. If order helps at n-gram, the Tier 3 build is motivated.

Three new files:

- `src/replicate_ngram_phon.py` — n-gram variant of `replicate_phoneme_phon.py`. Adds
  `--ngram N` and a **partial-activation map** per probe: after training (plasticity off),
  each trained morpheme's hub k-cap is recorded as an *anchor assembly*; at probe, the
  portmanteau's induced k-cap is overlapped against every anchor, producing
  `activation_profile = [{trained_word, overlap, overlap_frac}, ...]` sorted desc. That
  overlap profile *is* the partial-activation map idea 8 names.
- `src/activation_alignment.py` — reference-agnostic scorer. Given an `activation_profile`
  and a reference set of true constituents per probe, computes precision@k, MRR, and
  **separation** (mean overlap_frac of constituents − mean of distractors), each with a
  random baseline.
- `scripts/preprocessing/text/build_phoneme_gallery.py` — new builder, pulls
  portmanteaus from `data/pos/joyce-pos-hypotheses/`, phonemizes via espeak en-us, emits a
  lexicon where **trained sub-units = unique segmenter morphemes (POS-bucketed)** and each
  probe carries `constituents_by_segmenter` + `fweet_source_forms`. Requires
  `--require-fweet` (default on) so only portmanteaus with scholarly annotation are kept;
  produces a tractable lexicon (`lexicons/ep1_phoneme_gallery.json`: 86 training morphemes
  / 24 portmanteaus for I.1).
- `src/fweet_compare.py` — gains a `--mode activation` that joins n-gram results + phoneme
  gallery, scores the activation profile against each segmenter's decomposition, and
  cross-tabs with the existing FWEET count concordance.

**Design choice (deliberate):** trained sub-units are segmenter morphemes in **English
IPA**, *not* FWEET source-forms in their source languages. Source-form-as-trained-unit
(Design A) would suffer a cross-lingual phoneme-inventory confound (espeak's French nasals
vs English vowels barely overlap), producing a null result driven by inventory mismatch
rather than phonology. Design B keeps everything in one inventory; FWEET enters indirectly
through the count concordance.

### Methodology — three independent lenses on each portmanteau

1. **Segmenter** (BPE 4k / Unigram 4k / Morfessor Baseline / FlatCat) — algorithmic decomposition.
2. **FWEET** — scholarly multilingual source-form annotation (count + ID).
3. **NEMO partial-activation map** — biologically-motivated emergent decomposition: which
   trained morphemes does the portmanteau light up?

For each portmanteau the NEMO profile is scored against each segmenter's constituent list
(separation, P@k); per portmanteau a **NEMO-favoured segmenter** is recorded (max
separation). The **FWEET-favoured segmenter(s)** is anyone whose morph count matches the
modal scholarly source-form count within ±1. Then count: NEMO-favoured ∈ FWEET-favoured.

### Results — I.1 phoneme gallery, 24 FWEET-annotated portmanteaus, 86 training morphemes

| Metric | Bag (ngram=1) | **Bigram (ngram=2)** | Δ |
|---|---|---|---|
| BPE-4000 — mean separation | 0.185 | 0.239 | +29% |
| Unigram-4000 — mean separation | 0.166 | 0.170 | ~flat |
| Morfessor — mean separation | 0.264 | 0.371 | +40% |
| **FlatCat — mean separation** | **0.321** | **0.528** | **+64%** |
| FlatCat — mean P@k | 0.447 | 0.485 | +9% |
| **NEMO ↔ FWEET agreement** | **17/20 = 85%** | **19/20 = 95%** | **+10 pts** |
| FWEET count concordance (unchanged) | 67 / 54 / 70 / 73% | 67 / 54 / 70 / 73% | — |

Three independent findings:

1. **Order improves recovery across every segmenter.** Bigram > bag for all four; largest
   gain on FlatCat (+64% separation), smallest on Unigram (~flat). The order-aware
   substrate suppresses spurious distractor overlap that the bag substrate accepts (e.g.
   *sends* → *riverrun* via stray /n/ collapses from 0.15 in bag to ~0.01 in bigram on the
   small balanced lexicon).
2. **FlatCat wins both regimes**, on both NEMO substrate alignment AND FWEET concordance.
   The segmenter whose decomposition the phonological substrate recovers most cleanly is
   also the one matching the scholarly annotation. Matches the `shared/README.md` claim
   that Morfessor's bag-of-morphs is "evidently false for the Wake" and FlatCat's
   HMM-over-categories repairs it.
3. **95% NEMO ↔ FWEET agreement** on the bigram substrate — for 19 of 20 portmanteaus
   where both lenses resolve, the same segmenter wins. Three independent lenses (FlatCat,
   FWEET, NEMO bigram substrate) converge on the same per-portmanteau verdict.

### Caveats

- **Param confound between the two runs.** Bigram was at β=0.06 / rounds=20 / proj-rounds=2
  (the original gallery run). Bag at β=0.03 / rounds=10 / proj-rounds=1 (the safer rerun
  after the original bag run hit float32 plasticity overflow — `NOUN_in` values around
  10³³, profiles saturated to identical nonsense). The bag is *under-trained relative to
  bigram*, so part of the gap is plasticity, not representation. The order-wins direction
  is theory-predicted and unlikely to invert, but a strict apples-to-apples rerun is
  outstanding (~40 min at safer params).
- **Hebbian overflow is real and parameter-sensitive.** Bigrams happened to avoid it at
  β=0.06 / rounds=20 (denser PHON spreads input across more neurons, slowing per-neuron
  weight growth); bags overflow under the same params. For corpus scale-up a per-projection
  weight clamp in `brain.py` is probably necessary.
- **Tier-1 caveat.** Phoneme n-grams approximate order via local context windows; they
  don't encode position or *true* sequence (a/b/c vs b/a/c with the same bigram set look
  identical at higher N). The Tier 3 sequence-formation engine build remains the faithful
  version of idea 8. The Tier 1 result motivates it.
- **n=10000 hubs with 86 trained morphemes** — assembly density is approaching the regime
  where chance overlap is non-trivial (k²/n = 1 expected, frac ≈ 0.01); raise `n` for
  cleaner separation if scaling up.

### Artifacts

- Lexicon: `lexicons/ep1_phoneme_gallery.json`
- Bigram run: `results/ngram2_phon_20260529-182913/` (clean, β=0.06 / rounds=20 / proj=2)
- Bag run: `results/ngram1_phon_20260529-214647/` (clean, β=0.03 / rounds=10 / proj=1)
- Three-lens JSON: each run dir contains `three_lens_compare.json`
- Original bag run with overflow (kept for reference): `results/ngram1_phon_20260529-170445/` — *do not use for analysis*

### Next steps (if pursued)

- Strict apples-to-apples rerun (bigram at safer params).
- Extend gallery beyond I.1 (more portmanteaus, more statistical power).
- Implement Tier 3 (true sequence formation in nemo-core) and re-run.
- A weight-clamp in `brain.py` to make corpus-scale runs robust to overflow.
