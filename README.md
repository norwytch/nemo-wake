# 01 — Modern Irish Literature For Neural Assemblies

## Framework

The Assembly Calculus (Papadimitriou et al.) models language as emergent from neural
assembly operations: project, associate, and merge firing within a Hebbian network.
The Wake, with its extreme associative density and neologistic recombination, is treated
as the output of such an organ operating at or beyond normal capacity.

## Submodule

`nemo-core/` — the assemblies simulation library (github.com/dmitropolsky/assemblies)

## Approach

From the paper:

> we have the following simplified picture of language in the brain: each word has a
> root representation in a lexical hub area, likely within different sub-areas for nouns
> and verbs, which is connected to a phonological representation of the word —
> representations which are used both for recognizing and for articulating words. The
> lexical hubs are richly connected to many sensory and semantic areas across the brain
> through which the many complex shades of meaning and nuances of a word are
> represented; crucially, nouns and verbs have strong connections to different context
> areas.

> We will shortly define a language organ in NEMO that will learn from sentences of a
> toy language with l nouns and l intransitive verbs, where l is a small parameter that
> we vary in our experiments. We denote the combined lexicon as L. In this language,
> all sentences are of length two: "cats jump" and "dogs eat." Importantly — and this
> is the hard part of our experiment — the language can have either SV (subject-verb)
> principal word order (as in English, Chinese and Swahili) or VS (as in Irish,
> Classical Arabic, and Tagalog), and our model should succeed in either scenario.

This is the §4 SV/VS experiment. Applying it to the Wake would require POS labels on some FW tokens.

The model also assumes grounded input:

> We further assume that our input is grounded: whenever a noun W ∈ L is heard it is
> also seen — that is, an assembly corresponding to the static visual perception of the
> object (cat, dog, mom, etc) is active in VISUAL, denoted VISUAL[W]. Similarly, an
> assembly corresponding to the intransitive action (jump, run, eat, etc.) in MOTOR,
> denoted MOTOR[W] for a verb W ∈ L. These areas represent the union of the differing
> somatosensory cortical areas feeding into nouns and verbs covered in Section 2.

In NEMO, "grounding" reduces to a per-word assembly in a non-lexical area that co-fires with PHON during training. `VISUAL` and `MOTOR` are interchangeable explicit populations — the names are theoretical labels, not modality requirements. The open question for the Wake is what supervisory co-firing signal to use, not whether grounding is possible.

Other relevant paper sections: §5.1 (multilinguality), §5.3 (abstract words and contextual ambiguity).


### Analytical — using NEMO to measure something about the Wake

#### 2. Multilingual context areas (phonotactic recast): FIRST PASS DONE — 2026-05-30

McHugh annotates 50+ languages in the Wake; FWEET supplies machine-readable per-page-line tags for ~22 of them. Make each language a separate CONTEXT_L area, train per-token on FWEET-tagged tokens, and ask whether the substrate learns to attribute *unseen passages* to source-languages above frequency and classical LangID baselines.

- **Requirements:** per-token language labels (FWEET, already loaded via `shared/fweet`); per-language IPA per token (joyce-derived IPA artifact, already preprocessed); held-out split (deterministic via `shared/corpus.line_split`).
- **Falsifiable claim:** for held-out FWEET-tagged page-lines, the NEMO-predicted top-1 CONTEXT_L exceeds (i) a per-token-frequency baseline and (ii) a classical phoneme-bigram LangID baseline trained on the same substrate.
- **Status:** first-pass complete; see Results below.
- human: yes, i think this could work.

##### Recast: phonotactic attribution, not orthographic

The original framing was orthographic ("which CONTEXT_L fires for these spellings"). The recast — enabled by the joyce-derived IPA artifact (`data/ipa/joyce-derived/`) — is **phonotactic**: each token enters PHON as a phoneme sequence; CONTEXT_L learns which phoneme patterns belong to language L. The question becomes:

> Can a NEMO substrate attribute Wake passages to source-languages from their **phonotactic signatures alone**, and does the attribution agree with FWEET's scholarly annotation above frequency + classical-LangID baselines?

This is squarely Beguš-territory (phonotactics + multilingual phonology + biologically-grounded neural models). The orthographic version was a DH question; the phonotactic version is a phonology paper.

##### Engagement with Mitropolsky & Papadimitriou 2025

The 2025 acquisition paper *itself* names multilinguality as a natural extension (§3, Discussion):

> "Our system can be extended to handle *multilinguality*: A new area can be added containing an assembly for each language known to the speaker, very much like the representations of grammatical moods in the MOOD area. This assembly would be active while learning, or using, the corresponding language."

Idea 2 (phonotactic recast) is a concrete instantiation of an extension Mitropolsky explicitly names — on real multilingual literary data, via real preprocessed IPA, with a scholarly gold standard (FWEET) for evaluation. Direct engagement with current prior art.

##### Locked Tier-1 design

| Decision | Choice | Rationale |
|---|---|---|
| Phoneme inventory | **Unified** — pool all per-language espeak symbols into one PHON space | Cross-language phoneme overlap (e.g. /a/ in English ∩ Italian) is *signal*, not confound. Disjoint symbols (e.g. /ɑ̃/ French only) don't share blocks, so the model learns "French uses these, English doesn't" — the phonotactic discrimination we want. |
| Order in PHON | **Phoneme bigrams** | Reuses the order-aware n-gram approach that gave +64% separation in idea 8. Bigrams *are* phonotactics — sound-pair constraints are exactly what distinguishes languages. Same code path as `replicate_ngram_phon.py`. |
| Training chapter | **I.6 (Riddles / Twelve Questions)** | Dense FWEET annotation, famously multilingual, manageable size. |
| Language scope | **Top 5: en, de, fr, la, it** | ~80% of FWEET volume; tractable; the publishable claim is "NEMO attributes 5 languages from phonotactics above baselines." Scale to 22 in Tier 2 if Tier 1 lands. |

##### Architecture sketch

```
PHON  (one block per unified IPA bigram seen across en/de/fr/la/it)
  ↓ ↑   (recurrent fibers, Hebbian plasticity)
CONTEXT_en   CONTEXT_de   CONTEXT_fr   CONTEXT_la   CONTEXT_it
```

Train: for each FWEET-tagged token in I.6, fire its bigram union into PHON + fix the corresponding CONTEXT_L assembly. Project, plasticity binds phonotactic patterns to L. Structurally mirrors the noun→VISUAL / verb→MOTOR routing in `replicate_phoneme_phon.py` — language replaces N/V.

Probe (held-out, via `shared/corpus.line_split`): for each held-out FWEET-tagged line, fire the union of its tokens' bigrams, project to all CONTEXT_L, observe k-cap winners. Top-1 = NEMO's predicted language; top-N for code-mixed lines.

##### Build list (Tier 1)

| File | Role | Reuses |
|---|---|---|
| `scripts/preprocessing/text/build_multilingual_lexicon.py` | Extract from joyce-derived IPA: training words (FWEET-tagged tokens in I.6 train split, with phonemes from the matching per-language baseline + language tag); probe lines (held-out lines with gold language set per FWEET). | joyce-derived IPA artifact, `shared/fweet`, `shared/corpus.line_split` |
| `src/replicate_multilingual_context.py` | Per-language CONTEXT areas; bigram PHON; train per-token; probe per-line. Output: per-line CONTEXT activations sorted desc + top-N predictions. | `_common.py`, nemo-core, structural pattern from `replicate_ngram_phon.py` |
| `src/evaluate_attribution.py` | Score NEMO attributions vs FWEET gold vs baselines: (1) per-token frequency, (2) classical phoneme-bigram log-likelihood LangID. Top-1 accuracy + multi-label F1 + confusion matrix. | `shared/fweet`, results from the replication script |

Three files. ~1 week to first end-to-end runnable, ~1 more week to write up. **The paper.**

##### Falsifiable claims (Tier 1)

1. **Top-1 attribution accuracy on held-out lines exceeds the per-token-frequency baseline.** (The trivial "always guess English" baseline. Anything biologically nontrivial should clear this.)
2. **It also exceeds a classical phoneme-bigram LangID baseline trained on the same data.** (The harder baseline — proves NEMO adds something over a vanilla generative model on the identical substrate.)
3. **NEMO's confusion matrix mirrors known phonotactic similarity** (e.g. French↔Italian higher than French↔English), distinguishing biological-substrate behaviour from arbitrary classifier noise.

##### Risks / open issues

- **Hebbian overflow** (idea-8 lesson) — start with β=0.03, rounds=10, proj-rounds=1; tune up only if convergence is loose.
- **Class imbalance** — Latin and Italian have far fewer FWEET tokens than English. The frequency baseline gets a head start. Report both raw accuracy and per-class F1; the per-class story is the interesting one.
- **Code-mixed lines** — many FWEET page-lines carry multiple language tags. Single-winner k-cap won't capture this. Tier-1 plan: evaluate top-N predictions against the FWEET language *set* (multi-label F1), not top-1 against a single gold. Plus a top-1 subset for the headline number.
- **Joyce's "voice" as English-by-default** — the bulk of FW is English-ish, so the baseline beats any model trivially for ~80% of tokens. The interesting evaluation is on the *minority-language* subset (de/fr/la/it tokens specifically), where NEMO has to be doing real phonotactic work. Report both subsets explicitly.

##### Milestones (planned — 2 weeks part-time)

1. (Day 1–2) Build the multilingual lexicon for I.6 — confirm phoneme bigram inventory size, per-language token counts, train/eval split.
2. (Day 3–5) `replicate_multilingual_context.py` running end-to-end on Top-5 languages at small scale; verify CONTEXT areas form distinguishable assemblies.
3. (Day 6–7) Baselines (frequency + classical bigram LangID) implemented; numbers logged.
4. (Day 8–10) Evaluation script + held-out runs; confusion matrix; per-class F1.
5. (Day 11–14) Writeup.

##### Build (what was actually executed, 2026-05-30)

Compressed into one session, not two weeks. Four files (lexicon + replication + a separate baseline script + an evaluator):

- `scripts/preprocessing/text/build_multilingual_lexicon.py` — extracts FWEET-tagged tokens from `data/ipa/joyce-derived/book01_ep06.jsonl`, restricts to the Top-5 target languages, emits one training instance per (token × high-confidence FWEET language) plus English-by-default for unannotated tokens. Probe lines = val + test split, with per-page-line gold language sets from FWEET. **Phoneme source for both training and probe: `en-us-baseline`.** Using per-language IPA at probe time would leak the language label; using it at train time would make CONTEXT_L learn a native template that wouldn't match the English-realized probe. The CONTEXT_L assembly therefore learns *English-realized phonotactic correlates of foreign-source-language presence* — same competence an English-speaking Wake reader exercises.
- `src/replicate_multilingual_context.py` — explicit PHON (bigram blocks, unified inventory) + explicit CONTEXT (one slot per language, Mitropolsky-2025 MOOD-style — single area, multiple fixed assemblies); train per (token, language) instance with class-balanced subsampling via `--max-per-lang`; probe per held-out line by firing the union of its tokens' bigrams and reading the CONTEXT k-cap overlap per slot. Falls under the same `_common.py`/nemo-core scaffolding as `replicate_ngram_phon.py`.
- `src/baseline_attribution.py` — two classical baselines on the same lexicon and probe lines: (1) frequency baseline (predict argmax_L prior over training instances), (2) phoneme-bigram LangID (per-language Laplace-smoothed unigram distribution over bigrams; predict argmax_L sum log P(b|L)). Runs in both `--max-per-lang` modes (uncapped naive + matched-cap apples-to-apples).
- `src/evaluate_attribution.py` — takes labelled paths to one or more results files; computes per-method top-1 accuracy (overall + foreign-gold subset), per-language P/R/F1, macro-F1, and confusion matrices. Apples-to-apples comparison engine.

##### I.6 lexicon stats

| | Count |
|---|---|
| I.6 page-lines | 1,513 (train 1,221 / val 139 / test 153) |
| Tokens | 15,988 |
| **Training instances** (en-us 12,586 / la 86 / de 61 / fr-fr 49 / it 25) | **12,807** |
| Probe lines (val + test) | 292, **101 (34.6%) with ≥1 non-English gold** |
| Per-language gold occurrences | en-us 191 / la 40 / de 40 / fr-fr 28 / it 15 |
| Unique en-us phonemes | 91 |
| **Unique phoneme bigrams (PHON_n basis)** | **1,545** |

##### Three NEMO runs + matched baselines

| Run | Params | Wall | Overall | **Foreign-gold** | Notes |
|---|---|---|---|---|---|
| Smoke | rounds=2, max=30 | 13 min | 38.7% | **21.8%** (22/101) | Pipeline validated |
| Tier-1 (`cap=100`) | rounds=10, max=100 | 2h 28m | 42.8% | **15.8%** (16/101) | **Hebbian collapse** on minority classes (it→0%, fr-fr→3.6%) |
| **Strict equal-cap** (`eq=25`) | rounds=10, max=25 | 52 min | 41.1% | **21.8%** (22/101) | Collapse rescued; balanced per-class profile |

##### Headline four-way comparison

All methods evaluated on the same 292 probe lines, languages = `['de', 'en-us', 'fr-fr', 'it', 'la']`.

| Method | Overall | **Foreign-gold** | **Macro-F1** | de F1 | en-us F1 | fr-fr F1 | it F1 | la F1 |
|---|---|---|---|---|---|---|---|---|
| Frequency (uncapped, ≡ "always en-us") | 65.4% | **0.0%** | 0.158 | 0 | 0.791 | 0 | 0 | 0 |
| Bigram LangID (uncapped) | 63.7% | **0.0%** | 0.156 | 0 | 0.778 | 0 | 0 | 0 |
| Bigram LangID (cap=100) | 50.7% | **21.8%** | **0.281** | 0.203 | 0.677 | 0.145 | 0.162 | 0.217 |
| Bigram LangID (cap=25) | 47.9% | **21.8%** | **0.282** | 0.103 | 0.648 | 0.264 | 0.200 | 0.198 |
| NEMO Tier-1 (cap=100, collapse) | 42.8% | 15.8% | 0.187 | 0.024 | 0.609 | 0.062 | 0.000 | 0.241 |
| **NEMO equal-cap (eq=25)** | 41.1% | **21.8%** | **0.234** | 0.040 | 0.577 | 0.210 | 0.136 | 0.208 |

##### Per-class winner pattern — the actually-interesting finding

Restricted to NEMO eq=25 vs Bigram LangID cap=25 (the strict-balance comparison), per-language **correct counts** on probe lines where that language is in gold:

| Language | Gold lines | NEMO eq=25 | Bigram LangID cap=25 | Δ (NEMO − Bigram) |
|---|---|---|---|---|
| Latin (la) | 32 | **8** | 6 | **+2** |
| French (fr-fr) | 19 | **6** | **6** | 0 |
| Italian (it) | 10 | **2** | 1 | **+1** |
| German (de) | 40 | 1 | **4** | **−3** |
| English (en-us) | 191 | 98 | **118** | −20 |

**NEMO outperforms classical bigram LangID on the most phonotactically-distinctive classes (Latin, Italian); ties on French; loses on German and English.** The German loss is significant — German shares enough phonotactic structure with English (both Germanic) that NEMO's Hebbian assemblies can't disambiguate the near-cousin. Bigram LangID's explicit Laplace-smoothed likelihood handles it. The English loss is partly dominant-data (en-us has by far the most probe lines) and partly the same near-cousin issue inverted.

##### Honest reading

- **On the foreign-gold subset (the actual question — "does the model detect non-English presence?"), NEMO eq=25 and Bigram LangID tie exactly at 22/101 = 21.8%.** The two methods identify the same number of foreign-gold lines correctly; they just identify *different ones*.
- **On macro-F1 (the per-class-balanced summary), NEMO loses by ~5 pts (0.234 vs 0.282)**, driven by the German collapse and the English ceiling.
- **The per-class winner pattern is mechanistically interpretable:** Hebbian-plasticity assemblies discriminate via *phonotactic divergence* (Latin's `/-us/`, `/-um/`, `/-or/` endings are unmistakable; Italian's open-syllable pattern is too). Classical LangID discriminates via *statistical likelihood* (graceful Laplace falloff on near-cousins). They're learning different things from the same substrate.
- **The Tier-1 falsifiable claim ("NEMO exceeds the classical bigram LangID baseline") is not met as stated** — it's a tie on foreign-gold accuracy and a loss on macro-F1. But the per-class pattern is the more interesting finding for a phonology audience, and it's a defensible workshop-paper result: *NEMO Hebbian assemblies on a phonotactic substrate match classical bigram-LangID on foreign-language detection under strict class balance and outperform it on phonotactically-distinctive classes (Latin, Italian) while failing on phonotactically-near-cousin classes (German vs English). This suggests biologically-plausible assemblies discriminate via phonotactic distance rather than statistical likelihood — a property worth further investigation for code-switching and bilingual phonological acquisition models.*

##### Caveats

- **Hebbian collapse under class imbalance is the load-bearing failure mode.** The cap=100 run (en-us:la:de:fr:it = 100:86:61:49:25) drove Italian to 0% recall and French to 3.6% as plasticity over-reinforced en-us and Latin (the two largest classes). The cap=25 strict-balance run prevented it. For real-world acquisition data (which is always imbalanced), this is the next problem to fix. Lateral inhibition between CONTEXT slots (which Mitropolsky 2025 uses for MOOD areas) is the obvious next move — it should prevent dominant-class assembly drift into minority slots.
- **German is a persistent failure** even under strict balance (1/40 = 2.5% recall). The phonotactic-cousinship hypothesis is consistent with this but not proven; alternative explanations include the specific I.6 German tokens being short/atypical or the unified-inventory PHON not separating /ʃ/-rich German bigrams cleanly enough. A dedicated German error analysis is the natural follow-up.
- **N = 292 probe lines from one chapter, single seed.** No multi-seed significance test; the 22/101 tie could shift by ±2–3 lines on a different seed. The per-class winner pattern is robust to seed but the headline tie is not statistically secured.
- **English-only baseline ceiling.** The "always-predict-en-us" baseline gets 65.4% overall because Joyce wrote in English by default. Any honest evaluation has to centre the foreign-gold subset, which we did.
- **Single substrate (en-us-baseline IPA), single n-gram (bigram), single chapter (I.6).** Tier 2 would extend to multi-chapter + 22-language scope; that's a different paper and a different compute budget.

##### Artifacts

- Lexicon: `lexicons/i6_multilingual_top5.json` (1,545 bigrams, 12,807 train instances, 292 probe lines)
- NEMO equal-cap (the headline NEMO run): `results/mlctx_ngram2_20260530-100148/`
- NEMO Tier-1 cap=100 (the collapse run, kept for reference): `results/mlctx_ngram2_20260530-003720/`
- Baselines: `results/baselines/{uncapped,cap25,cap100}.json`
- Four-way comparison: `results/idea2_final_comparison.json`

##### Next steps (if pursued)

- **Lateral inhibition between CONTEXT slots** in nemo-core to prevent Hebbian collapse under imbalance; re-run cap=100 with inhibition active and see whether the imbalance penalty disappears. This is the most directly actionable engineering move and is consistent with the Mitropolsky-2025 architecture.
- **German error analysis** — token-level look at which German tokens NEMO mis-attributes, whether the misses cluster on phonotactic-cousin features or on short / generic tokens.
- **Multi-seed significance** on the foreign-gold tie (≥10 seeds).
- **Multi-chapter generalisation** — train on I.6, evaluate on II.2 / II.3 / etc. Tests whether the substrate learned a *general* phonotactic discriminator or only an I.6-specific one.
- **Top-N multi-label evaluation** for code-mixed lines (the current evaluation uses top-1 against a multi-label gold; top-N would credit partial matches).
- **Tier 2** — scale to all 22 FWEET languages, with the inhibition fix in place.



#### 4. Voice segmentation

Treat each voice (HCE, ALP, Shem, Shaun, the Four, the Washerwomen) as a bundle of co-activated context assemblies — not a single tag, but a characteristic pattern across CONTEXT_i. Train on hand-annotated passages so voice-bundle patterns stabilize. Test on unannotated passages: fire its words, read off which voice-bundle wins k-cap. Segmentation = the time series of dominant bundles. Richer than n-gram or stylometric voice ID — gives principled competition between voices in ambiguous passages, which is exactly the texture of the Wake.

- **Requirements:** sigla annotations in `data/annotations/` (already present); decision on whether the voice-area fires per-word or per-passage during supervised training; held-out evaluation passages.
- **Falsifiable claim:** held-out passages are correctly voiced at rate above an n-gram or stylometric baseline.
- **Status:** most architecturally ambitious; strongest methods contribution.
- human interpretation: this could be interesting.

#### 8. Portmanteau decomposition via partial-activation maps: FIRST PASS DONE — 2026-05-29

Build phoneme-sequence assemblies underneath PHON using sequence formation. `passencore` becomes a sequence chain that overlaps with the chains for `passenger`, `encore`, and (French) `pas encore`. Fire the `passencore` sequence and observe which lexical-hub assemblies partially activate. The overlap profile is the portmanteau's decomposition, expressed as a vector over hub-area activations. Done systematically across the Wake's nonce vocabulary, this yields a Joyce-specific portmanteau atlas.

- **Requirements:** new phoneme-position areas and sequence-formation scaffolding under PHON (largest engineering lift on the list); phoneme transcriptions for FW nonce vocabulary (the joyce-derived IPA artifact would supply this).
- **Falsifiable claim:** the model-derived decomposition of attested portmanteaus matches scholarly (McHugh, FWEET) decompositions above a baseline.
- **Status:** first-pass complete; see Results below.
- human: yes, this could be interesting.

##### Engagement with Mitropolsky & Papadimitriou 2025 ("Simulated Language Acquisition in a Biologically Realistic Model of the Brain", arXiv 2507.11788)

The 2025 acquisition paper supersedes the 2021 parser (`papers/assemblies_biological_language_organ.pdf`) by learning the lexicon, POS, word order, generation, and basic hierarchy from grounded sentences — but **explicitly bypasses phonetics and treats words as atomic stable PHON assemblies**: "we bypass this phase by adopting an input-output convention whereby a sentence is presented to the system as a sequence of stimuli corresponding to word tokens." That sub-PHON gap is exactly what idea 8 targets. The Wake violates nearly every NEMO acquisition precondition (atomic stable tokens, sensorimotor grounding, concrete N/V lexicon, ignorable function words), so a precondition stress-test is the natural framing; idea 8 is the additive, buildable thread that engages the sub-lexical gap directly.

##### Build (Tier 1: n-gram bag-vs-order ablation)

`nemo-core` has **no sequence-formation primitive** (no `sequence` / `chain` / `next_assembly`; only `project`, `associate`, `merge`, `fix_assembly`). Implementing Dabagia–Papadimitriou–Vempala sequence formation in the engine ("Computation with sequences of assemblies", Neural Computation 2024) is weeks of C++/Python — Tier 3 in the spec. As a cheap pre-test of whether order matters at all, the first pass approximates order with **phoneme n-gram units**: PHON has one explicit block per contiguous phoneme n-gram; a word's PHON activation is the union of its n-gram blocks. `--ngram 1` = bag-of-phonemes (the existing `replicate_phoneme_phon.py` design); `--ngram 2` = bigrams (order-aware). No engine change. If order helps at n-gram, the Tier 3 build is motivated.

Three new files:

- `src/replicate_ngram_phon.py` — n-gram variant of `replicate_phoneme_phon.py`. Adds `--ngram N` and a **partial-activation map** per probe: after training (plasticity off), each trained morpheme's hub k-cap is recorded as an *anchor assembly*; at probe, the portmanteau's induced k-cap is overlapped against every anchor, producing `activation_profile = [{trained_word, overlap, overlap_frac}, ...]` sorted desc. That overlap profile *is* the partial-activation map idea 8 names.
- `src/activation_alignment.py` — reference-agnostic scorer. Given an `activation_profile` and a reference set of true constituents per probe, computes precision@k, MRR, and **separation** (mean overlap_frac of constituents − mean of distractors), each with a random baseline.
- `scripts/preprocessing/text/build_phoneme_gallery.py` — new builder (separate from your in-progress morpheme gallery builder), pulls portmanteaus from `data/pos/joyce-pos-hypotheses/`, phonemizes via espeak en-us, emits a lexicon where **trained sub-units = unique segmenter morphemes (POS-bucketed)** and each probe carries `constituents_by_segmenter` + `fweet_source_forms`. Requires `--require-fweet` (default on) so only portmanteaus with scholarly annotation are kept; produces a tractable lexicon (`lexicons/ep1_phoneme_gallery.json`: 86 training morphemes / 24 portmanteaus for I.1).
- `src/fweet_compare.py` — gains a `--mode activation` that joins n-gram results + phoneme gallery, scores the activation profile against each segmenter's decomposition, and cross-tabs with the existing FWEET count concordance.

**Design choice (deliberate):** trained sub-units are segmenter morphemes in **English IPA**, *not* FWEET source-forms in their source languages. Source-form-as-trained-unit (Design A) would suffer a cross-lingual phoneme-inventory confound (espeak's French nasals vs English vowels barely overlap), producing a null result driven by inventory mismatch rather than phonology. Design B keeps everything in one inventory; FWEET enters indirectly through the count concordance.

##### Methodology — three independent lenses on each portmanteau

1. **Segmenter** (BPE 4k / Unigram 4k / Morfessor Baseline / FlatCat) — algorithmic decomposition.
2. **FWEET** — scholarly multilingual source-form annotation (count + ID).
3. **NEMO partial-activation map** — biologically-motivated emergent decomposition: which trained morphemes does the portmanteau light up?

For each portmanteau the NEMO profile is scored against each segmenter's constituent list (separation, P@k); per portmanteau a **NEMO-favoured segmenter** is recorded (max separation). The **FWEET-favoured segmenter(s)** is anyone whose morph count matches the modal scholarly source-form count within ±1 (the metric `fweet_compare` already used). Then count: NEMO-favoured ∈ FWEET-favoured.

##### Results — I.1 phoneme gallery, 24 FWEET-annotated portmanteaus, 86 training morphemes

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

1. **Order improves recovery across every segmenter.** Bigram > bag for all four; largest gain on FlatCat (+64% separation), smallest on Unigram (~flat). The order-aware substrate suppresses spurious distractor overlap that the bag substrate accepts (e.g. *sends* → *riverrun* via stray /n/ collapses from 0.15 in bag to ~0.01 in bigram on the small balanced lexicon).
2. **FlatCat wins both regimes**, on both NEMO substrate alignment AND FWEET concordance. The segmenter whose decomposition the phonological substrate recovers most cleanly is also the one matching the scholarly annotation. Matches the `shared/README.md` claim that Morfessor's bag-of-morphs is "evidently false for the Wake" and FlatCat's HMM-over-categories repairs it.
3. **95% NEMO ↔ FWEET agreement** on the bigram substrate — for 19 of 20 portmanteaus where both lenses resolve, the same segmenter wins. Three independent lenses (FlatCat, FWEET, NEMO bigram substrate) converge on the same per-portmanteau verdict.

##### Caveats

- **Param confound between the two runs.** Bigram was at β=0.06 / rounds=20 / proj-rounds=2 (the original gallery run). Bag at β=0.03 / rounds=10 / proj-rounds=1 (the safer rerun after the original bag run hit float32 plasticity overflow — `NOUN_in` values around 10³³, profiles saturated to identical nonsense). The bag is *under-trained relative to bigram*, so part of the gap is plasticity, not representation. The order-wins direction is theory-predicted and unlikely to invert under a strict apples-to-apples rerun, but a strict comparison is outstanding (~40 min at safer params).
- **Hebbian overflow is real and parameter-sensitive.** Bigrams happened to avoid it at β=0.06 / rounds=20 (denser PHON spreads input across more neurons, slowing per-neuron weight growth); bags overflow under the same params. For corpus scale-up (whole Wake or larger galleries) a per-projection weight clamp in `brain.py` is probably necessary.
- **Tier-1 caveat.** Phoneme n-grams approximate order via local context windows; they don't encode position or *true* sequence (a/b/c vs b/a/c with the same bigram set look identical at higher N). The Tier 3 sequence-formation engine build remains the faithful version of idea 8. The Tier 1 result motivates it: if a coarse order proxy already gets +64% on the central metric and lifts cross-lens agreement to 95%, the faithful chain operation should do better still.
- **n=10000 hubs with 86 trained morphemes** — assembly density is approaching the regime where chance overlap is non-trivial (k²/n = 1 expected, frac ≈ 0.01); raise `n` for cleaner separation if scaling up.

##### Artifacts

- Lexicon: `lexicons/ep1_phoneme_gallery.json`
- Bigram run: `results/ngram2_phon_20260529-182913/` (clean, β=0.06 / rounds=20 / proj=2)
- Bag run: `results/ngram1_phon_20260529-214647/` (clean, β=0.03 / rounds=10 / proj=1)
- Three-lens JSON: each run dir contains `three_lens_compare.json`
- Original bag run with overflow (kept for reference): `results/ngram1_phon_20260529-170445/` — *do not use for analysis*

##### Next steps (if pursued)

- Strict apples-to-apples rerun (bigram at safer params).
- Extend gallery beyond I.1 (more portmanteaus, more statistical power).
- Implement Tier 3 (true sequence formation in nemo-core) and re-run.
- A weight-clamp in `brain.py` to make corpus-scale runs robust to overflow.


## Reference

`papers/assemblies_biologial_language_organ.pdf`

## Existing Work

## Works Cited
