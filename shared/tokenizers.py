"""
Tokenizer training and loading for the Finnegans Wake corpus.

Three families of unsupervised segmenters, trained on the Wake itself:
  - BPE (byte-pair encoding): greedy frequency-based character-pair merges.
  - Unigram LM (SentencePiece-style): probabilistic, trained by EM to maximize
    corpus likelihood under a unigram model over subwords. Same library as BPE.
  - Morfessor (Baseline + FlatCat): MDL-based morphological segmentation; FlatCat
    adds an HMM over morph categories for sequential structure.

Training is one-time and writes model files to data/tokenizers/. Loaders are
imported at runtime by projects that need to segment Wake text.

Usage:
    from shared.tokenizers import load_bpe, load_unigram, load_morfessor
    bpe = load_bpe(vocab_size=8000)
    bpe.encode("passencore").tokens

    uni = load_unigram(vocab_size=8000)
    uni.encode("passencore").tokens

    morf = load_morfessor()
    morf.viterbi_segment("passencore")[0]

Train via: python scripts/preprocessing/text/train_tokenizers.py

Caveat on Morfessor Baseline: its model assumes the morphs in a compound
occur independently (bag-of-morphs). This is a known limitation that bites
the Wake especially hard — *passencore* is a sequence-of-source-words
construction, not an unordered set. Categories-MAP and FlatCat relax this;
revisit if Baseline outputs look uninformative on Wake test cases.
"""

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from shared.corpus import Split, iter_lines, line_split, load_all

if TYPE_CHECKING:  # heavy deps are imported lazily inside functions
    import flatcat
    import morfessor
    from tokenizers import Tokenizer

_REPO_ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_DIR = _REPO_ROOT / "data" / "tokenizers"
MORFESSOR_PATH = TOKENIZER_DIR / "morfessor.bin"
FLATCAT_PATH = TOKENIZER_DIR / "flatcat.bin"

# Sweep range for BPE and Unigram: small enough to force subword merges, large
# enough to learn productive patterns. The Wake has ~72k unique types, so vocab
# sizes at or above that range degenerate to whole-word memorization.
DEFAULT_VOCAB_SIZES = (1000, 2000, 4000, 8000, 16000, 32000)


def bpe_path(vocab_size: int) -> Path:
    """Path to the BPE model trained at a given vocab size."""
    return TOKENIZER_DIR / f"bpe_{vocab_size}.json"


def unigram_path(vocab_size: int) -> Path:
    """Path to the Unigram LM model trained at a given vocab size."""
    return TOKENIZER_DIR / f"unigram_{vocab_size}.json"


def superbpe_path(vocab_size: int) -> Path:
    """Path to the SuperBPE model trained at a given total vocab size."""
    return TOKENIZER_DIR / f"superbpe_{vocab_size}.json"


def iter_tokens(split: Split | None = "train") -> Iterator[str]:
    """Whitespace-split corpus tokens with line-break hyphens rejoined.

    A token ending in a single '-' (not '--', Joyce's em-dash) is carried to
    the next line's main text and prepended to its first token. Margin tokens
    (Book II Episode 2 only) are emitted independently and not subject to
    carry-over from main text.

    Defaults to the training split. Pass split="val", "test", or None (all
    lines) to evaluate trained tokenizers on held-out data.
    """
    carry = ""
    for line in iter_lines(split=split):
        if line.text:
            toks = line.text.split()
            if toks:
                if carry:
                    toks[0] = carry + toks[0]
                    carry = ""
                last = toks[-1]
                if last.endswith("-") and not last.endswith("--"):
                    # Drop the typographic line-break hyphen; it isn't part of the word.
                    carry = last[:-1]
                    toks = toks[:-1]
                yield from toks
        if line.left_margin:
            yield from line.left_margin.split()
        if line.right_margin:
            yield from line.right_margin.split()
    if carry:
        yield carry


def iter_training_chunks(split: Split | None = "train") -> Iterator[str]:
    """One string per chapter of main text (hyphen-rejoined) plus margin strings.

    Suitable for BPE/Unigram training via Tokenizer.train_from_iterator. Hyphen
    rejoining is applied within each chapter; margins are yielded as separate
    strings so margin text is not concatenated with main text.

    Defaults to the training split. Lines not in the requested split are
    skipped entirely (hyphen carry is broken at split boundaries, which is the
    correct behavior — we don't want a held-out line to be silently merged
    with a training line).
    """
    for chapter in load_all():
        main_tokens: list[str] = []
        margins: list[str] = []
        carry = ""
        for line in chapter.lines:
            if split is not None and line_split(line.page_line) != split:
                carry = ""
                continue
            if line.text:
                toks = line.text.split()
                if toks:
                    if carry:
                        toks[0] = carry + toks[0]
                        carry = ""
                    last = toks[-1]
                    if last.endswith("-") and not last.endswith("--"):
                        # Drop the typographic line-break hyphen.
                        carry = last[:-1]
                        toks = toks[:-1]
                    main_tokens.extend(toks)
            if line.left_margin:
                margins.append(line.left_margin)
            if line.right_margin:
                margins.append(line.right_margin)
        if carry:
            main_tokens.append(carry)
            carry = ""
        if main_tokens:
            yield " ".join(main_tokens)
        yield from margins


def corpus_token_counts(split: Split | None = "train") -> Counter[str]:
    """Word frequency counter for Morfessor training input."""
    return Counter(iter_tokens(split=split))


def train_bpe(
    vocab_size: int,
    out_path: Path | None = None,
    show_progress: bool = True,
) -> "Tokenizer":
    """Train a whitespace-pretokenized BPE model on the Wake corpus.

    Saves to data/tokenizers/bpe_{vocab_size}.json unless out_path overrides.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import BpeTrainer

    if out_path is None:
        out_path = bpe_path(vocab_size)
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>"],
        show_progress=show_progress,
    )
    tok.train_from_iterator(iter_training_chunks(), trainer=trainer)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    return tok


def load_bpe(vocab_size: int, path: Path | None = None) -> "Tokenizer":
    """Load the BPE model trained at the given vocab size."""
    from tokenizers import Tokenizer

    if path is None:
        path = bpe_path(vocab_size)
    return Tokenizer.from_file(str(path))


def train_unigram(
    vocab_size: int,
    out_path: Path | None = None,
    show_progress: bool = True,
) -> "Tokenizer":
    """Train a whitespace-pretokenized Unigram LM segmenter on the Wake corpus.

    Unigram LM (Kudo 2018) trains by EM to maximize corpus likelihood under a
    unigram distribution over subwords, then prunes to the target vocab size.
    The probabilistic objective is closer to Morfessor's MDL than to BPE's
    frequency-merge heuristic.

    Saves to data/tokenizers/unigram_{vocab_size}.json unless out_path overrides.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import Unigram
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import UnigramTrainer

    if out_path is None:
        out_path = unigram_path(vocab_size)
    tok = Tokenizer(Unigram())
    tok.pre_tokenizer = Whitespace()
    trainer = UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>"],
        unk_token="<unk>",
        show_progress=show_progress,
    )
    tok.train_from_iterator(iter_training_chunks(), trainer=trainer)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    return tok


def load_unigram(vocab_size: int, path: Path | None = None) -> "Tokenizer":
    """Load the Unigram LM model trained at the given vocab size."""
    from tokenizers import Tokenizer

    if path is None:
        path = unigram_path(vocab_size)
    return Tokenizer.from_file(str(path))


def train_superbpe(
    vocab_size: int,
    transition_point: int | None = None,
    out_path: Path | None = None,
) -> None:
    """SuperBPE (Liu et al. 2025) — stub. Training requires a parallel environment.

    The upstream implementation at https://github.com/PythonNut/superbpe uses a
    custom fork of huggingface/tokenizers that conflicts with the standard
    package installed in this venv. To produce a SuperBPE model file:

        1. Create a separate venv (e.g. .venv-superbpe) with the upstream fork.
        2. Run the upstream two-stage scripts:
             - train_tokenizer.sh   (stage 1: BPE with whitespace pretokenization
               to `transition_point` vocab — learns subwords only)
             - extend_tokenizer.sh  (stage 2: continue without pretokenization
               to `vocab_size` total — learns superwords bridging whitespace)
        3. Use upstream's construct_hf_tokenizer() helper to produce an
           HF-compatible tokenizer.json.
        4. Drop the result at data/tokenizers/superbpe_{vocab_size}.json.

    Once the JSON file exists, load_superbpe(vocab_size) works in our standard
    environment — only *training* needs the fork; the output format is HF-
    compatible.

    Reserved for future motif-detection work where cross-whitespace tokens
    (e.g. "Here Comes Everybody", thunderwords as single units) are useful.
    """
    raise NotImplementedError(
        "SuperBPE training requires a parallel venv with the upstream fork. "
        f"See shared.tokenizers.train_superbpe docstring. Target: "
        f"vocab_size={vocab_size}, transition_point={transition_point}, "
        f"out_path={out_path or superbpe_path(vocab_size)}."
    )


def load_superbpe(vocab_size: int, path: Path | None = None) -> "Tokenizer":
    """Load a SuperBPE model trained at the given vocab size.

    Works in the standard venv — upstream's construct_hf_tokenizer() produces
    an HF-compatible JSON, so only training requires the upstream fork.
    """
    from tokenizers import Tokenizer

    if path is None:
        path = superbpe_path(vocab_size)
    if not path.exists():
        raise FileNotFoundError(
            f"No SuperBPE model at {path}. SuperBPE must be trained externally — "
            f"see shared.tokenizers.train_superbpe docstring for setup instructions."
        )
    return Tokenizer.from_file(str(path))


def train_morfessor(
    out_path: Path = MORFESSOR_PATH,
    freqthreshold: int = 1,
) -> "morfessor.BaselineModel":
    """Train a Morfessor Baseline model on Wake word types and counts."""
    import morfessor

    counts = corpus_token_counts()
    data = [(c, w) for w, c in counts.items()]
    model = morfessor.BaselineModel()
    model.load_data(data, freqthreshold=freqthreshold)
    model.train_batch()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    io = morfessor.MorfessorIO()
    io.write_binary_model_file(str(out_path), model)
    return model


def load_morfessor(path: Path = MORFESSOR_PATH) -> "morfessor.BaselineModel":
    import morfessor

    io = morfessor.MorfessorIO()
    return io.read_binary_model_file(str(path))


def train_flatcat(
    out_path: Path = FLATCAT_PATH,
    baseline_model: "morfessor.BaselineModel | None" = None,
) -> "flatcat.FlatcatModel":
    """Train an unsupervised Morfessor FlatCat model.

    FlatCat initializes from a Baseline segmentation, then trains an HMM over
    morph categories (prefix/stem/suffix/non-morpheme) to add sequential
    structure on top of Baseline's bag-of-morphs likelihood.

    If baseline_model is None, loads the Baseline model from MORFESSOR_PATH
    (so train_morfessor() must have been run first).
    """
    import flatcat

    if baseline_model is None:
        baseline_model = load_morfessor()

    model = flatcat.FlatcatModel()
    # Baseline yields (count, compound, morphs) triples; FlatCat expects (count, morphs).
    segmentations = (
        (count, morphs) for count, _compound, morphs in baseline_model.get_segmentations()
    )
    model.add_corpus_data(segmentations)
    model.initialize_baseline()
    model.initialize_hmm()
    model.train_batch()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    io = flatcat.FlatcatIO()
    io.write_binary_model_file(str(out_path), model)
    return model


def load_flatcat(path: Path = FLATCAT_PATH) -> "flatcat.FlatcatModel":
    import flatcat

    io = flatcat.FlatcatIO()
    return io.read_binary_model_file(str(path))
