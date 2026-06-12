"""
Loader for IPA transcriptions of the Finnegans Wake corpus.

IPA artifacts are organized per-language at data/ipa/{language}/book*.jsonl,
because the polyglot Wake produces a different IPA reading under each language's
G2P rules. The first artifact (en-us) is the naive English G2P baseline; later
passes (fr-fr, de, it, la, ...) will be added once we have FWEET-derived
per-token language tags. A "joyce-derived" set, predicted from Joyce's own
1929 ALP recording, is a separate planned track.

Tokenization is whitespace-split (same convention as elsewhere in the project);
line-break hyphens are NOT rejoined at the IPA level — the artifact reflects the
physical lines of the Trent edition. Downstream consumers can apply hyphen
rejoining and re-transcribe rejoined tokens if morphological alignment matters.

Record schema (per language subdir):
  {
    "book": int, "episode": int, "page": int, "line": int,
    "page_line": "NNN.NN",
    "tokens": [[orth, ipa], ...],                  # main text
    "left_margin_tokens": [[orth, ipa], ...] | null,
    "right_margin_tokens": [[orth, ipa], ...] | null
  }

Produced by: python scripts/preprocessing/text/transcribe_ipa.py --language ...
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from shared.corpus import Split, line_split

_DATA_BASE_DIR = Path(__file__).resolve().parent.parent / "data" / "ipa"


def ipa_dir(language: str = "en-us") -> Path:
    """Path to the per-language IPA artifact directory."""
    return _DATA_BASE_DIR / language


@dataclass(frozen=True)
class IPAToken:
    orth: str
    ipa: str


@dataclass(frozen=True)
class IPALine:
    book: int
    episode: int
    page: int
    line: int
    page_line: str
    tokens: tuple[IPAToken, ...]
    left_margin_tokens: tuple[IPAToken, ...] | None
    right_margin_tokens: tuple[IPAToken, ...] | None


def _parse_tokens(raw: list[list[str]] | None) -> tuple[IPAToken, ...] | None:
    if not raw:
        return None
    return tuple(IPAToken(orth=o, ipa=i) for o, i in raw)


def iter_ipa_lines(
    books: list[int] | None = None,
    episodes: list[tuple[int, int]] | None = None,
    split: Split | None = None,
    language: str = "en-us",
    data_dir: Path | None = None,
) -> Iterator[IPALine]:
    """Iterate over IPA-transcribed lines, optionally filtered by book/episode/split.

    Defaults to the en-us pass; pass language="fr-fr" etc. for other passes,
    or data_dir=... to override the path entirely.
    """
    if data_dir is None:
        data_dir = ipa_dir(language)
    for path in sorted(data_dir.glob("book*.jsonl")):
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(raw_line)
            if books and r["book"] not in books:
                continue
            if episodes and (r["book"], r["episode"]) not in episodes:
                continue
            if split is not None and line_split(r["page_line"]) != split:
                continue
            yield IPALine(
                book=r["book"],
                episode=r["episode"],
                page=r["page"],
                line=r["line"],
                page_line=r["page_line"],
                tokens=_parse_tokens(r["tokens"]) or (),
                left_margin_tokens=_parse_tokens(r.get("left_margin_tokens")),
                right_margin_tokens=_parse_tokens(r.get("right_margin_tokens")),
            )


def iter_ipa_tokens(
    split: Split | None = None,
    include_margins: bool = True,
    language: str = "en-us",
    data_dir: Path | None = None,
) -> Iterator[IPAToken]:
    """Flat iteration over IPA tokens from all chapters, optionally filtered by split."""
    for line in iter_ipa_lines(split=split, language=language, data_dir=data_dir):
        yield from line.tokens
        if include_margins:
            if line.left_margin_tokens:
                yield from line.left_margin_tokens
            if line.right_margin_tokens:
                yield from line.right_margin_tokens
