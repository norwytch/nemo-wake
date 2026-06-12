"""
Canonical corpus loader for the Finnegans Wake JSONL files in data/raw/.

Usage:
    from shared.corpus import load_chapter, load_all, iter_lines

All projects should use this module rather than reading JSONL directly.

Corpus files: data/raw/book{NN}_ep{NN}.jsonl
  One JSON record per line, schema:
    {
      "book": int,
      "episode": int,
      "page": int,
      "line": int,
      "page_line": "NNN.NN",    # canonical 1939 F&F citation coordinate
      "text": str,               # [N] inline markers for footnote refs
      "left_margin": str|null,   # Book II Episode 2 only
      "right_margin": str|null   # Book II Episode 2 only
    }

Footnote body text: data/raw/book{NN}_ep{NN}_footnotes.json
  {"book": int, "episode": int, "source": "telelib.com", "footnotes": {"1": "..."}}
  Load with load_footnotes(book, episode).
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

Split = Literal["train", "val", "test"]


def line_split(page_line: str) -> Split:
    """Deterministically assign a line to train/val/test by hashing its coordinate.

    Hash-based assignment is reproducible without a state file and produces an
    approximately stratified 80/10/10 split across the corpus (each chapter
    converges to 80/10/10 in expectation by the law of large numbers).

    Uses SHA-256 of the page.line coordinate (e.g. "276.03") mod 100:
      0..79  → train
      80..89 → val
      90..99 → test
    """
    h = int(hashlib.sha256(page_line.encode()).hexdigest(), 16) % 100
    if h < 80:
        return "train"
    if h < 90:
        return "val"
    return "test"


@dataclass(frozen=True)
class Line:
    book: int
    episode: int
    page: int
    line: int
    page_line: str            # e.g. "003.01"
    text: str
    left_margin: str | None   # Book II Episode 2 only
    right_margin: str | None  # Book II Episode 2 only


@dataclass(frozen=True)
class Chapter:
    book: int
    episode: int
    lines: tuple[Line, ...]


def load_chapter(book: int, episode: int, data_dir: Path = _DATA_DIR) -> Chapter:
    path = data_dir / f"book{book:02d}_ep{episode:02d}.jsonl"
    lines = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(raw_line)
        lines.append(Line(
            book=r["book"],
            episode=r["episode"],
            page=r["page"],
            line=r["line"],
            page_line=r["page_line"],
            text=r["text"],
            left_margin=r.get("left_margin"),
            right_margin=r.get("right_margin"),
        ))
    return Chapter(book=book, episode=episode, lines=tuple(lines))


def load_all(data_dir: Path = _DATA_DIR) -> Iterator[Chapter]:
    """Yield all chapters in book/episode order."""
    for path in sorted(data_dir.glob("book*.jsonl")):
        lines = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(raw_line)
            lines.append(Line(
                book=r["book"],
                episode=r["episode"],
                page=r["page"],
                line=r["line"],
                page_line=r["page_line"],
                text=r["text"],
                left_margin=r.get("left_margin"),
                right_margin=r.get("right_margin"),
            ))
        if lines:
            yield Chapter(book=lines[0].book, episode=lines[0].episode, lines=tuple(lines))


def iter_lines(
    books: list[int] | None = None,
    episodes: list[tuple[int, int]] | None = None,
    split: Split | None = None,
    data_dir: Path = _DATA_DIR,
) -> Iterator[Line]:
    """Iterate over lines, optionally filtered by book/episode and split.

    split: "train", "val", "test", or None for all lines. Train/val/test
    assignment is deterministic per line via line_split().
    """
    for chapter in load_all(data_dir):
        if books and chapter.book not in books:
            continue
        if episodes and (chapter.book, chapter.episode) not in episodes:
            continue
        for line in chapter.lines:
            if split is not None and line_split(line.page_line) != split:
                continue
            yield line


def load_footnotes(book: int, episode: int, data_dir: Path = _DATA_DIR) -> dict[str, str]:
    """Load footnote body text dict for a chapter. Returns {} if file absent."""
    path = data_dir / f"book{book:02d}_ep{episode:02d}_footnotes.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("footnotes", {})
