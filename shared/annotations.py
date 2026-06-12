"""
Loader for corpus annotation overlays in data/annotations/.

Annotations are supplementary data that layer over the scraped corpus — for
example, sigla that are absent from the telelib HTML source but present in the
printed 1939 Faber & Faber edition.

Usage:
    from shared.annotations import load_sigla, SiglaAnnotation

    annotations = load_sigla(book=1, episode=6)
    legend = load_sigla_legend(book=1, episode=6)
"""

import json
from dataclasses import dataclass
from pathlib import Path

_ANNOTATIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "annotations"


@dataclass(frozen=True)
class SiglaAnnotation:
    section_idx: int
    char_offset: int   # -1 means the siglum precedes the section
    siglum: str        # Unicode character from sigla_legend
    note: str | None   # optional source reference


@dataclass(frozen=True)
class SiglaLegendEntry:
    siglum: str
    referent: str       # e.g. "HCE"
    full_name: str      # e.g. "Humphrey Chimpden Earwicker"
    unicode: str        # e.g. "U+018E"
    unicode_name: str   # e.g. "LATIN CAPITAL LETTER REVERSED E"
    note: str | None


def _annotation_path(book: int, episode: int) -> Path:
    return _ANNOTATIONS_DIR / f"book{book:02d}_ep{episode:02d}_sigla.json"


def load_sigla(book: int, episode: int) -> tuple[SiglaAnnotation, ...]:
    """Return sigla annotations for the given chapter. Empty tuple if none exist."""
    path = _annotation_path(book, episode)
    if not path.exists():
        return ()
    raw = json.loads(path.read_text())
    return tuple(
        SiglaAnnotation(
            section_idx=a["section_idx"],
            char_offset=a["char_offset"],
            siglum=a["siglum"],
            note=a.get("note"),
        )
        for a in raw.get("annotations", [])
    )


def load_sigla_legend(book: int, episode: int) -> dict[str, SiglaLegendEntry]:
    """Return the sigla legend for the given chapter, keyed by Unicode character."""
    path = _annotation_path(book, episode)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {
        char: SiglaLegendEntry(
            siglum=char,
            referent=entry["referent"],
            full_name=entry["full_name"],
            unicode=entry["unicode"],
            unicode_name=entry["unicode_name"],
            note=entry.get("note"),
        )
        for char, entry in raw.get("sigla_legend", {}).items()
    }


def validate(book: int, episode: int, n_sections: int) -> list[str]:
    """Return a list of validation errors for the annotation file, or [] if clean."""
    path = _annotation_path(book, episode)
    if not path.exists():
        return [f"No annotation file found at {path}"]
    raw = json.loads(path.read_text())
    legend_chars = set(raw.get("sigla_legend", {}).keys())
    errors = []
    for i, a in enumerate(raw.get("annotations", [])):
        if a["section_idx"] < 0 or a["section_idx"] >= n_sections:
            errors.append(f"annotations[{i}]: section_idx {a['section_idx']} out of range (0–{n_sections-1})")
        if a["siglum"] != "UNCONFIRMED" and a["siglum"] not in legend_chars:
            errors.append(f"annotations[{i}]: siglum {repr(a['siglum'])} not in sigla_legend")
    return errors


if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Validate a sigla annotation file")
    parser.add_argument("file", help="Annotation filename (e.g. book01_ep06_sigla.json)")
    parser.add_argument("--sections", type=int, default=None, help="Expected section count")
    args = parser.parse_args()

    fname = Path(args.file).name
    parts = fname.replace("_sigla.json", "").split("_")
    book = int(parts[0].replace("book", ""))
    ep = int(parts[1].replace("ep", ""))

    if args.sections is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from shared.corpus import load_chapter
        chapter = load_chapter(book, ep)
        n = len(chapter.sections)
    else:
        n = args.sections

    errors = validate(book, ep, n)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        sys.exit(1)
    raw = json.loads(_annotation_path(book, ep).read_text())
    print(f"OK — {len(raw.get('annotations', []))} annotations, {len(raw.get('sigla_legend', {}))} sigla defined")
