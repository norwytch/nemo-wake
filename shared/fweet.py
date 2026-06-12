"""
Loader for parsed FWEET elucidations.

Source data: data/raw/fweet_elucidations.jsonl (produced by
scripts/preprocessing/text/parse_fweet.py from papers/FWEET_elucidation).

For language-tagged elucidations (where the first tag is a glossary tag whose
code is in LANGUAGE_CODES), we additionally extract:
  - language_code: the FWEET code (e.g., "_F_")
  - espeak_code: corresponding espeak-ng language code (e.g., "fr-fr") for
    use in multi-language G2P passes
  - source_form: the source-language fragment from the elucidation text
  - gloss: the English gloss following the colon, if present

Pattern: "Latin Helveticus: Swiss" → lang="_L_"/la, source_form="Helveticus",
gloss="Swiss".

Usage:
    from shared.fweet import iter_elucidations, load_by_page_line, languages_for
    by_pl = load_by_page_line()
    for elu in by_pl.get("003.04", []):
        if elu.language_code:
            print(elu.language_code, elu.source_form, '→', elu.gloss)

    langs = languages_for("003.04", by_pl)  # set of FWEET codes
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator

_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "fweet_elucidations.jsonl"


# FWEET language codes → espeak-ng language codes.
# Conservative: only codes that (a) appear in the FWEET dump, (b) refer to
# actual languages distinct from English, (c) have espeak-ng support.
# English-register codes (Slang, Colloquial, Archaic, Obsolete, Anglo-Irish,
# Dialect) are deliberately excluded — they're English G2P + Joyce overlay,
# not a different language.
LANGUAGE_CODES: dict[str, str] = {
    "_F_": "fr-fr",     # French
    "_G_": "de",        # German
    "_L_": "la",        # Latin (espeak-ng has limited Latin support)
    "_It_": "it",       # Italian
    "_I_": "ga",        # Irish (Gaelic)
    "_Du_": "nl",       # Dutch
    "_Da_": "da",       # Danish
    "_Gr_": "el",       # Greek (modern; espeak does not distinguish Ancient)
    "_N_": "nb",        # Norwegian Bokmål
    "_Ru_": "ru",       # Russian
    "_Heb_": "he",      # Hebrew
    "_Cn_": "cmn",      # Mandarin Chinese
    "_Sw_": "sv",       # Swedish
    "_Sp_": "es",       # Spanish
    "_Pg_": "pt",       # Portuguese
    "_Pol_": "pl",      # Polish
    "_Hu_": "hu",       # Hungarian
    "_Cz_": "cs",       # Czech
    "_Tk_": "tr",       # Turkish
    "_Fi_": "fi",       # Finnish
    "_Wel_": "cy",      # Welsh
    "_Skt_": "hi",      # Sanskrit (no exact espeak match; Hindi as closest)
    "_Ar_": "ar",       # Arabic
    "_Jp_": "ja",       # Japanese
    "_Per_": "fa",      # Persian
}


@dataclass(frozen=True)
class Tag:
    kind: str
    code: str
    label: str


@dataclass(frozen=True)
class Elucidation:
    page_line: str
    text: str
    tags: tuple[Tag, ...]
    sigla: tuple[str, ...]
    cross_refs: tuple[str, ...]
    language_code: str | None   # FWEET code (e.g., "_F_") if applicable
    espeak_code: str | None     # espeak-ng code (e.g., "fr-fr") if applicable
    source_form: str | None     # source-language fragment
    gloss: str | None           # English gloss after the colon


def _extract_language_info(
    record: dict,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Return (lang_code, espeak_code, source_form, gloss).

    Returns (None, None, None, None) if the elucidation isn't language-tagged.
    """
    if not record["tags"]:
        return None, None, None, None
    first = record["tags"][0]
    if first["kind"] != "glossary":
        return None, None, None, None
    code = first["code"]
    if code not in LANGUAGE_CODES:
        return None, None, None, None

    label = first["label"]
    text = record["text"]
    # Find label in text (usually at the start, but allow leading whitespace)
    idx = text.find(label)
    if idx == -1:
        return code, LANGUAGE_CODES[code], None, None
    rest = text[idx + len(label):].lstrip()
    if ":" in rest:
        source_form, gloss = rest.split(":", 1)
        return code, LANGUAGE_CODES[code], source_form.strip(), gloss.strip()
    return code, LANGUAGE_CODES[code], rest.strip() or None, None


def _build_elucidation(record: dict) -> Elucidation:
    lang_code, espeak_code, source_form, gloss = _extract_language_info(record)
    return Elucidation(
        page_line=record["page_line"],
        text=record["text"],
        tags=tuple(
            Tag(kind=t["kind"], code=t["code"], label=t["label"])
            for t in record["tags"]
        ),
        sigla=tuple(record["sigla"]),
        cross_refs=tuple(record["cross_refs"]),
        language_code=lang_code,
        espeak_code=espeak_code,
        source_form=source_form,
        gloss=gloss,
    )


def iter_elucidations(data_path: Path = _DATA_PATH) -> Iterator[Elucidation]:
    """Stream-yield all FWEET elucidations."""
    with data_path.open(encoding="utf-8") as f:
        for line in f:
            yield _build_elucidation(json.loads(line))


@lru_cache(maxsize=1)
def load_by_page_line() -> dict[str, tuple[Elucidation, ...]]:
    """Return dict mapping each page.line to its elucidations.

    Cached: subsequent calls are O(1). The full dict is ~100k elucidations
    across ~21k page.lines (~30 MB in memory).
    """
    out: dict[str, list[Elucidation]] = {}
    for elu in iter_elucidations():
        out.setdefault(elu.page_line, []).append(elu)
    return {pl: tuple(elus) for pl, elus in out.items()}


def languages_for(
    page_line: str,
    by_page_line: dict[str, tuple[Elucidation, ...]] | None = None,
) -> set[str]:
    """Return set of FWEET language codes active on this page.line."""
    if by_page_line is None:
        by_page_line = load_by_page_line()
    return {
        e.language_code
        for e in by_page_line.get(page_line, ())
        if e.language_code is not None
    }


def espeak_codes_for(
    page_line: str,
    by_page_line: dict[str, tuple[Elucidation, ...]] | None = None,
) -> set[str]:
    """Return set of espeak-ng language codes for non-English readings on this page.line."""
    if by_page_line is None:
        by_page_line = load_by_page_line()
    return {
        e.espeak_code
        for e in by_page_line.get(page_line, ())
        if e.espeak_code is not None
    }
