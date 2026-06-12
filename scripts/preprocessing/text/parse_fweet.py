#!/usr/bin/env python3
"""
Parse FWEET (Finnegans Wake Extensible Elucidation Treasury) HTML dump
into structured per-elucidation JSONL records keyed by page.line.

Source: papers/FWEET_elucidation — HTML search-results page from fweet.org,
provided by Raphael Slepon (maintainer) for this project. Contains the
complete elucidation database (~80–100k entries) as of Apr 2026.

Output schema (one record per elucidation):
  {
    "page_line": "004.19",
    "text": "Latin Helveticus: Swiss",        # plain-text body of <td>
    "tags": [                                  # all /cgi-bin/g?... lookups
      {"kind": "glossary", "code": "_L_", "label": "Latin"}
    ],
    "sigla": [],                               # sigla mentions (HCE, *E*, etc.)
    "cross_refs": ["004.24"],                  # references to other page.lines
    "html": "<i>...</i>"                       # raw <td> HTML for fallback
  }

Kind classification (heuristic from FWEET URL conventions):
  _M,X_       → motif
  _P,X_       → person
  <X>         → source/book reference
  _X_         → glossary entry (languages, registers, genres — disambiguate
                downstream with a known-language whitelist)
  other       → other

The "kind" of a tag does NOT yet identify language codes specifically; that's
a downstream concern that needs a known-good language-code list. Common
language codes observed: _F_ French, _G_ German, _L_ Latin, _It_ Italian,
_Heb_ Hebrew, _Cn_ Chinese, _Pi_ Chinese Pidgin, _Arch_ Archaic, _Leg_ Legalese.

Usage: python scripts/preprocessing/parse_fweet.py
"""

import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote

from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = REPO_ROOT / "papers/FWEET_elucidation"
OUTPUT_PATH = REPO_ROOT / "data/raw/fweet_elucidations.jsonl"

G_HREF_RE = re.compile(r"^/cgi-bin/g\?(.+)$")
F_HREF_RE = re.compile(r"^/cgi-bin/f\?(\d+\.\d+)$")


def classify_g_code(code: str) -> str:
    if code.startswith("_M,"):
        return "motif"
    if code.startswith("_P,"):
        return "person"
    if code.startswith("<") and code.endswith(">"):
        return "source"
    if code.startswith("_") and code.endswith("_") and "," not in code:
        return "glossary"
    return "other"


def parse_row(row) -> dict | None:
    th = row.find("th", class_="grep_th")
    if th is None:
        return None
    td = row.find("td")
    if td is None:
        return None

    page_line = None
    for a in th.find_all("a"):
        m = F_HREF_RE.match(a.get("href", ""))
        if m and a.get_text().startswith("."):
            page_line = m.group(1)
            break
    if page_line is None:
        return None

    tags: list[dict] = []
    cross_refs: list[str] = []
    for a in td.find_all("a"):
        href = a.get("href", "")
        gm = G_HREF_RE.match(href)
        if gm:
            code = unquote(gm.group(1))
            tags.append({
                "kind": classify_g_code(code),
                "code": code,
                "label": a.get_text(),
            })
            continue
        fm = F_HREF_RE.match(href)
        if fm:
            ref = fm.group(1)
            if ref != page_line and ref not in cross_refs:
                cross_refs.append(ref)

    sigla: list[str] = []
    for span in td.find_all("span", class_="all_sigla"):
        s = span.get_text()
        if s and s not in sigla:
            sigla.append(s)

    return {
        "page_line": page_line,
        "text": td.get_text(strip=True),
        "tags": tags,
        "sigla": sigla,
        "cross_refs": cross_refs,
        "html": td.decode_contents().strip(),
    }


def main() -> None:
    if not INPUT_PATH.exists():
        sys.stderr.write(f"Missing {INPUT_PATH}\n")
        sys.exit(1)

    size_mb = INPUT_PATH.stat().st_size // 1024 // 1024
    print(f"Loading {INPUT_PATH.name} ({size_mb} MB)...")
    html = INPUT_PATH.read_text(encoding="utf-8")
    print("Parsing HTML with lxml...")
    soup = BeautifulSoup(html, "lxml")
    print("Finding elucidation rows...")
    rows = soup.find_all("tr")
    print(f"  {len(rows):,} total <tr>; extracting elucidations...")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skipped = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            record = parse_row(row)
            if record is None:
                n_skipped += 1
                continue
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"\nWrote {n_written:,} elucidations to {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    print(f"Skipped {n_skipped:,} non-elucidation rows (headers/decorative)")


if __name__ == "__main__":
    main()
