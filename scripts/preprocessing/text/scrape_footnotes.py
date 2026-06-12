#!/usr/bin/env python3
"""
Extract footnote body text from telelib.com and write per-chapter
footnotes files that companion the Trent JSONL corpus.

The Trent edition does not include footnote body text. Inline [N] markers
survive in the Trent JSONL text fields. This script pulls the body text
from telelib (the only machine-readable source that has it) and writes:

    data/raw/book{NN}_ep{NN}_footnotes.json

Schema:
    {
      "book": int,
      "episode": int,
      "source": "telelib.com",
      "footnotes": {
        "1": "footnote body text...",
        "2": "..."
      },
      "footnote_pages": {           # NEW: footnote N → page number where
        "1": 262, "2": 262, ...     # the [N] marker first appears in Trent.
      },
      "corrections_applied": [...]  # NEW: log of FWEET corrections applied
    }

FWEET corrections targeting footnotes (page_line containing 'F', e.g.
"262.F08") are loaded from data/annotations/fweet_corrections.json. Each
is applied via (page, per-page footnote ordinal) lookup; the per-page
ordinal is parsed from the elaboration's "(in footnote #N)". If no
ordinal is given, falls back to unique substring match within that page's
footnotes. Skipped corrections are recorded with a reason.

Chapters with no footnotes produce an empty "footnotes" dict.

Usage:
    python scripts/preprocessing/text/scrape_footnotes.py
    python scripts/preprocessing/text/scrape_footnotes.py --force
    python scripts/preprocessing/text/scrape_footnotes.py --delay 2.0
"""

import argparse
import copy
import html
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "data" / "raw"
FWEET_CORRECTIONS_PATH = REPO_ROOT / "data" / "annotations" / "fweet_corrections.json"

BASE_URL = "https://www.telelib.com/authors/J/JoyceJames/prose/finneganswake"

CHAPTERS = [
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),
    (2, 1), (2, 2), (2, 3), (2, 4),
    (3, 1), (3, 2), (3, 3), (3, 4),
    (4, 1),
]


def fetch(book: int, episode: int, session: requests.Session) -> BeautifulSoup:
    url = f"{BASE_URL}/finneganswake_{book:02d}{episode:02d}.html"
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "lxml")


def extract_footnotes(soup: BeautifulSoup) -> dict[str, str]:
    """Collect footnote texts keyed by footnote number (as string)."""
    notes: dict[str, str] = {}
    for a in soup.find_all("a", attrs={"name": re.compile(r"^foot\d+")}):
        n = re.search(r"\d+", a["name"]).group()
        parent = a.find_parent()
        if parent is None:
            continue
        parent = copy.copy(parent)
        for script in parent.find_all("script"):
            script.decompose()
        text = re.sub(r"^\s*\d+[\.\)]\s*", "", parent.get_text(" ", strip=True)).strip()
        if text:
            notes[n] = text
    return notes


def build_footnote_page_map(book: int, episode: int) -> dict[str, int]:
    """Map each footnote N (string key matching telelib's numbering) to the
    page where the [N] reference first appears in the Trent main text.

    Used to apply FWEET corrections whose locations are specified by page +
    per-page footnote ordinal.
    """
    raw_path = RAW_DIR / f"book{book:02d}_ep{episode:02d}.jsonl"
    mapping: dict[str, int] = {}
    if not raw_path.exists():
        return mapping
    marker_re = re.compile(r"\[(\d+)\]")
    for raw_line in raw_path.read_text(encoding="utf-8").splitlines():
        r = json.loads(raw_line)
        page = r["page"]
        for field in ("text", "left_margin", "right_margin"):
            t = r.get(field) or ""
            for m in marker_re.finditer(t):
                n = m.group(1)
                if n not in mapping:
                    mapping[n] = page
    return mapping


def load_fweet_footnote_corrections() -> list[dict]:
    """Load FWEET substitution corrections targeting footnotes (page_line contains 'F')."""
    if not FWEET_CORRECTIONS_PATH.exists():
        return []
    data = json.loads(FWEET_CORRECTIONS_PATH.read_text())
    return [
        c for c in data.get("corrections", [])
        if "F" in str(c.get("page_line", "")) and c.get("type") == "substitution"
    ]


def apply_footnote_corrections(
    footnotes: dict[str, str],
    page_map: dict[str, int],
    chapter_corrections: list[dict],
) -> tuple[dict[str, str], list[dict]]:
    """Apply per-page FWEET footnote corrections.

    Locates the target footnote via (page, per-page ordinal) from the
    correction's elaboration ("(in footnote #N)"). Falls back to unique
    string match within that page's footnotes if the ordinal isn't given.

    Returns (updated_footnotes, log).
    """
    page_to_fns: dict[int, list[str]] = {}
    for fn_n, page in page_map.items():
        page_to_fns.setdefault(page, []).append(fn_n)
    for p in page_to_fns:
        page_to_fns[p].sort(key=int)

    out = dict(footnotes)
    log: list[dict] = []
    fn_num_re = re.compile(r"in footnote #(\d+)")
    page_re = re.compile(r"^(\d+)\.F\d+")

    for corr in chapter_corrections:
        pl = corr.get("page_line", "")
        pm = page_re.match(pl)
        if not pm:
            log.append({"page_line": pl, "status": "skipped", "reason": "unparseable page_line"})
            continue
        page = int(pm.group(1))
        trent = html.unescape(corr.get("trent", ""))
        correct = html.unescape(corr.get("correct", ""))
        elab = corr.get("elaboration", "")

        global_fn: str | None = None
        m = fn_num_re.search(elab)
        if m:
            per_page_n = int(m.group(1))
            page_fns = page_to_fns.get(page, [])
            if 1 <= per_page_n <= len(page_fns):
                global_fn = page_fns[per_page_n - 1]
        else:
            candidates = [fn for fn in page_to_fns.get(page, []) if trent in out.get(fn, "")]
            if len(candidates) == 1:
                global_fn = candidates[0]

        if global_fn and global_fn in out and trent in out[global_fn]:
            out[global_fn] = out[global_fn].replace(trent, correct, 1)
            log.append({
                "page_line": pl,
                "status": "applied",
                "footnote_key": global_fn,
                "page": page,
                "from": trent,
                "to": correct,
            })
        else:
            log.append({
                "page_line": pl,
                "status": "skipped",
                "reason": "target not located or trent string not in candidate footnote",
                "page": page,
                "from": trent,
                "to": correct,
            })

    return out, log


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract FW footnote body text from telelib")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if output exists")
    parser.add_argument("--delay", type=float, default=1.5, help="Seconds between requests")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_corrections = load_fweet_footnote_corrections()
    print(f"Loaded {len(all_corrections)} FWEET footnote corrections from annotations.")

    session = requests.Session()
    session.headers["User-Agent"] = (
        "FinnegansWakeResearch/1.0 (academic; extracting footnote body text from telelib)"
    )

    total_footnotes = 0
    total_applied = 0

    for book, episode in CHAPTERS:
        out_path = RAW_DIR / f"book{book:02d}_ep{episode:02d}_footnotes.json"
        if out_path.exists() and not args.force:
            data = json.loads(out_path.read_text())
            n = len(data["footnotes"])
            print(f"  SKIP  {out_path.name} ({n} footnotes)")
            total_footnotes += n
            continue

        print(f"  Fetching footnotes for book{book:02d}_ep{episode:02d}...", flush=True)
        try:
            soup = fetch(book, episode, session)
            footnotes = extract_footnotes(soup)
        except requests.RequestException as exc:
            print(f"    WARN: {exc}")
            footnotes = {}

        page_map = build_footnote_page_map(book, episode)
        chapter_pages = set(page_map.values())
        chapter_corrections = [
            c for c in all_corrections
            if int(re.match(r"^(\d+)\.", c["page_line"]).group(1)) in chapter_pages
        ]
        footnotes_corrected, correction_log = apply_footnote_corrections(
            footnotes, page_map, chapter_corrections
        )
        n_applied = sum(1 for x in correction_log if x["status"] == "applied")
        total_applied += n_applied

        out = {
            "book": book,
            "episode": episode,
            "source": "telelib.com",
            "footnotes": footnotes_corrected,
            "footnote_pages": page_map,
            "corrections_applied": correction_log,
        }
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
        applied_str = f", {n_applied} corrections applied" if chapter_corrections else ""
        print(f"    → {len(footnotes_corrected)} footnotes{applied_str} → {out_path.name}")
        total_footnotes += len(footnotes_corrected)
        time.sleep(args.delay)

    print(f"\nDone. {total_footnotes} total footnotes across {len(CHAPTERS)} chapters.")
    print(f"FWEET corrections applied: {total_applied} / {len(all_corrections)}")


if __name__ == "__main__":
    main()
