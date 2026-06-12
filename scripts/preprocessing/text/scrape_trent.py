#!/usr/bin/env python3
"""
Scrape Finnegans Wake from the archived Trent University digital edition
(Michael Groden / Tim Szeliga / Donald Theall, c.1999) via the Wayback Machine,
applying all known text corrections from the FWEET typo list
(http://www.fweet.org/pages/fw_typo.php).

Output: data/raw/book{NN}_ep{NN}.jsonl per chapter (17 files), one JSON
record per line, schema:
  {
    "book": int,
    "episode": int,
    "page": int,
    "line": int,
    "page_line": "NNN.NN",      # canonical 1939 F&F citation coordinate
    "text": str,                 # corrected text; [N] inline for footnote refs
    "left_margin": str|null,     # Book II Episode 2 only
    "right_margin": str|null     # Book II Episode 2 only
  }

Corrections applied from data/annotations/fweet_corrections.json:
  - substitution (311): text replacement at the specified page_line
  - placement (35): word moved from wrong adjacent line to correct line
  - range (9): stored in corrections JSON; complex cases flagged for manual review

Footnote body text is NOT in the Trent HTML. Inline [N] markers survive in
"text". For body text, run scrape_footnotes.py (telelib source).

Book II Episode 2 (pages 260-308) uses a four-column table:
  left margin (italic) | main text | right margin (all-caps) | line number
All other chapters use a two-column table:
  main text | line number

Usage:
    python scripts/preprocessing/scrape_trent.py
    python scripts/preprocessing/scrape_trent.py --force     # re-fetch existing
    python scripts/preprocessing/scrape_trent.py --delay 2.0 # seconds between requests
"""

import argparse
import copy
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"
MANIFEST_PATH = Path(__file__).resolve().parents[3] / "data" / "manifest.json"
CORRECTIONS_PATH = (
    Path(__file__).resolve().parents[3]
    / "data" / "annotations" / "fweet_corrections.json"
)

WAYBACK = "http://web.archive.org/web"
TRENT_BASE = "http://www.trentu.ca:80/faculty/jjoyce/fw-{n}.htm"
CDX_API = "http://web.archive.org/cdx/search/cdx"

# Chapter page ranges (inclusive) for the 1939 Faber & Faber edition.
# Pages 217-218, 400-402, 591-592 are inter-book blanks — not in Trent.
CHAPTERS = [
    (1, 1, 3,   29),
    (1, 2, 30,  47),
    (1, 3, 48,  74),
    (1, 4, 75,  103),
    (1, 5, 104, 125),
    (1, 6, 126, 168),
    (1, 7, 169, 195),
    (1, 8, 196, 216),
    (2, 1, 219, 259),
    (2, 2, 260, 308),
    (2, 3, 309, 382),
    (2, 4, 383, 399),
    (3, 1, 403, 428),
    (3, 2, 429, 473),
    (3, 3, 474, 554),
    (3, 4, 555, 590),
    (4, 1, 593, 628),
]

BLANK_PAGES = {217, 218, 400, 401, 402, 591, 592}
II2_PAGES = set(range(260, 309))


# ---------------------------------------------------------------------------
# Corrections
# ---------------------------------------------------------------------------

def load_corrections(path: Path) -> tuple[dict, dict, list]:
    """
    Return:
      substitutions: {page_line: (trent, correct, elaboration)}
      placements:    {wrong_page_line: (correct_page_line, word)}
      range_fixes:   list of raw dicts (applied manually / case-by-case)
    """
    raw = json.loads(path.read_text())["corrections"]
    substitutions: dict[str, tuple] = {}
    placements: dict[str, tuple] = {}
    range_fixes: list = []

    for c in raw:
        t = c["type"]
        if t == "substitution":
            substitutions[c["page_line"]] = (c["trent"], c["correct"], c.get("elaboration", ""))
        elif t == "placement":
            placements[c["wrong_page_line"]] = (c["correct_page_line"], c["word"])
        else:
            range_fixes.append(c)

    return substitutions, placements, range_fixes


def apply_substitution(text: str, trent: str, correct: str) -> tuple[str, bool]:
    """Replace first occurrence of trent with correct. Return (new_text, changed)."""
    if trent in text:
        return text.replace(trent, correct, 1), True
    return text, False


# ---------------------------------------------------------------------------
# CDX snapshot lookup
# ---------------------------------------------------------------------------

def fetch_snapshot_map(session: requests.Session) -> dict[int, str]:
    """Return {page_number: wayback_timestamp} using one CDX prefix query."""
    print("Fetching CDX snapshot index...", flush=True)
    r = session.get(CDX_API, params={
        "url": "www.trentu.ca/faculty/jjoyce/fw-",
        "matchType": "prefix",
        "output": "text",
        "fl": "original,timestamp",
        "filter": "statuscode:200",
        "collapse": "original",
        "limit": 5000,
    }, timeout=60)
    r.raise_for_status()

    page_ts: dict[int, str] = {}
    for line in r.text.strip().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        url, ts = parts[0], parts[1]
        m = re.search(r"/fw-(\d+)\.htm", url)
        if not m:
            continue
        n = int(m.group(1))
        if n < 3 or n > 628:
            continue
        if n not in page_ts or ts < page_ts[n]:
            page_ts[n] = ts

    print(f"  {len(page_ts)} pages indexed in Wayback Machine")
    return page_ts


# ---------------------------------------------------------------------------
# HTML fetch
# ---------------------------------------------------------------------------

def fetch_page(n: int, ts: str, session: requests.Session) -> BeautifulSoup | None:
    url = f"{WAYBACK}/{ts}/{TRENT_BASE.format(n=n)}"
    try:
        r = session.get(url, timeout=60)
        r.raise_for_status()
        return BeautifulSoup(r.text, "lxml")
    except requests.RequestException as exc:
        print(f"    WARN: failed to fetch page {n}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def cell_text(tag: Tag | None) -> str | None:
    """Return clean text from a BS4 tag, converting <sup>N</sup> to [N]."""
    if tag is None:
        return None
    tag = copy.copy(tag)
    for sup in tag.find_all("sup"):
        txt = sup.get_text(strip=True)
        if txt.isdigit():
            sup.replace_with(f"[{txt}]")
        else:
            sup.decompose()
    for script in tag.find_all("script"):
        script.decompose()
    text = re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip()
    return text or None


def page_line_from_href(href: str) -> tuple[int, int] | None:
    """Extract (page, line) from 'cgi-bin/row.cgi?FP=003.01...' href."""
    m = re.search(r"FP=(\d+)\.(\d+)", href)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


# ---------------------------------------------------------------------------
# Page parsers
# ---------------------------------------------------------------------------

def parse_regular_page(soup: BeautifulSoup, n: int) -> list[dict]:
    lines = []
    table = soup.find("table")
    if not table:
        return lines

    for tr in table.find_all("tr"):
        tds = tr.find_all("td", recursive=False)
        if len(tds) < 2:
            continue

        line_td = None
        for td in tds:
            a = td.find("a", href=re.compile(r"row\.cgi\?FP="))
            if a:
                line_td = (td, a)
                break
        if not line_td:
            continue

        td_lineno, a_lineno = line_td
        pl = page_line_from_href(a_lineno.get("href", ""))
        if not pl:
            continue
        page, line = pl
        if page != n:
            continue

        text_td = next((td for td in tds if td is not td_lineno), None)
        text = cell_text(text_td)
        if not text:
            continue

        lines.append({
            "page": page,
            "line": line,
            "page_line": f"{page:03d}.{line:02d}",
            "text": text,
            "left_margin": None,
            "right_margin": None,
        })

    return lines


def _is_line_number_link(href: str) -> bool:
    """True if href is the line-number column reference (not a T=L/R/F margin or footnote link)."""
    return bool(re.search(r"row\.cgi\?", href)) and not re.search(r"&T=[LRF]", href)


def parse_ii2_page(soup: BeautifulSoup, n: int) -> list[dict]:
    """Parse one II.2 HTML page.

    In II.2, the line-number column's FP coordinate reflects an internal Trent
    CGI flow index that does NOT match the physical printed page number. We use
    the file's page number (n) as the page coordinate and the digit text in the
    line-number cell as the line number, so that the resulting page_line values
    correspond to the 1939 F&F printed page.line used in Wake scholarship.

    Column layout: left_margin | main_text | right_margin | line_number
    Margin cells carry T=L / T=R links; footnote rows carry T=F links.
    The line-number cell carries any other row.cgi link (no T= or T=P).
    """
    lines = []
    table = soup.find("table")
    if not table:
        return lines

    for tr in table.find_all("tr"):
        tds = tr.find_all("td", recursive=False)
        if len(tds) < 2:
            continue

        # Find the rightmost TD that holds a line-number link (not T=L/R/F).
        line_td_idx = None
        for i, td in enumerate(tds):
            for a in td.find_all("a"):
                if _is_line_number_link(a.get("href", "")):
                    line_td_idx = i
                    break

        if line_td_idx is None:
            continue

        # Extract the line number from the text of the line-number cell.
        line_m = re.search(r"\d+", tds[line_td_idx].get_text())
        if not line_m:
            continue
        line = int(line_m.group())
        page = n  # physical printed page number, not the Trent CGI flow index

        content_tds = [tds[i] for i in range(line_td_idx)]
        if len(content_tds) == 1:
            text = cell_text(content_tds[0])
            left_margin = right_margin = None
        elif len(content_tds) >= 3:
            left_margin = cell_text(content_tds[0])
            text = cell_text(content_tds[1])
            right_margin = cell_text(content_tds[2])
        else:
            text = cell_text(content_tds[0]) if content_tds else None
            left_margin = right_margin = None

        if not text:
            continue

        lines.append({
            "page": page,
            "line": line,
            "page_line": f"{page:03d}.{line:02d}",
            "text": text,
            "left_margin": left_margin,
            "right_margin": right_margin,
        })

    return lines


# ---------------------------------------------------------------------------
# Correction application
# ---------------------------------------------------------------------------

def apply_corrections_to_page(
    lines: list[dict],
    substitutions: dict,
    placements: dict,
    applied_log: list,
) -> list[dict]:
    """Apply FWEET corrections to a list of line dicts. Mutates in place."""
    # Build index for placement lookups
    pl_index = {l["page_line"]: i for i, l in enumerate(lines)}

    for rec in lines:
        pl = rec["page_line"]

        # Simple substitution
        if pl in substitutions:
            trent, correct, elab = substitutions[pl]
            new_text, changed = apply_substitution(rec["text"], trent, correct)
            if changed:
                rec["text"] = new_text
                applied_log.append({
                    "type": "substitution",
                    "page_line": pl,
                    "from": trent,
                    "to": correct,
                    "elaboration": elab,
                })

        # Placement: word is on this (wrong) line; move it to the correct line
        if pl in placements:
            correct_pl, word = placements[pl]
            # Try to remove the word from this line
            if word in rec["text"]:
                rec["text"] = rec["text"].replace(word, "", 1).strip()
                # Add it to the correct line if present on this page
                if correct_pl in pl_index:
                    target = lines[pl_index[correct_pl]]
                    # Append to end of correct line (word wraps from line end)
                    target["text"] = target["text"].rstrip() + " " + word
                applied_log.append({
                    "type": "placement",
                    "wrong_page_line": pl,
                    "correct_page_line": correct_pl,
                    "word": word,
                })

    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape FW from Trent/Wayback edition")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if output exists")
    parser.add_argument("--delay", type=float, default=1.5, help="Seconds between requests")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Load FWEET corrections
    if CORRECTIONS_PATH.exists():
        substitutions, placements, range_fixes = load_corrections(CORRECTIONS_PATH)
        print(f"Loaded {len(substitutions)} substitutions, {len(placements)} placements, "
              f"{len(range_fixes)} range fixes from FWEET corrections.")
        if range_fixes:
            print(f"  Note: {len(range_fixes)} range corrections require manual review "
                  f"— see {CORRECTIONS_PATH.name}.")
    else:
        print("WARN: fweet_corrections.json not found — applying no corrections.")
        substitutions, placements, range_fixes = {}, {}, []

    session = requests.Session()
    session.headers["User-Agent"] = (
        "FinnegansWakeResearch/1.0 (academic; scraping Wayback Machine archive "
        "of Trent/Groden FW edition for page.line corpus construction)"
    )

    snapshot_map = fetch_snapshot_map(session)

    chapters_meta = []
    total_lines = 0
    all_corrections_applied: list[dict] = []

    for book, episode, page_start, page_end in CHAPTERS:
        out_path = RAW_DIR / f"book{book:02d}_ep{episode:02d}.jsonl"
        if out_path.exists() and not args.force:
            line_count = sum(1 for _ in out_path.open())
            print(f"  SKIP  {out_path.name} ({line_count} lines; use --force to re-fetch)")
            chapters_meta.append({
                "file": out_path.name,
                "book": book,
                "episode": episode,
                "page_range": [page_start, page_end],
                "lines": line_count,
            })
            total_lines += line_count
            continue

        print(f"  Scraping book{book:02d}_ep{episode:02d}  "
              f"(pages {page_start}–{page_end})...", flush=True)

        all_lines: list[dict] = []
        missing: list[int] = []
        chapter_corrections: list[dict] = []

        for n in range(page_start, page_end + 1):
            if n in BLANK_PAGES:
                continue

            ts = snapshot_map.get(n)
            if not ts:
                print(f"    WARN: no snapshot for page {n} — skipping")
                missing.append(n)
                time.sleep(args.delay)
                continue

            soup = fetch_page(n, ts, session)
            if soup is None:
                missing.append(n)
                time.sleep(args.delay)
                continue

            if n in II2_PAGES:
                page_lines = parse_ii2_page(soup, n)
            else:
                page_lines = parse_regular_page(soup, n)

            if not page_lines:
                print(f"    WARN: no lines parsed for page {n}")
                missing.append(n)
            else:
                apply_corrections_to_page(
                    page_lines, substitutions, placements, chapter_corrections
                )
                all_lines.extend(page_lines)

            time.sleep(args.delay)

        if missing:
            print(f"    Missing pages: {missing}")

        # Write JSONL — one record per line, chapter metadata on every record
        with out_path.open("w", encoding="utf-8") as f:
            for rec in all_lines:
                row = {"book": book, "episode": episode}
                row.update(rec)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        n_corrections = len(chapter_corrections)
        print(f"    → {len(all_lines)} lines, {n_corrections} corrections "
              f"applied → {out_path.name}")

        all_corrections_applied.extend(chapter_corrections)
        chapters_meta.append({
            "file": out_path.name,
            "book": book,
            "episode": episode,
            "page_range": [page_start, page_end],
            "lines": len(all_lines),
        })
        total_lines += len(all_lines)

    # Write corrections log
    corrections_log_path = RAW_DIR / "corrections_applied.json"
    corrections_log_path.write_text(
        json.dumps({
            "source": "FWEET typo list (http://www.fweet.org/pages/fw_typo.php)",
            "total_applied": len(all_corrections_applied),
            "corrections": all_corrections_applied,
        }, ensure_ascii=False, indent=2)
    )

    # Write manifest
    import datetime
    manifest = {
        "scraped_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": "Trent University / Michael Groden FW digital edition (c.1999), via Wayback Machine",
        "corrections": "FWEET typo list applied; see data/raw/corrections_applied.json",
        "edition": "1939 Faber & Faber",
        "schema_version": 3,
        "format": "jsonl",
        "totals": {
            "chapters": len(CHAPTERS),
            "lines": total_lines,
        },
        "chapters": chapters_meta,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\nDone. {total_lines} total lines across {len(CHAPTERS)} chapters.")
    print(f"{len(all_corrections_applied)} corrections applied "
          f"(log: data/raw/corrections_applied.json).")


if __name__ == "__main__":
    main()
