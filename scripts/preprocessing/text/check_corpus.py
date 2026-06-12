#!/usr/bin/env python3
"""Verify data/raw/ against data/manifest.json. Exit 1 if any mismatch found."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "raw"
MANIFEST_PATH = ROOT / "data" / "manifest.json"


def main() -> None:
    if not MANIFEST_PATH.exists():
        print("ERROR: data/manifest.json not found. Run 'scrape_trent.py' first.")
        sys.exit(1)

    manifest = json.loads(MANIFEST_PATH.read_text())
    failures: list[str] = []

    for entry in manifest["chapters"]:
        path = DATA_DIR / entry["file"]
        if not path.exists():
            failures.append(f"  MISSING   {entry['file']}")
            continue
        got_l = sum(1 for _ in path.open(encoding="utf-8"))
        exp_l = entry["lines"]
        if got_l != exp_l:
            failures.append(
                f"  MISMATCH  {entry['file']}: "
                f"expected {exp_l} lines, got {got_l}"
            )

    if failures:
        print(f"Corpus check FAILED ({len(failures)} issue(s)):")
        for line in failures:
            print(line)
        sys.exit(1)

    t = manifest["totals"]
    print(f"Corpus OK — {t['chapters']} chapters, {t['lines']} lines")
    print(f"Source: {manifest.get('source', 'unknown')}")
    print(f"Scraped: {manifest['scraped_at']}")


if __name__ == "__main__":
    main()
