# Corpus Annotations

Supplementary annotations that layer over the scraped corpus in `data/raw/`. These files are tracked in git; the scraped JSON files are not. Projects should load annotations via `shared/annotations.py`.

## Sigla Unicode Reference

The full cast of Joyce's drawn sigla and their Unicode representations. These characters must be treated as atomic tokens — never normalized, decomposed, or mapped to look-alikes.

Referents confirmed from the Hart Concordance Symbols page (299.F4 sequence) and McHugh (1976). Unicode approximations are the best available single codepoints — the 1939 characters are hand-drawn and none have exact Unicode equivalents.

| Siglum | Unicode | Referent | McHugh ASCII | Note |
|--------|---------|----------|--------------|------|
| Ш | U+0428 | HCE — M-form (299.F4 pos. 1) | M | McHugh p.144: "call M Earwicker". Distinct from Ǝ below — HCE has multiple printed forms |
| Ǝ | U+018E | HCE E-form; also I.2 036.17 (referent unresolved) | E / 4 | I.5 119.17: "E pointing down", text: "chrismon trilithon sign…Hec". I.2 036.17: same visual form confirmed by 1939 screenshot; McHugh "used by M to address the cad" — whether HCE marks his own speech or a distinct usage is unresolved. Do not substitute ∃ (U+2203) |
| △ | U+25B3 | ALP (Anna Livia Plurabelle) | A | Preferred over Δ (U+0394 GREEK CAPITAL LETTER DELTA) |
| ⊣ | U+22A3 | Issy — 299.F4 pos. 3 only | 4 | McHugh uses "4" for both this and Ǝ at I.2 036.17 — they are visually distinct characters. ⊣ applies only to 299.F4 position 3 |
| ✕ | U+2715 | The Four (four evangelists) | ¥ | Concordance 299.F4 pos. 4. Cross shape = fourfold structure. NOT Shem |
| □ | U+25A1 | House (Earwicker inn) | O | Concordance 299.F4 pos. 5. McHugh: "their old fourwheedler". NOT Shaun the person |
| ∧ | U+2227 | Shaun the Post | A | Concordance 299.F4 pos. 6. McHugh uses A a second time (same as ALP). NOT Issy |
| ⌐ | U+2310 | Shem the Penman | £ / C | Concordance 299.F4 pos. 7. McHugh p.143: "C here is the family gibbet" |
| Ⅎ | U+2132 | F-variant (referent unresolved) | — | McHugh p.145: "their mystery remains closed". Laterally inverted F-pairs at 018.36, 121.03, 121.07, 266.22. Not in 299.F4 family listing |

**Note on I.6:** The Hart Concordance lists no sigla in the I.6 page range (126–168) and McHugh (1976) pp. 133–134 confirms none were printed in the 1939 Faber & Faber edition. The sigla system is the analytical framework scholars apply to I.6's speaker structure; it lives in the manuscripts and McHugh's apparatus, not on the printed page. There is no I.6 annotation file.

## Schema

Each annotation file is `book{NN}_ep{NN}_sigla.json`. Top-level arrays vary by chapter; all files have `annotations` (confirmed located sigla) and `sigla_legend` (chapter-local subset of the reference table above). Additional arrays used where relevant: `drawings`, `music`, `math`, `greek`.

```json
{
  "book": int,
  "episode": int,
  "sigla_legend": { "<unicode_char>": { "referent": "...", "unicode": "U+XXXX", ... } },
  "annotations": [
    {
      "page_line": "NNN.NN",
      "char_offset": int,
      "siglum": "<unicode_char> or UNCONFIRMED",
      "note": "source page/line reference or editorial note"
    }
  ]
}
```

`page_line` is the canonical 1939 Faber & Faber page.line citation (e.g. `"119.17"`), matching the Hart Concordance reference system used throughout Wake scholarship. `char_offset` is 0-based within the Trent corpus `line.text` for that page_line; -1 means the offset has not yet been verified against the Trent corpus. Use `"UNCONFIRMED"` as the `siglum` value when a gap is detected in the corpus but the siglum's identity has not been verified against the 1939 edition.

## Status

All entries sourced from the Hart Concordance (rosenlake.net) Symbols/Drawings/Music/Math/Greek pages, cross-referenced with McHugh (1976).

| Chapter | Page.Line | Description | File | Status |
|---------|-----------|-------------|------|--------|
| I.1 | 018.36 | F down/up pair | `book01_ep01_sigla.json` | Located: s51 @1269 and @1276 — initial letters of "face to face"; no gap in corpus |
| I.2 | 036.17 | Ǝ (upside-down backwards E) | `book01_ep02_sigla.json` | Located: s1 @3794 — no space gap (precedes "!") |
| I.2 | 044.25 | Ballad of Persse O'Reilly (sheet music) | `book01_ep02_sigla.json` | Completely absent (drawn image) |
| I.5 | 119.17 | Ǝ (HCE) | `book01_ep05_sigla.json` | Located: s20 @515 |
| I.5 | 119.18 | △ (ALP) | `book01_ep05_sigla.json` | Located: s20 @641 |
| I.5 | 121.03 | upside-down backwards F | `book01_ep05_sigla.json` | Located: s20 @4117 |
| I.5 | 121.07 | upside-down F | `book01_ep05_sigla.json` | Located: s20 @4408 (double-comma pattern) |
| I.5 | 124.08 | arrow (→) | `book01_ep05_sigla.json` | Confirmed in corpus: s22 @911 |
| I.5 | 124.09 | lambda (Λ) | `book01_ep05_sigla.json` | Located: s22 @944 |
| I.5 | 124.10 | equals (=) | `book01_ep05_sigla.json` | Confirmed in corpus: s22 @1030 (spaces stripped) |
| II.2 | 266.22 | F, backwards F (Ⅎ) | `book02_ep02_sigla.json` | Located: s21 @96 |
| II.2 | 269.24 | Greek (ouk elabon polin) | `book02_ep02_sigla.json` | Confirmed in corpus |
| II.2 | 272.09 | B C A D (left margin music) | `book02_ep02_sigla.json` | Completely absent |
| II.2 | 284.11 | ∞ (infinity) | `book02_ep02_sigla.json` | Confirmed in corpus |
| II.2 | 292.11 | ∴ (therefore) | `book02_ep02_sigla.json` | Confirmed in corpus |
| II.2 | 292.12 | ∵ (because) | `book02_ep02_sigla.json` | Confirmed in corpus |
| II.2 | 293.12 | view from Dublin (drawing) | `book02_ep02_sigla.json` | Completely absent (drawn image) |
| II.2 | 299.F4 | Full siglum set (footnote 188) | `book02_ep02_sigla.json` | Sequence Ш,△,⊣,✕,□,∧,⌐ — all 7 referents confirmed; Unicode approx uncertain for Ш, ⊣, ⌐ |
| II.2 | 308.F1 | nose thumbing (drawing) | `book02_ep02_sigla.json` | Completely absent (drawn image) |
| II.2 | 308.F2 | crossed spoons (drawing) | `book02_ep02_sigla.json` | Completely absent (drawn image) |
