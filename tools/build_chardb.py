"""Builds memetagger/data/characters.tsv.gz from the Danbooru character table of Laxhar/noob-wiki.

    python tools/build_chardb.py [path/to/danbooru_character_webui.csv]

Without an argument the CSV is downloaded from Hugging Face. Characters with at least MIN_POSTS posts are kept,
plus every character the WD14 v3 taggers know (so their franchise is always available)."""
import csv, gzip, sys, urllib.request
from pathlib import Path

SRC = "https://huggingface.co/datasets/Laxhar/noob-wiki/resolve/main/danbooru_character_webui.csv"
WD_TAGS = "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/selected_tags.csv"
MIN_POSTS = 50
OUT = Path(__file__).resolve().parent.parent / "memetagger" / "data" / "characters.tsv.gz"


def _lines(src: str):
    if Path(src).exists():
        return Path(src).read_text("utf-8").splitlines()
    with urllib.request.urlopen(src) as r:
        return r.read().decode("utf-8").splitlines()


def main():
    wd = {row["name"] for row in csv.DictReader(_lines(WD_TAGS)) if row["category"] == "4"}
    rows = []
    for row in csv.DictReader(_lines(sys.argv[1] if len(sys.argv) > 1 else SRC)):
        count = int(row["count"] or 0)
        if count >= MIN_POSTS or row["character"] in wd:
            rows.append((row["character"], row["copyright"] if row["copyright"] != "original" else "", count))
    rows.sort(key=lambda r: -r[2])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT, "wt", encoding="utf-8", compresslevel=9) as f:
        f.write("# character\tcopyright\tposts  (Danbooru, via Laxhar/noob-wiki)\n")
        for name, cop, count in rows:
            f.write(f"{name}\t{cop}\t{count}\n")
    print(f"{len(rows)} characters -> {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
