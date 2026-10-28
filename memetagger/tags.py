"""Tag names, tag kinds and normalization. No ML and no Qt here."""
from __future__ import annotations
import re
from dataclasses import dataclass

KINDS = ("medium", "franchise", "character", "meme_format", "subject", "theme", "other")
KIND_LABELS = {"medium": "Medium", "franchise": "Franchise", "character": "Character", "meme_format": "Meme",
               "subject": "Subject", "theme": "Theme", "other": "Other"}
# File names list tags from general to specific; folders and quotas use kinds, so the order here is only cosmetic.
DISPLAY_ORDER = {k: i for i, k in enumerate(KINDS)}

MEDIUMS = ("anime", "manga", "cartoon", "comic", "drawing", "painting", "3d_render", "pixel_art", "ai_generated",
           "photo", "screenshot", "game_screenshot", "other")
ART_MEDIUMS = frozenset({"anime", "manga", "cartoon", "comic", "drawing", "painting", "3d_render", "pixel_art",
                         "ai_generated", "game_screenshot"})
STOP = frozenset({"meme", "memes", "image", "picture", "text", "funny", "humor", "humour", "caption", "other", "none",
                  "unknown", "illustration", "art", "photo_of", "screenshot_of", "n_a", "na", "null"})

_TOKEN_SPLIT = re.compile(r"[\W_]+")


@dataclass
class Tag:
    name: str
    kind: str = "other"
    score: float = 1.0
    source: str = ""          # vlm | wd14 | folder | derived | classic | user


def clean_tag(raw: str) -> str:
    """'Re:Zero' -> 're_zero', 'K-On!' -> 'k-on', 'Fate/Grand Order' -> 'fate_grand_order'."""
    t = (raw or "").strip().lower()
    t = re.sub(r"[\s/:;.&+]+", "_", t)
    t = re.sub(r"[^\w\-]", "", t)
    t = re.sub(r"_{2,}", "_", t)
    return t.strip("_-")


def tag_key(tag: str) -> str:
    """Spelling-insensitive key: re_zero / rezero / Re-Zero -> 'rezero'."""
    return _TOKEN_SPLIT.sub("", (tag or "").lower()) or (tag or "")


def name_key(tag: str) -> str:
    """Order-insensitive key for person names: yui_hirasawa and hirasawa_yui -> 'hirasawa yui'."""
    return " ".join(sorted(t for t in _TOKEN_SPLIT.split((tag or "").lower()) if t))


def singular_key(key: str) -> str | None:
    """Candidate singular form of a tag key (cats -> cat, puppies -> puppy). Only used when that form is already known."""
    if len(key) > 4 and key.endswith("ies"):
        return key[:-3] + "y"
    if len(key) > 3 and key.endswith("s") and not key.endswith("ss"):
        return key[:-1]
    return None


def squash(s: str) -> str:
    return _TOKEN_SPLIT.sub("", (s or "").lower())


def order_tags(tags: list[Tag]) -> list[Tag]:
    return sorted(tags, key=lambda t: DISPLAY_ORDER.get(t.kind, len(KINDS)))
