"""Turns the vision model's answer, WD14 candidates and context into the final tag list.

Pure logic without I/O, so every rule is unit-tested. The flow per image:
  claims()   -> characters/franchises proposed by the vision model and by WD14
  triage()   -> accept now / ask one batched verification question / drop
  resolve()  -> apply the verification verdicts
  assemble() -> add medium, memes, subjects, themes, folder tags; normalize; apply per-kind quotas."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from .characters import CharacterDB
from .tags import ART_MEDIUMS, MEDIUMS, STOP, Tag, clean_tag, order_tags, singular_key, squash, tag_key
from .tagstore import TagStore
from .wd14 import WdResult

STRICTNESS = ("lenient", "balanced", "strict")
QUOTAS = {"character": 3, "franchise": 2, "meme_format": 2, "medium": 1, "subject": 3, "theme": 2, "other": 3}
# When max_tags is reached, kinds earlier in this list win (characters are what the user searches for most).
PRIORITY = ("character", "franchise", "meme_format", "medium", "subject", "theme", "other")
ACCEPT, VERIFY_NOT_NO, VERIFY_YES, DROP = "accept", "verify_not_no", "verify_yes", "drop"
RULES = {   # unsupported claim, by model confidence
    "lenient": {"high": ACCEPT, "medium": ACCEPT, "low": VERIFY_YES},
    "balanced": {"high": ACCEPT, "medium": VERIFY_NOT_NO, "low": VERIFY_YES},
    "strict": {"high": VERIFY_NOT_NO, "medium": VERIFY_YES, "low": DROP},
}
VLM_FIELDS = (("meme_format", "meme_format"), ("subjects", "subject"), ("themes", "theme"))


@dataclass
class Claim:
    kind: str                       # character | franchise
    tag: str
    confidence: str = "medium"      # high | medium | low
    source: str = "vlm"             # vlm | wd14
    franchise: str | None = None    # for characters: the work they come from
    score: float = 0.0              # WD14 probability
    support: str = ""               # why it needs no verification
    rule: str = ""

    def describe(self) -> str:
        bits = [self.source, self.confidence if self.source == "vlm" else f"{self.score:.2f}"]
        if self.support:
            bits.append(f"supported by {self.support}")
        return f"{self.tag} ({', '.join(bits)})"


@dataclass
class Evidence:
    text: str = ""                                      # meaningful file name + OCR text
    folder_tags: list[Tag] = field(default_factory=list)
    wd: WdResult | None = None
    medium: str | None = None


@dataclass
class Policy:
    strictness: str = "balanced"
    wd_trust: float = 0.85          # WD14 probability accepted without the vision model's agreement
    max_tags: int = 8
    banned: frozenset = frozenset()
    folder_tags_all: bool = False   # also add folder tags that are not franchises/characters


@dataclass
class Triage:
    accepted: list[Claim] = field(default_factory=list)
    to_verify: list[Claim] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _words(text: str) -> set[str]:
    return {w for w in re.split(r"[\W_]+", text.lower()) if w}


class Fusion:
    def __init__(self, store: TagStore, chardb: CharacterDB, parents: dict | None = None, policy: Policy | None = None):
        self.store, self.db, self.parents, self.policy = store, chardb, parents or {}, policy or Policy()

    # ---- normalization ----
    def norm(self, raw, kind: str) -> str:
        if not isinstance(raw, str):
            return ""
        t = clean_tag(raw)
        if not t or t in STOP or len(t) > 40:            # overlong "tags" are sentences, not tags
            return ""
        if kind == "franchise":
            t = self.db.known_franchise(t) or t
        return self.store.canonical(t, kind)

    def franchise_of(self, character: str) -> str | None:
        for fr in (self.db.franchise_of(character), self.store.franchise_of(character)):
            if fr:
                return self.store.canonical(fr, "franchise")
        return next((self.norm(p, "franchise") for p in self.parents.get(character, []) if p), None)

    # ---- claims ----
    def claims(self, data: dict | None, wd: WdResult | None) -> tuple[list[Claim], list[Claim]]:
        vlm, seen = [], set()
        data = data or {}
        for item in _as_list(data.get("characters")):
            name, fr, conf = _item(item)
            if not name:
                continue
            m = self.db.match(name, fr or None)
            if m:
                tag, fr_tag = self.store.canonical(m[0], "character"), m[1] and self.store.canonical(m[1], "franchise")
            else:
                tag, fr_tag = self.norm(name, "character"), self.norm(fr, "franchise") if fr else None
            if tag and tag not in seen:
                seen.add(tag); vlm.append(Claim("character", tag, conf, "vlm", fr_tag or self.franchise_of(tag)))
        for item in _as_list(data.get("franchise")):
            name, _, conf = _item(item)
            tag = self.norm(name, "franchise")
            if tag and tag not in seen:
                seen.add(tag); vlm.append(Claim("franchise", tag, conf, "vlm"))
        wdc = []
        for c in (wd.characters if wd else []):
            tag = self.store.canonical(c.tag, "character")
            fr = self.store.canonical(c.franchise, "franchise") if c.franchise else self.franchise_of(tag)
            wdc.append(Claim("character", tag, "high" if c.score >= self.policy.wd_trust else "low", "wd14", fr, c.score))
        return vlm, wdc

    # ---- decisions ----
    def triage(self, vlm: list[Claim], wd: list[Claim], ev: Evidence, have_vlm: bool = True) -> Triage:
        out, p = Triage(), self.policy
        wd_by_tag = {c.tag: c for c in wd}
        words, squashed = _words(ev.text), squash(ev.text)
        folder = {t.name for t in ev.folder_tags}
        art = ev.medium is None or ev.medium in ART_MEDIUMS or ev.medium == "other"

        def text_support(tag: str) -> bool:
            k = tag_key(tag)
            if len(k) >= 6 and k in squashed:
                return True
            toks = [t for t in re.split(r"[\W_]+", tag) if len(t) >= 3]
            return bool(toks) and all(t in words for t in toks)

        vlm_chars = {c.tag for c in vlm if c.kind == "character"}
        for c in wd:                                         # WD14 is reliable for anime art when it is confident
            if c.tag in vlm_chars:
                continue
            if c.score >= p.wd_trust and (art or not have_vlm):
                c.support = "WD14"; out.accepted.append(c)
            elif have_vlm:
                out.notes.append(f"WD14 candidate not confirmed: {c.describe()}")

        strong_fr = {t.name for t in ev.folder_tags if t.kind == "franchise"}
        strong_fr |= {c.franchise for c in out.accepted if c.franchise}
        for c in sorted(vlm, key=lambda c: c.kind != "character"):      # characters first: they support franchises
            w = wd_by_tag.get(c.tag)
            if c.kind == "character":
                if w is not None:
                    c.support = f"WD14 {w.score:.2f}"
                elif c.tag in folder:
                    c.support = "folder"
                elif text_support(c.tag):
                    c.support = "text"
                elif c.franchise and c.franchise in strong_fr and self.db.is_character(c.tag):
                    c.support = "franchise"
            else:
                char_fr = {x.franchise for x in out.accepted if x.kind == "character" and x.franchise}
                if c.tag in folder or c.tag in strong_fr:
                    c.support = "folder" if c.tag in folder else "WD14"
                elif c.tag in char_fr:
                    c.support = "character"
                elif text_support(c.tag):
                    c.support = "text"
            c.rule = ACCEPT if c.support else RULES.get(p.strictness, RULES["balanced"]).get(c.confidence, VERIFY_YES)
            if c.rule == ACCEPT:
                out.accepted.append(c)
                if c.kind == "character" and c.franchise and c.support:
                    strong_fr.add(c.franchise)
            elif c.rule == DROP:
                out.notes.append(f"dropped {c.describe()}: low confidence")
            else:
                out.to_verify.append(c)
        return out

    def resolve(self, tri: Triage, verdicts: list[str] | None) -> Triage:
        """verdicts: one per tri.to_verify claim, or None when verification is off (treated as 'unsure')."""
        verdicts = verdicts or ["unsure"] * len(tri.to_verify)
        for c, v in zip(tri.to_verify, verdicts):
            keep = v == "yes" or (c.rule == VERIFY_NOT_NO and v != "no")
            if keep:
                c.support = c.support or f"check: {v}"; tri.accepted.append(c)
            else:
                tri.notes.append(f"dropped {c.describe()}: check said {v}")
        tri.to_verify = []
        return tri

    # ---- final list ----
    def assemble(self, data: dict | None, accepted: list[Claim], ev: Evidence, extra: list[Tag] = ()) -> list[Tag]:
        p, data = self.policy, data or {}
        cands: list[Tag] = []
        for c in accepted:
            cands.append(Tag(c.tag, c.kind, c.score or 1.0, c.source))
        for c in accepted:                                   # an accepted character implies its franchise
            if c.kind == "character" and c.franchise:
                cands.append(Tag(c.franchise, "franchise", 1.0, "derived"))
        for t in ev.folder_tags:
            if t.kind in ("franchise", "character") or p.folder_tags_all:
                cands.append(Tag(t.name, t.kind, 1.0, "folder"))
        medium = data.get("medium")
        if isinstance(medium, str) and medium in MEDIUMS and medium != "other":
            cands.append(Tag(medium, "medium", 1.0, "vlm"))
        for key, kind in VLM_FIELDS:
            for raw in _as_list(data.get(key)):
                if (t := self.norm(raw, kind)):
                    cands.append(Tag(t, kind, 1.0, "vlm"))
        cands += [Tag(self.norm(t.name, t.kind) or t.name, t.kind, t.score, t.source) for t in extra]

        chosen, keys, per_kind = [], set(), {}
        for kind in PRIORITY:
            for t in cands:
                if t.kind != kind and not (kind == "other" and t.kind not in PRIORITY):
                    continue
                k = tag_key(t.name)
                plain = t.kind in ("subject", "theme", "other")
                forms = {k, singular_key(k)} - {None} if plain else {k}
                if not t.name or t.name in STOP or t.name in p.banned or forms & keys:
                    continue
                if per_kind.get(t.kind, 0) >= QUOTAS.get(t.kind, 2) or len(chosen) >= p.max_tags:
                    continue
                chosen.append(t); keys |= forms; per_kind[t.kind] = per_kind.get(t.kind, 0) + 1
        return order_tags(chosen)


def _as_list(v) -> list:
    return [v] if isinstance(v, (str, dict)) else v if isinstance(v, list) else []


def _item(item) -> tuple[str, str, str]:
    """A character/franchise entry -> (name, franchise, confidence); tolerates plain strings from older models."""
    if isinstance(item, str):
        return item.strip(), "", "medium"
    if isinstance(item, dict):
        conf = str(item.get("confidence") or "medium").lower()
        return str(item.get("name") or "").strip(), str(item.get("franchise") or "").strip(), conf if conf in ("high", "medium", "low") else "medium"
    return "", "", "medium"
