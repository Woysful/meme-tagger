import pytest
from memetagger.characters import CharacterDB
from memetagger.fusion import Evidence, Fusion, Policy
from memetagger.tags import Tag
from memetagger.tagstore import TagStore
from memetagger.wd14 import WdCandidate, WdResult

DB = CharacterDB([("hirasawa_yui", "k-on!", 8000), ("akiyama_mio", "k-on!", 9000), ("rem_(re:zero)", "re:zero_kara_hajimeru_isekai_seikatsu", 8000),
                  ("hakurei_reimu", "touhou", 70000), ("paimon_(genshin_impact)", "genshin_impact", 5000)])


def fusion(**kw):
    return Fusion(TagStore(), DB, {"aigis": ["persona"]}, Policy(**kw))


def run(f, data, wd=None, ev=None, verdicts=None, have_vlm=True):
    ev = ev or Evidence(wd=wd)
    ev.medium = (data or {}).get("medium", ev.medium)
    vlm, wdc = f.claims(data, wd)
    tri = f.triage(vlm, wdc, ev, have_vlm)
    asked = [c.tag for c in tri.to_verify]
    tri = f.resolve(tri, [verdicts.get(t, "unsure") for t in asked] if verdicts is not None else None)
    return [t.name for t in f.assemble(data, tri.accepted, ev)], asked, tri.notes


def ch(name, fr="", conf="high"):
    return {"name": name, "franchise": fr, "confidence": conf}


def test_high_confidence_character_is_normalized_and_implies_franchise():
    tags, asked, _ = run(fusion(), {"medium": "anime", "characters": [ch("Yui Hirasawa", "K-On!")], "franchise": []})
    assert tags == ["anime", "k-on", "hirasawa_yui"] and asked == []


def test_strong_wd14_candidate_accepted_on_art_only():
    wd = WdResult([WdCandidate("hakurei_reimu", "touhou", 0.95)])
    assert run(fusion(), {"medium": "anime", "characters": []}, wd)[0] == ["anime", "touhou", "hakurei_reimu"]
    assert run(fusion(), {"medium": "photo", "characters": []}, wd)[0] == ["photo"]      # cosplay photo != the character
    weak = WdResult([WdCandidate("hakurei_reimu", "touhou", 0.5)])
    tags, _, notes = run(fusion(), {"medium": "anime", "characters": []}, weak)
    assert tags == ["anime"] and any("not confirmed" in n for n in notes)


def test_weak_wd14_candidate_supports_the_vision_model():
    wd = WdResult([WdCandidate("rem", "re_zero", 0.4)])
    tags, asked, _ = run(fusion(), {"medium": "anime", "characters": [ch("Rem", "Re:Zero", "low")]}, wd)
    assert tags == ["anime", "re_zero", "rem"] and asked == []


@pytest.mark.parametrize("strictness,conf,verdict,kept,asked", [
    ("balanced", "medium", "unsure", True, True), ("balanced", "medium", "no", False, True),
    ("balanced", "low", "unsure", False, True), ("balanced", "low", "yes", True, True),
    ("lenient", "medium", "no", True, False), ("strict", "high", "unsure", True, True),
    ("strict", "medium", "unsure", False, True), ("strict", "low", "yes", False, False),
])
def test_strictness_table(strictness, conf, verdict, kept, asked):
    data = {"medium": "anime", "characters": [ch("Akiyama Mio", "K-On!", conf)]}
    tags, q, _ = run(fusion(strictness=strictness), data, verdicts={"akiyama_mio": verdict})
    assert ("akiyama_mio" in tags) == kept and bool(q) == asked


def test_verification_off_counts_as_unsure():
    data = {"medium": "anime", "characters": [ch("Akiyama Mio", "K-On!", "medium"), ch("Rem", "Re:Zero", "low")]}
    tags, asked, _ = run(fusion(), data, verdicts=None)
    assert "akiyama_mio" in tags and "rem" not in tags and set(asked) == {"akiyama_mio", "rem"}


def test_text_and_folder_support():
    data = {"medium": "anime", "characters": [ch("Rem", "Re:Zero", "low")], "franchise": [{"name": "Touhou", "confidence": "low"}]}
    ev = Evidence(text="Rem best girl", folder_tags=[Tag("touhou", "franchise", 1, "folder")])
    tags, asked, _ = run(fusion(), data, ev=ev)
    assert {"rem", "re_zero", "touhou"} <= set(tags) and asked == []
    assert not run(fusion(), data, ev=Evidence(text="remember this"))[0].count("rem")   # no substring matches


def test_franchise_supported_by_its_character():
    data = {"medium": "anime", "characters": [ch("Paimon", "Genshin Impact", "high")],
            "franchise": [{"name": "Genshin Impact", "confidence": "low"}]}
    tags, asked, _ = run(fusion(), data)
    assert tags == ["anime", "genshin_impact", "paimon"] and asked == []


def test_quotas_priority_and_max_tags():
    data = {"medium": "anime", "characters": [ch("Yui Hirasawa", "K-On!"), ch("Akiyama Mio", "K-On!")],
            "meme_format": ["wojak"], "subjects": ["guitar", "girl", "stage", "amp"], "themes": ["music", "school", "friends"]}
    tags = run(fusion(max_tags=6), data)[0]
    assert tags == ["anime", "k-on", "hirasawa_yui", "akiyama_mio", "wojak", "guitar"]    # characters never cut first
    assert len(run(fusion(max_tags=20), data)[0]) == 1 + 1 + 2 + 1 + 3 + 2               # per-kind quotas


def test_folder_tags_kinds_banned_stop_and_dedupe():
    ev = Evidence(folder_tags=[Tag("genshin_impact", "franchise", 1, "folder"), Tag("sad", "other", 1, "folder")])
    data = {"medium": "anime", "subjects": ["Meme", "cat", "Cats", "dog"], "themes": ["gaming"]}
    tags = run(fusion(banned=frozenset({"dog"})), data, ev=ev)[0]
    assert tags == ["anime", "genshin_impact", "cat", "gaming"]          # cat/Cats collapse; meme, dog, sad are gone
    assert "sad" in run(fusion(folder_tags_all=True), data, ev=ev)[0]


def test_plain_strings_from_older_models_and_aliases():
    data = {"medium": "anime", "characters": ["Aigis", "Hakurei Reimu"], "franchise": "Touhou"}
    tags, asked, _ = run(fusion(), data, verdicts={"aigis": "yes"})
    assert {"aigis", "persona", "hakurei_reimu", "touhou"} <= set(tags)
