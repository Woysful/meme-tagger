from memetagger.tags import clean_tag, name_key, order_tags, singular_key, tag_key, Tag
from memetagger.tagstore import TagStore


def test_clean_tag():
    assert clean_tag("Re:Zero") == "re_zero"
    assert clean_tag("K-On!") == "k-on"
    assert clean_tag("Fate/Grand Order") == "fate_grand_order"
    assert clean_tag("  Dr. Stone  ") == "dr_stone"
    assert clean_tag("this is fine") == "this_is_fine"
    assert clean_tag("") == "" and clean_tag(None) == ""


def test_keys():
    assert tag_key("re_zero") == tag_key("ReZero") == tag_key("re-zero")
    assert name_key("yui_hirasawa") == name_key("hirasawa_yui") == "hirasawa yui"
    assert singular_key("cats") == "cat" and singular_key("puppies") == "puppy" and singular_key("glass") is None


def test_order_tags():
    tags = [Tag("rem", "character"), Tag("cat", "subject"), Tag("anime", "medium"), Tag("re_zero", "franchise")]
    assert [t.name for t in order_tags(tags)] == ["anime", "re_zero", "rem", "cat"]


def test_store_spelling_and_names(tmp_path):
    s = TagStore(tmp_path / "m.json")
    s.seed([("re_zero", "franchise"), ("hirasawa_yui", "character"), ("cat", "subject")])
    assert s.canonical("rezero") == "re_zero" and s.canonical("Re-Zero".lower()) == "re_zero"
    assert s.canonical("yui_hirasawa") == "hirasawa_yui"
    assert s.canonical("cats") == "cat"                    # plural folded only into a known singular
    assert s.canonical("news") == "news" and s.canonical("series") == "series"
    assert s.canonical("cats", "franchise") == "cats"       # never for franchises/characters


def test_store_learn_commit_and_v1_migration(tmp_path):
    p = tmp_path / "m.json"
    p.write_text('{"re_zero": 5, "cat": 2}')                 # v1 format: plain counts
    s = TagStore(p)
    assert s.count("re_zero") == 5 and s.canonical("rezero") == "re_zero"
    s.learn([Tag("rem", "character"), Tag("re_zero", "franchise")])
    assert s.franchise_of("rem") == "re_zero" and s.kind_of("rem") == "character"
    assert s.count("rem") == 0                               # learning never counts as evidence
    s.commit([["rem", "re_zero"]])
    s2 = TagStore(p)
    assert s2.count("rem") == 1 and s2.count("re_zero") == 6 and s2.kind_of("re_zero") == "franchise"
    assert s2.franchise_of("rem") == "re_zero"


def test_store_rename_delete(tmp_path):
    s = TagStore(tmp_path / "m.json")
    s.commit([["rezero", "rem"], ["rezero"]])
    s.set_kind("rezero", "franchise")
    s.rename("rezero", "re_zero")
    assert s.count("re_zero") == 2 and s.count("rezero") == 0 and s.kind_of("re_zero") == "franchise"
    assert s.canonical("rezero") == "re_zero"                # future model output follows the rename
    s.rename("re_zero", "rz")
    assert s.canonical("rezero") == "rz" and s.canonical("re_zero") == "rz"   # chains collapse
    s.delete("rem")
    assert "rem" not in {t for t, _, _ in TagStore(tmp_path / "m.json").all_tags()}
