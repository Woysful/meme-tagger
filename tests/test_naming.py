from pathlib import Path
from types import SimpleNamespace

from memetagger.naming import (DEFAULT_TEMPLATE, LEGACY_STYLES, PRESETS, VARS, build_filename, clean_name_text,
                               fit_bytes, looks_random, render, render_filename, split_name, validate)
from memetagger.tags import clean_tag

V = dict(original="bunny karate", text="Kick! 2x", description="A white rabbit in a karate pose",
         tags=["bunny", "rabbit", "anime"])
RND = dict(V, original="00IoBi56ZE8")


def item(base, tags, text="", description=""):
    return SimpleNamespace(base=base, tags=list(tags), text=text, description=description)


# ---------- file names ----------
def test_split_and_build():
    assert split_name("lol [a b]") == ("lol", ["a", "b"])
    assert split_name("no tags here") == ("no tags here", [])
    assert split_name("[anime k-on] (2)") == ("", ["anime", "k-on"])
    assert split_name("lol [a] (3)") == ("lol", ["a"])
    assert split_name("Kick - 2x [bunny] (2)") == ("Kick - 2x", ["bunny"])
    assert build_filename("lol", ["a", "b"], ".jpg") == "lol [a b].jpg"
    assert build_filename("", ["anime", "k-on"], ".jpg") == "[anime k-on].jpg"
    assert build_filename("", [], ".jpg") == "image.jpg"
    assert len(build_filename("я" * 300, ["x" * 50] * 10, ".jpg").encode()) <= 255


def test_clean_tag():
    assert clean_tag("K-On!") == "k-on"
    assert clean_tag("Re:Zero") == "re_zero"
    assert clean_tag("Fate/Grand Order") == "fate_grand_order"


def test_random_names_from_real_library():
    random_names = ["00IoBi56ZE8", "10314_zdLfhQbruNi", "1473214070", "16TPxM1M_eY", "20230928_020536", "52fa937ce9bd3fbd",
        "-5370782485853618053_120", "Image", "images", "original", "tenor", "photo_2020-12-27_20-40-56", "IMG_20200905_184603",
        "F7_c1kIWsAAEucE", "GD-iI78awAABUS4", "E8HqCTIVUAYwdnK",
        "ac14dc731bc22a95b6e7a70c4cd67c93856de4a369c6b7a3b2cf2377076b19f5_thumb",
        "tumblr_fd2ce0e1c57fa61ba9e4d4385c17ac0c_e66bd36e_500", "ezgif-2-7492ba43ba",
        "318743935_827379245221766_5248117334255235550_n", "4956_z2n2ANls3Ue", "Screenshot_2026-05-20_160201", "unnamed",
        "maxresdefault", "PXL_20240204_004627998", "b4e68285-ccba-4acf-a326-9da083bae455 — копия", "Ha8P8sJT_400x400",
        "video_2025-12-04_17-48-33", "Zr5VES-0QBg", "photo_2021-02-25_03-46-57 (2)", "5631_zxqRI1q4VUA", "9gwumjr5x5831",
        "", "   "]
    meaningful = ["bunny karate", "Peak", "persona 4 its peak", "yui i do care i did ask", "Aigis Inteligent", "horse handshake",
        "кирпич в ебало", "anime girl yo wtf", "NTR Slayer", "pacman -Syu", "2kb of joy", "K-ON!! - 23 - Large 12",
        "Reimu fumo considers all opinions carefully", "gem alarm marathon", "автору 10 лет", "Yui Naz Clean", "ehe",
        "Marisa smoke", "ur not funny sorry", "Усаги бизнессмен", "cat sad", "Do u hate me", "I DON'T REMEMBER ASKING"]
    assert [n for n in random_names if not looks_random(n)] == []
    assert [n for n in meaningful if looks_random(n)] == []


def test_clean_name_text():
    c = clean_name_text
    assert c("Look at this [photo]: www.imgflip.com @someuser  What?!") == "Look at this photo What!"
    assert c("  ...ПРИВЕТ, мир / как дела?  ") == "ПРИВЕТ, мир как дела"
    assert c('he said "no" <yes> | maybe') == "he said no yes maybe"
    assert c("") == "" and c(None) == "" and c("https://example.com/x?y=1") == ""
    long = "one two three four five six seven eight nine ten eleven twelve"
    out = c(long, 30)
    assert out == "one two three four five six" and len(out) <= 30
    assert c("x" * 100, 20) == "x" * 20                     # no spaces: cut by length


# ---------- templates ----------
def test_presets_and_variables():
    r = lambda t, v=V: render(t, v)
    assert r("{original} [{tags}]") == "bunny karate [bunny rabbit anime]"
    assert r("{text|original} [{tags}]", RND) == "Kick! 2x [bunny rabbit anime]"
    assert r("{meaningful|text|original} [{tags}]") == "bunny karate [bunny rabbit anime]"
    assert r("{meaningful|text|original} [{tags}]", RND) == "Kick! 2x [bunny rabbit anime]"
    assert r("{meaningful}< - {text}> [{tags}]") == "bunny karate - Kick! 2x [bunny rabbit anime]"
    assert r("{meaningful}< - {text}> [{tags}]", RND) == "Kick! 2x [bunny rabbit anime]"   # leading separator dropped
    assert r("{description} [{tags}]") == "A white rabbit in a karate pose [bunny rabbit anime]"
    assert r("[{tags}]") == "[bunny rabbit anime]"
    for name, t in PRESETS:
        assert validate(t) == ([], []), name
    assert DEFAULT_TEMPLATE == PRESETS[0][1]
    assert set(LEGACY_STYLES) == {"keep", "text", "both", "tags"}
    assert set(LEGACY_STYLES.values()) <= {t for _, t in PRESETS}
    assert VARS == ("original", "meaningful", "text", "description", "tags")


def test_fallbacks_when_no_text_use_description():
    notext = dict(RND, text="")
    assert render("{text|description|original} [{tags}]", notext) == "A white rabbit in a karate pose [bunny rabbit anime]"
    assert render("{text|description|original} [{tags}]", dict(notext, description="")) == "00IoBi56ZE8 [bunny rabbit anime]"
    assert render("{original}< - {text}> [{tags}]", notext) == "00IoBi56ZE8 [bunny rabbit anime]"   # empty group removed
    assert render("{text} [{tags}]", dict(notext, tags=[])) == "00IoBi56ZE8"                      # empty -> original name
    assert render("[{tags}]", dict(notext, tags=[])) == "00IoBi56ZE8"
    assert render("[{tags}]", dict(notext, original="", tags=[])) == "image"


def test_limits_cleanup_and_safety():
    assert render("{text:5} [{tags:2}]", V) == "Kick! [bunny rabbit]"
    assert render("{description:10}", V) == "A white"
    assert "/" not in render("{text}/{description}", V)                                          # slashes removed
    assert render("(){original} []", V) == "bunny karate"                                         # empty brackets removed
    assert render("{original} {text}", dict(V, original="Kick! 2x", text="kick 2x")) == "Kick! 2x"  # no duplicated text
    # {text|original} is a choice, not two places: the text is kept even when the name contains it
    assert render("{text|original} [{tags}]", dict(V, original="Kick! 2x", text="Kick! 2x")) == "Kick! 2x [bunny rabbit anime]"


def test_render_filename_byte_limits():
    long = item("x" * 200, V["tags"], "Очень длинный текст " * 10, "Подробное описание сцены " * 10)
    name = render_filename("{original} - {text} - {description} [{tags}]", long, ".jpg")
    assert len(name.encode()) <= 245 and name.endswith("[bunny rabbit anime].jpg"), (len(name.encode()), name)
    cyr = item("x1", ["cat"], "Я " + "очень " * 20 + "дурак")
    name = render_filename("{text|original} [{tags}]", cyr, ".jpg", text_max=60)
    assert len(name.encode()) <= 255 and name.endswith("[cat].jpg")
    many = item("a", [f"tag_number_{i}" for i in range(60)])
    name = render_filename(DEFAULT_TEMPLATE, many, ".jpeg")
    assert len(name.encode()) + len(" (99)".encode()) <= 255 and name.endswith(".jpeg")


def test_fit_bytes_keeps_tags():
    s = "a" * 300 + " [cat dog]"
    out = fit_bytes(s, 100)
    assert len(out.encode()) <= 100 and out.endswith(" [cat dog]")
    assert fit_bytes("short [x]", 100) == "short [x]"
    huge = "a [" + " ".join(["t" * 20] * 20) + "]"
    assert len(fit_bytes(huge, 100).encode()) <= 100


def test_validation():
    assert validate("{original} [{tags}]") == ([], [])
    assert "Unknown variable: {foo}" in validate("{foo} [{tags}]")[0][0]
    for bad in ("{text:x}", "{text:0}", "{text", "<a<{text}>>", "   ", "", "{text} {{tags}}", "<{text}"):
        assert validate(bad)[0], bad
    assert "No {tags}" in validate("{text}")[1][0]
    assert "End the template with" in validate("{tags} {text}")[1][0]
    assert "only N tags" in validate("{text} [{tags:3}]")[1][0]
    assert render("{foo}", V) == render(DEFAULT_TEMPLATE, V)                                     # error -> default template


def test_vocab():
    from memetagger.vocab import parse_vocab, version_of
    txt = (Path(__file__).resolve().parent.parent / "memetagger" / "data" / "vocab_default.txt").read_text()
    e = {x.label: x for x in parse_vocab(txt)}
    assert version_of(txt) >= 2
    assert e["cat"].group == e["doge"].group == "animals" and e["doge"].min_p == 0.5
    assert e["~person"].hidden and e["~person"].group is None
    assert e["wojak"].group is None and e["anime"].group == "medium"
