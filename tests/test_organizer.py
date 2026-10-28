import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from memetagger import organizer as org


def test_hierarchy_and_undo():
    tmp = Path(tempfile.mkdtemp()); lib = tmp / "lib"; src = tmp / "in"; src.mkdir()
    items = []
    def add(n, tags):
        for i in range(n):
            p = src / f"{'_'.join(tags) or 'x'}{i}.jpg"; p.write_bytes(b"x"); items.append(org._item(p, src))
            items[-1].tags = tags
    add(20, ["anime", "k-on", "hirasawa_yui"])
    add(5, ["anime", "k-on", "akiyama_mio"])
    add(15, ["anime", "bocchi"])
    add(16, ["cat", "bunny"])
    add(3, ["frog"])
    (src / "plain.jpg").write_bytes(b"x"); items.append(org._item(src / "plain.jpg", src))

    plan = org.make_plan(items, out_dir=lib, group_min=12, max_depth=4, copy=True)
    rel = {m.dst.relative_to(lib).parent.as_posix() for m in plan.moves}
    assert rel == {"anime/k-on", "anime/k-on/hirasawa_yui", "anime/bocchi", "cat", "."}, rel
    assert any("hirasawa_yui" in m.dst.name for m in plan.moves)
    n, err = org.execute(plan, tmp / "j"); assert n == len(items) and not err
    assert (src / "plain.jpg").exists()                      # copy-режим: оригиналы целы
    assert len(list((lib / "anime" / "k-on").rglob("*.jpg"))) == 25

    # повторный запуск идемпотентен, библиотека пересобирается из имён файлов
    items2 = org.collect([lib])
    plan2 = org.make_plan([], out_dir=None) ; assert plan2.moves == []
    plan3 = org.make_plan(items2, out_dir=None, group_min=12)
    assert all(m.src == m.dst for m in plan3.moves)

    # растущая библиотека: добавляем ещё 12 'frog' -> появляется папка frog
    add(10, ["frog"])
    new = [i for i in items if "frog" in i.tags]
    plan4 = org.make_plan(new, out_dir=lib, group_min=12, copy=True, merge_library=True)
    assert (lib / "frog") in {m.dst.parent for m in plan4.moves}
    org.execute(plan4, tmp / "j")

    # откат
    n, err = org.undo_last(tmp / "j"); assert not err
    assert not (lib / "frog").exists()


def test_move_mode_and_prune():
    tmp = Path(tempfile.mkdtemp()); d = tmp / "memes" / "old"; d.mkdir(parents=True)
    items = []
    for i in range(13):
        p = d / f"a{i}.png"; p.write_bytes(b"x"); it = org._item(p, tmp / "memes"); it.tags = ["bunny"]; items.append(it)
    plan = org.make_plan(items, group_min=12)
    org.execute(plan, tmp / "j")
    assert len(list((tmp / "memes" / "bunny").iterdir())) == 13 and not d.exists()
    org.undo_last(tmp / "j"); assert len(list(d.iterdir())) == 13 and not (tmp / "memes" / "bunny").exists()


def test_names():
    assert org.split_name("lol [a b]") == ("lol", ["a", "b"])
    assert org.build_filename("lol", ["a", "b"], ".jpg") == "lol [a b].jpg"
    assert org.clean_tag("K-On!") == "k-on"
    assert len(org.build_filename("я" * 300, ["x" * 50] * 10, ".jpg").encode()) <= 255


def test_replace_name_and_counter():
    assert org.split_name("[anime k-on] (2)") == ("", ["anime", "k-on"])
    assert org.split_name("lol [a] (3)") == ("lol", ["a"])
    assert org.build_filename("", ["anime", "k-on"], ".jpg") == "[anime k-on].jpg"
    tmp = Path(tempfile.mkdtemp()); items = []
    for i in range(3):
        p = tmp / f"orig{i}.jpg"; p.write_bytes(b"x"); it = org._item(p, tmp); it.tags = ["cat"]; items.append(it)
    p = tmp / "untagged.jpg"; p.write_bytes(b"x"); items.append(org._item(p, tmp))
    plan = org.make_plan(items, replace_name=True, group_min=50)
    names = sorted(m.dst.name for m in plan.moves)
    assert names == ["[cat] (2).jpg", "[cat] (3).jpg", "[cat].jpg", "untagged.jpg"], names
    org.execute(plan, tmp / "j")
    again = org.make_plan(org.collect([tmp]), replace_name=True, group_min=50)   # идемпотентно
    assert all(m.src == m.dst for m in again.moves), [(m.src.name, m.dst.name) for m in again.moves]


def test_vocab():
    from memetagger.vocab import parse_vocab, version_of
    txt = Path(__file__).resolve().parent.parent.joinpath("memetagger/vocab_default.txt").read_text()
    e = {x.label: x for x in parse_vocab(txt)}
    assert version_of(txt) >= 2
    assert e["cat"].group == e["doge"].group == "animals" and e["doge"].min_p == 0.5
    assert e["~person"].hidden and e["~person"].group is None
    assert e["wojak"].group is None and e["anime"].group == "medium"


def test_random_names_from_real_library():
    random_names = ["00IoBi56ZE8", "10314_zdLfhQbruNi", "1473214070", "16TPxM1M_eY", "20230928_020536", "52fa937ce9bd3fbd",
        "-5370782485853618053_120", "Image", "images", "original", "tenor", "photo_2020-12-27_20-40-56", "IMG_20200905_184603",
        "F7_c1kIWsAAEucE", "GD-iI78awAABUS4", "E8HqCTIVUAYwdnK",
        "ac14dc731bc22a95b6e7a70c4cd67c93856de4a369c6b7a3b2cf2377076b19f5_thumb",
        "tumblr_fd2ce0e1c57fa61ba9e4d4385c17ac0c_e66bd36e_500", "ezgif-2-7492ba43ba",
        "318743935_827379245221766_5248117334255235550_n", "4956_z2n2ANls3Ue", "Screenshot_2026-05-20_160201", "unnamed",
        "maxresdefault", "PXL_20240204_004627998", "b4e68285-ccba-4acf-a326-9da083bae455 — копия", "Ha8P8sJT_400x400",
        "video_2025-12-04_17-48-33", "Zr5VES-0QBg", "photo_2021-02-25_03-46-57 (2)", "5631_zxqRI1q4VUA", "9gwumjr5x5831"]
    meaningful = ["bunny karate", "Peak", "persona 4 its peak", "yui i do care i did ask", "Aigis Inteligent", "horse handshake",
        "кирпич в ебало", "anime girl yo wtf", "NTR Slayer", "pacman -Syu", "2kb of joy", "K-ON!! - 23 - Large 12",
        "Reimu fumo considers all opinions carefully", "gem alarm marathon", "автору 10 лет", "Yui Naz Clean", "ehe",
        "Marisa smoke", "ur not funny sorry", "Усаги бизнессмен", "cat sad", "Do u hate me", "I DON'T REMEMBER ASKING"]
    bad = [n for n in random_names if not org.looks_random(n)]
    assert not bad, f"должны быть случайными: {bad}"
    bad = [n for n in meaningful if org.looks_random(n)]
    assert not bad, f"должны быть осмысленными: {bad}"


def test_keep_meaningful():
    tmp = Path(tempfile.mkdtemp()); items = []
    for n in ("bunny karate", "00IoBi56ZE8"):
        p = tmp / f"{n}.jpg"; p.write_bytes(b"x"); it = org._item(p, tmp); it.tags = ["bunny"]; items.append(it)
    both = sorted(m.dst.name for m in org.make_plan(items, replace_name=True, group_min=50).moves)
    assert both == ["[bunny] (2).jpg", "[bunny].jpg"], both
    keep = sorted(m.dst.name for m in org.make_plan(items, replace_name=True, keep_meaningful=True, group_min=50).moves)
    assert keep == ["[bunny].jpg", "bunny karate [bunny].jpg"], keep
    ctx = org.item_context(org._item(tmp / "Destiny" / "a.jpg", tmp)); assert ctx["folders"] == [tmp.name, "Destiny"]


def test_clean_name_text():
    c = org.clean_name_text
    assert c("Look at this [photo]: www.imgflip.com @someuser  What?!") == "Look at this photo What!"
    assert c("  ...ПРИВЕТ, мир / как дела?  ") == "ПРИВЕТ, мир как дела"
    assert c('he said "no" <yes> | maybe') == "he said no yes maybe"
    assert c("") == "" and c(None) == "" and c("https://example.com/x?y=1") == ""
    long = "one two three four five six seven eight nine ten eleven twelve"
    out = c(long, 30); assert len(out) <= 30 and out == "one two three four five six" and not out.endswith(" ")
    assert c("x" * 100, 20) == "x" * 20                                    # без пробелов режем по длине


def _mk(tmp, name, tags, text=""):
    p = tmp / f"{name}.jpg"; p.write_bytes(b"x"); it = org._item(p, tmp); it.tags, it.text = tags, text
    return it


def test_name_styles_with_text():
    tmp = Path(tempfile.mkdtemp())
    rnd, good = _mk(tmp, "00IoBi56ZE8", ["bunny"], "I'm stupid"), _mk(tmp, "bunny karate", ["bunny"], "Kick! [2x]")
    names = lambda **kw: sorted(m.dst.name for m in org.make_plan([rnd, good], group_min=50, **kw).moves)
    assert names(name_style="keep") == ["00IoBi56ZE8 [bunny].jpg", "bunny karate [bunny].jpg"]
    assert names(name_style="text") == ["I'm stupid [bunny].jpg", "Kick! 2x [bunny].jpg"]
    assert names(name_style="text", keep_meaningful=True) == ["I'm stupid [bunny].jpg", "bunny karate [bunny].jpg"]
    assert names(name_style="both") == ["I'm stupid [bunny].jpg", "bunny karate - Kick! 2x [bunny].jpg"]
    assert names(name_style="tags") == ["[bunny] (2).jpg", "[bunny].jpg"]
    assert names(name_style="text", text_max=15) == ["I'm stupid [bunny].jpg", "Kick! 2x [bunny].jpg"]
    nothing = _mk(tmp, "plain", ["cat"], "")                                   # текста нет -> исходное имя сохраняется
    assert [m.dst.name for m in org.make_plan([nothing], group_min=50, name_style="text").moves] == ["plain [cat].jpg"]
    assert [m.dst.name for m in org.make_plan([nothing], group_min=50, name_style="both").moves] == ["plain [cat].jpg"]


def test_text_in_name_is_idempotent():
    tmp = Path(tempfile.mkdtemp()); it = _mk(tmp, "bunny karate", ["bunny"], "Kick")
    for style in ("text", "both"):
        d = Path(tempfile.mkdtemp()); a = _mk(d, "bunny karate", ["bunny"], "Kick")
        org.execute(org.make_plan([a], group_min=50, name_style=style), d / "j")
        again = org.collect([d]); again[0].text = "Kick"                       # повторный анализ нашёл тот же текст
        plan = org.make_plan(again, group_min=50, name_style=style)
        assert all(m.src == m.dst for m in plan.moves), (style, [(m.src.name, m.dst.name) for m in plan.moves])
    assert org.split_name("Kick - 2x [bunny] (2)") == ("Kick - 2x", ["bunny"])
    cyr = _mk(tmp, "x1", ["cat"], "Я " + "очень " * 20 + "дурак")
    name = org.make_plan([cyr], group_min=50, name_style="text", text_max=60).moves[0].dst.name
    assert len(name.encode()) <= 255 and name.endswith("[cat].jpg")


if __name__ == "__main__":
    test_replace_name_and_counter(); test_vocab(); test_random_names_from_real_library(); test_keep_meaningful(); test_clean_name_text(); test_name_styles_with_text(); test_text_in_name_is_idempotent()
    test_names(); test_hierarchy_and_undo(); test_move_mode_and_prune(); print("OK")
