import pytest
from PIL import Image

from memetagger import metadata
from memetagger import organizer as org
from memetagger.naming import LEGACY_STYLES, PRESETS

KIND = {"anime": "medium", "k-on": "franchise", "bocchi": "franchise", "touhou": "franchise",
        "hirasawa_yui": "character", "akiyama_mio": "character", "cat": "subject", "frog": "subject",
        "bunny": "subject", "wojak": "meme_format"}
kind_of = KIND.get


def mk(d, name, tags=None, root=None, image=False, **attrs):
    """A file d/name (+ '.jpg' when no suffix) as an Item with the given tags (as if just analyzed)."""
    p = d / name if "." in name else d / f"{name}.jpg"
    p.parent.mkdir(parents=True, exist_ok=True)
    if image:
        Image.new("RGB", (4, 4), (200, 100, 50)).save(p)
    else:
        p.write_bytes(b"x")
    it = org.make_item(p, root or d)
    if tags is not None:
        it.tags = list(tags)
    for k, v in attrs.items():
        setattr(it, k, v)
    return it


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def rel(plan, root):
    return sorted(m.dst.relative_to(root).as_posix() for m in plan.moves)


# ---------- levels ----------
def test_parse_levels():
    assert org.parse_levels("franchise > character") == [("franchise",), ("character",)]
    assert org.parse_levels("Franchise › Character") == [("franchise",), ("character",)]
    assert org.parse_levels("franchise|meme_format > character") == [("franchise", "meme_format"), ("character",)]
    assert org.parse_levels("medium>franchise> >character") == [("medium",), ("franchise",), ("character",)]
    assert org.parse_levels("") == [] and org.parse_levels(None) == [] and org.parse_levels("  >  ") == []
    with pytest.raises(ValueError, match="nonsense"):
        org.parse_levels("franchise > nonsense")
    for _, spec in org.LEVEL_PRESETS:
        org.parse_levels(spec)
    assert org.DEFAULT_LEVELS == "franchise > character"


def test_assign_folders():
    lv = org.parse_levels("franchise > character")
    yui, mio = [("k-on", "franchise"), ("hirasawa_yui", "character")], [("k-on", "franchise"), ("akiyama_mio", "character")]
    entries = [yui] * 3 + [mio] + [[("bocchi", "franchise")]] * 2 + [[("cat", "subject")]] + [[]]
    out = org.assign_folders(entries, lv)
    assert out == [("k-on", "hirasawa_yui")] * 3 + [("k-on",)] + [()] * 4
    # below the threshold everything stays at the parent
    assert org.assign_folders(entries, lv, min_count=5) == [()] * 8
    # an existing folder is reused even for a single image
    assert org.assign_folders([[("frog", "franchise")]], lv, exists=lambda p: p == ("frog",)) == [("frog",)]
    assert org.assign_folders([[("frog", "franchise")]], lv) == [()]
    # the first tag of the level's kinds decides
    first = [[("cat", "subject"), ("touhou", "franchise"), ("k-on", "franchise")]] * 3
    assert org.assign_folders(first, lv) == [("touhou",)] * 3
    assert org.assign_folders(entries, []) == [()] * 8


# ---------- planning ----------
def _library(src):
    items = []
    def add(n, tags):
        for i in range(n):
            items.append(mk(src, f"{'_'.join(tags) or 'x'}{i}", tags))
    add(20, ["anime", "k-on", "hirasawa_yui"])
    add(2, ["anime", "k-on", "akiyama_mio"])
    add(4, ["anime", "bocchi"])
    add(6, ["cat", "bunny"])
    add(2, ["frog"])
    items.append(mk(src, "plain"))
    return items


def test_make_plan_out_dir_copy_and_idempotent(tmp_path):
    lib, src = tmp_path / "lib", tmp_path / "in"
    items = _library(src)
    plan = org.make_plan(items, out_dir=lib, copy=True, kind_of=kind_of)
    folders = {m.dst.relative_to(lib).parent.as_posix() for m in plan.moves}
    assert folders == {"k-on", "k-on/hirasawa_yui", "bocchi", "."}, folders
    assert all(m.copy for m in plan.moves) and plan.roots == [lib]
    assert (lib / "k-on" / "hirasawa_yui" / "anime_k-on_hirasawa_yui0 [anime k-on hirasawa_yui].jpg") in {m.dst for m in plan.moves}
    assert (lib / "plain.jpg") in {m.dst for m in plan.moves}
    n, err = org.execute(plan, tmp_path / "j")
    assert n == len(items) and not err
    assert len(list(src.iterdir())) == len(items)                     # copy mode: originals stay
    assert len(list((lib / "k-on").rglob("*.jpg"))) == 22

    again = org.collect([lib])
    assert sorted(i.tags for i in again) == sorted(i.tags for i in items)
    for kw in ({}, {"out_dir": lib}, {"out_dir": lib, "library": again}):
        plan2 = org.make_plan(again, kind_of=kind_of, **kw)
        assert plan2.operations == [], [(m.src, m.dst) for m in plan2.operations]
    assert org.make_plan([], kind_of=kind_of).moves == []


def test_make_plan_in_place_moves_and_prunes(tmp_path):
    root = tmp_path / "memes"
    items = [mk(root / "old", f"a{i}.png", ["bunny"], root=root) for i in range(3)]
    plan = org.make_plan(items, levels="subject", kind_of=kind_of)
    assert rel(plan, root) == [f"bunny/a{i} [bunny].png" for i in range(3)]
    assert all(not m.copy for m in plan.moves)
    org.execute(plan, tmp_path / "j")
    assert len(list((root / "bunny").iterdir())) == 3 and not (root / "old").exists()
    n, err = org.undo_last(tmp_path / "j")
    assert n == 3 and not err
    assert sorted(p.name for p in (root / "old").iterdir()) == ["a0.png", "a1.png", "a2.png"]
    assert not (root / "bunny").exists() and root.exists()
    assert org.undo_last(tmp_path / "j") == (0, ["Nothing to undo"])


def test_item_kinds_override_kind_of(tmp_path):
    items = [mk(tmp_path, f"a{i}", ["cat", "wojak"], kinds={"wojak": "franchise"}) for i in range(3)]
    plan = org.make_plan(items)                                          # no kind_of at all
    assert {m.dst.parent for m in plan.moves} == {tmp_path / "wojak"}


def test_settled_files_untouched_and_changed_tags_replaced(tmp_path):
    lib = tmp_path / "lib"
    for i in range(3):
        touch(lib / "k-on" / f"a{i} [k-on].jpg")
    touch(lib / "misc" / "odd [k-on].jpg")                              # settled even though "misplaced"
    items = org.collect([lib])
    assert all(not i.changed for i in items)
    assert org.make_plan(items, kind_of=kind_of).operations == []

    a0 = next(i for i in items if i.src.name == "a0 [k-on].jpg")
    a0.tags = ["bocchi"]
    assert a0.changed
    plan = org.make_plan(items, kind_of=kind_of)
    assert [(m.src.name, m.dst.relative_to(lib).as_posix()) for m in plan.operations] == [("a0 [k-on].jpg", "a0 [bocchi].jpg")]


def test_gather_settled_files_into_new_folder(tmp_path):
    lib = tmp_path / "lib"
    touch(lib / "first [k-on].jpg"); touch(lib / "second [anime k-on].jpg"); touch(lib / "cat [cat].jpg")
    library = org.collect([lib])
    new = mk(tmp_path / "in", "third", ["k-on"])
    plan = org.make_plan([new], out_dir=lib, library=library, kind_of=kind_of)
    assert rel(plan, lib) == ["k-on/first [k-on].jpg", "k-on/second [anime k-on].jpg", "k-on/third [k-on].jpg"]
    org.execute(plan, tmp_path / "j")
    assert sorted(p.name for p in (lib / "k-on").iterdir()) == ["first [k-on].jpg", "second [anime k-on].jpg", "third [k-on].jpg"]
    assert (lib / "cat [cat].jpg").exists()

    # the same with the whole library as the batch (in place)
    lib2 = tmp_path / "lib2"
    touch(lib2 / "first [k-on].jpg"); touch(lib2 / "second [k-on].jpg")
    items = org.collect([lib2]) + [mk(lib2, "third", ["k-on"])]
    assert rel(org.make_plan(items, kind_of=kind_of), lib2) == ["k-on/first [k-on].jpg", "k-on/second [k-on].jpg",
                                                               "k-on/third [k-on].jpg"]
    # without gather the settled files stay where they are
    assert rel(org.make_plan(items, kind_of=kind_of, gather=False), lib2) == ["first [k-on].jpg", "k-on/third [k-on].jpg",
                                                                             "second [k-on].jpg"]


def test_library_items_not_moved_without_rebuild_or_gather(tmp_path):
    lib = tmp_path / "lib"
    touch(lib / "old" / "a [k-on].jpg"); touch(lib / "b [k-on].jpg")
    library = org.collect([lib])
    new = mk(tmp_path / "in", "c", ["k-on"])
    plan = org.make_plan([new], out_dir=lib, library=library, kind_of=kind_of, gather=False)
    assert [(m.src, m.dst.relative_to(lib).as_posix()) for m in plan.moves] == [(new.src, "k-on/c [k-on].jpg")]
    # the library counts for the threshold: without it the new file stays at the root
    assert rel(org.make_plan([new], out_dir=lib, kind_of=kind_of), lib) == ["c [k-on].jpg"]
    # untagged library files are ignored
    touch(lib / "untagged.jpg")
    plan = org.make_plan([new], out_dir=lib, library=org.collect([lib]), kind_of=kind_of, gather=False)
    assert len(plan.moves) == 1


def test_rebuild_replans_everything(tmp_path):
    lib = tmp_path / "lib"
    for i in range(3):
        touch(lib / "wrong" / f"a{i} [k-on].jpg")
    touch(lib / "k-on" / "lonely [frog].jpg")
    items = org.collect([lib])
    assert org.make_plan(items, kind_of=kind_of).operations == []
    plan = org.make_plan(items, kind_of=kind_of, rebuild=True)
    assert rel(plan, lib) == ["k-on/a0 [k-on].jpg", "k-on/a1 [k-on].jpg", "k-on/a2 [k-on].jpg", "lonely [frog].jpg"]
    org.execute(plan, tmp_path / "j")
    assert not (lib / "wrong").exists()
    # library items are also re-planned on rebuild, with a template change
    library = org.collect([lib])
    plan = org.make_plan([], out_dir=lib, library=library, kind_of=kind_of, rebuild=True, template="[{tags}]")
    assert rel(plan, lib) == ["[frog].jpg", "k-on/[k-on] (2).jpg", "k-on/[k-on] (3).jpg", "k-on/[k-on].jpg"]


@pytest.mark.parametrize("tpl", ["{text|description|original} [{tags}]", "{meaningful}< - {text}> [{tags}]",
                                 "{description} [{tags}]", "{meaningful|text|original} [{tags}]"])
def test_template_roundtrip(tmp_path, tpl):
    it = mk(tmp_path, "bunny karate", ["bunny"], text="Kick!", description="A rabbit does karate")
    org.execute(org.make_plan([it], template=tpl, min_count=50), tmp_path / "j")
    again = org.collect([tmp_path])
    assert len(again) == 1 and again[0].tags == ["bunny"] and not again[0].changed      # tags read back from the name
    again[0].text, again[0].description = "Kick!", "A rabbit does karate"               # analyzed again
    plan = org.make_plan(again, template=tpl, min_count=50, rebuild=True)
    assert all(m.src == m.dst for m in plan.moves), [(m.src.name, m.dst.name) for m in plan.moves]


def test_name_styles_with_text(tmp_path):
    rnd = mk(tmp_path, "00IoBi56ZE8", ["bunny"], text="I'm stupid")
    good = mk(tmp_path, "bunny karate", ["bunny"], text="Kick! [2x]")
    names = lambda tpl, **kw: sorted(m.dst.name for m in org.make_plan([rnd, good], template=tpl, min_count=50, **kw).moves)
    assert names(LEGACY_STYLES["keep"]) == ["00IoBi56ZE8 [bunny].jpg", "bunny karate [bunny].jpg"]
    assert names(PRESETS[1][1]) == ["I'm stupid [bunny].jpg", "Kick! 2x [bunny].jpg"]
    assert names(LEGACY_STYLES["text"]) == ["I'm stupid [bunny].jpg", "bunny karate [bunny].jpg"]
    assert names(LEGACY_STYLES["both"]) == ["I'm stupid [bunny].jpg", "bunny karate - Kick! 2x [bunny].jpg"]
    assert names(LEGACY_STYLES["tags"]) == ["[bunny] (2).jpg", "[bunny].jpg"]
    assert names(PRESETS[1][1], text_max=5) == ["I'm [bunny].jpg", "Kick! [bunny].jpg"]
    nothing = mk(tmp_path, "plain", ["cat"])                              # no text -> the original name is kept
    for tpl in (PRESETS[1][1], LEGACY_STYLES["both"]):
        assert [m.dst.name for m in org.make_plan([nothing], template=tpl).moves] == ["plain [cat].jpg"]


def test_name_collisions(tmp_path):
    items = [mk(tmp_path, f"orig{i}", ["cat"]) for i in range(3)] + [mk(tmp_path, "untagged")]
    plan = org.make_plan(items, template="[{tags}]", min_count=50)
    assert sorted(m.dst.name for m in plan.moves) == ["[cat] (2).jpg", "[cat] (3).jpg", "[cat].jpg", "untagged.jpg"]
    org.execute(plan, tmp_path / "j")
    again = org.collect([tmp_path])
    assert sorted(i.tags for i in again) == [[], ["cat"], ["cat"], ["cat"]]
    assert org.make_plan(again, template="[{tags}]", min_count=50).operations == []
    # an existing file on disk is not overwritten
    new = mk(tmp_path / "in", "orig0", ["cat"])
    touch(tmp_path / "out" / "orig0 [cat].jpg")
    assert [m.dst.name for m in org.make_plan([new], out_dir=tmp_path / "out").moves] == ["orig0 [cat] (2).jpg"]


def test_plan_tag_edit(tmp_path):
    lib = tmp_path / "lib"
    touch(lib / "k-on" / "a [anime k-on yui].jpg")
    touch(lib / "k-on" / "b [k-on kon].jpg")
    touch(lib / "c [anime cat].jpg")
    touch(lib / "d [frog].jpg")
    items = org.collect([lib])
    plan = org.plan_tag_edit(items, {"yui": "hirasawa_yui", "kon": "k-on", "anime": None})
    assert sorted((m.src.relative_to(lib).as_posix(), m.dst.relative_to(lib).as_posix()) for m in plan.moves) == [
        ("c [anime cat].jpg", "c [cat].jpg"),
        ("k-on/a [anime k-on yui].jpg", "k-on/a [k-on hirasawa_yui].jpg"),
        ("k-on/b [k-on kon].jpg", "k-on/b [k-on].jpg")]
    assert all(m.meta is None for m in plan.moves) and plan.roots == [lib]
    org.execute(plan, tmp_path / "j")
    assert sorted(p.relative_to(lib).as_posix() for p in lib.rglob("*.jpg")) == [
        "c [cat].jpg", "d [frog].jpg", "k-on/a [k-on hirasawa_yui].jpg", "k-on/b [k-on].jpg"]
    org.undo_last(tmp_path / "j")
    assert (lib / "k-on" / "a [anime k-on yui].jpg").exists() and (lib / "c [anime cat].jpg").exists()


def test_plan_tag_edit_writes_metadata(tmp_path):
    it = mk(tmp_path, "a.png", ["kon", "cat"], image=True, kinds={"kon": "franchise"}, text="hi", description="desc")
    org.execute(org.make_plan([it], min_count=50), tmp_path / "j")
    items = org.collect([tmp_path])
    items[0].text, items[0].description, items[0].kinds = "hi", "desc", {"kon": "franchise"}
    plan = org.plan_tag_edit(items, {"kon": "k-on"}, kind_of=kind_of, metadata_mode="embed")
    org.execute(plan, tmp_path / "j")
    dst = tmp_path / "a [k-on cat].png"
    m = metadata.read(dst)
    assert m.tags == ["k-on", "cat"] and m.kinds == {"k-on": "franchise", "cat": "subject"}
    assert m.title == "hi" and m.description == "desc"


def test_execute_and_undo_copy(tmp_path):
    src, lib = tmp_path / "in", tmp_path / "lib"
    items = [mk(src, f"x{i}", ["k-on"]) for i in range(3)]
    org.execute(org.make_plan(items, out_dir=lib, copy=True, kind_of=kind_of), tmp_path / "j")
    assert len(list((lib / "k-on").iterdir())) == 3 and len(list(src.iterdir())) == 3
    n, err = org.undo_last(tmp_path / "j")
    assert n == 3 and not err
    assert not (lib / "k-on").exists() and lib.exists() and len(list(src.iterdir())) == 3
    assert list((tmp_path / "j").glob("*.undone"))


def test_execute_reports_errors_and_progress(tmp_path):
    items = [mk(tmp_path, f"x{i}", ["cat"]) for i in range(2)]
    plan = org.make_plan(items, min_count=50)
    items[0].src.unlink()
    calls = []
    n, err = org.execute(plan, tmp_path / "j", progress=lambda a, b: calls.append((a, b)))
    assert n == 1 and len(err) == 1 and err[0].startswith("x0.jpg") and calls == [(0, 2)]


@pytest.mark.parametrize("mode", ["embed", "sidecar"])
def test_metadata_written_for_touched_items(tmp_path, mode):
    root = tmp_path / "lib"
    touched = [mk(root, f"t{i}.png", ["anime", "k-on"], image=True, touched=True, text="Text!",
                  description="Two girls", kinds={"anime": "medium"}) for i in range(3)]
    untouched = mk(root, "u.png", ["k-on"], image=True)
    plan = org.make_plan(touched + [untouched], kind_of=kind_of, metadata_mode=mode)
    assert [m.meta is not None for m in plan.moves] == [True, True, True, False]
    n, err = org.execute(plan, tmp_path / "j")
    assert n == 4 and not err
    dst = root / "k-on" / "t0 [anime k-on].png"
    m = metadata.read(dst)
    assert m.tags == ["anime", "k-on"] and m.kinds == {"anime": "medium", "k-on": "franchise"}
    assert m.title == "Text!" and m.description == "Two girls"
    assert metadata.sidecar_path(dst).exists() == (mode == "sidecar")
    assert metadata.read(root / "k-on" / "u [k-on].png") is None
    # metadata only, no move: still an operation
    it = org.make_item(dst, root)
    it.touched, it.description = True, "changed"
    plan = org.make_plan([it], kind_of=kind_of, metadata_mode=mode)
    assert len(plan.operations) == 1 and plan.moves[0].dst == dst
    org.execute(plan, tmp_path / "j")
    assert metadata.read(dst).description == "changed"
    # off: nothing is written
    assert all(m.meta is None for m in org.make_plan(touched, kind_of=kind_of).moves)


def test_sidecar_follows_moves(tmp_path):
    root = tmp_path / "lib"
    it = mk(root, "s.png", ["cat"], image=True, touched=True)
    org.execute(org.make_plan([it], min_count=50, metadata_mode="sidecar", kind_of=kind_of), tmp_path / "j")
    first = root / "s [cat].png"
    assert metadata.sidecar_path(first).exists()
    items = [org.make_item(first, root)] + [mk(root, f"o{i}", ["cat"]) for i in range(2)]
    org.execute(org.make_plan(items, levels="subject", kind_of=kind_of), tmp_path / "j")
    moved = root / "cat" / "s [cat].png"
    assert moved.exists() and metadata.sidecar_path(moved).exists() and not metadata.sidecar_path(first).exists()
    org.undo_last(tmp_path / "j")
    assert metadata.sidecar_path(first).exists()


def test_make_item_reads_metadata(tmp_path):
    p = tmp_path / "x [k-on cat].png"
    Image.new("RGB", (4, 4)).save(p)
    metadata.write(p, metadata.Meta(["k-on", "cat", "gone"], {"k-on": "franchise", "cat": "bogus", "gone": "subject"},
                                    "a description", "the text"), "embed")
    it = org.make_item(p, tmp_path, read_meta=True)
    assert it.tags == ["k-on", "cat"] and it.text == "the text" and it.description == "a description"
    assert it.kinds == {"k-on": "franchise"}                     # only tags in the name, only known kinds
    assert not it.touched and not it.changed
    plain = org.make_item(p, tmp_path)
    assert plain.text == "" and plain.kinds == {}
    assert org.collect([tmp_path], read_meta=True)[0].text == "the text"
    assert org.make_item(tmp_path / "missing [a].png", tmp_path, read_meta=True).text == ""


def test_collect_and_context(tmp_path):
    touch(tmp_path / "a.jpg"); touch(tmp_path / "sub" / "b.PNG"); touch(tmp_path / ".hidden" / "c.jpg")
    touch(tmp_path / "notes.txt")
    items = org.collect([tmp_path, tmp_path / "a.jpg", tmp_path / "notes.txt"])
    assert sorted(i.src.relative_to(tmp_path).as_posix() for i in items) == ["a.jpg", "sub/b.PNG"]
    assert all(i.root == tmp_path for i in items)
    single = org.collect([tmp_path / "sub" / "b.PNG"])
    assert single[0].root == tmp_path / "sub"
    ctx = org.item_context(org.make_item(tmp_path / "Destiny" / "a.jpg", tmp_path))
    assert ctx == {"name": "a", "folders": [tmp_path.name, "Destiny"]}


# ---------- the user's own folders ----------
def test_in_place_keeps_user_folders(tmp_path):
    root = tmp_path / "memes"
    a = mk(root / "K-On", "a", ["anime", "k-on", "hirasawa_yui"], root=root)          # user's folder matches the tag
    b = mk(root / "Funny stuff", "b", ["cat"], root=root)                             # plain user folder
    c = mk(root / "bocchi", "c", ["cat"], root=root)                                  # tag folder it no longer belongs to
    plan = org.make_plan([a, b, c], kind_of=kind_of)
    assert rel(plan, root) == ["Funny stuff/b [cat].jpg", "K-On/a [anime k-on hirasawa_yui].jpg", "c [cat].jpg"]
    assert rel(org.make_plan([a, b, c], kind_of=kind_of, rebuild=True), root) == \
        ["a [anime k-on hirasawa_yui].jpg", "b [cat].jpg", "c [cat].jpg"]


def test_guess_kind_for_tags_from_file_names():
    from memetagger.characters import default_db
    from memetagger.engine import guess_kind
    from memetagger.tagstore import TagStore
    db, store = default_db(), TagStore()
    assert guess_kind(store, db, "hirasawa_yui") == "character"
    assert guess_kind(store, db, "megumin") == "character"                            # popular one-word name
    assert guess_kind(store, db, "k-on") == "franchise" and guess_kind(store, db, "kirby") == "franchise"
    assert guess_kind(store, db, "anime") == "medium"
    for word in ("cat", "ghost", "frog", "bunny"):                                    # Danbooru has cat_(kemono_friends)
        assert guess_kind(store, db, word) is None, word
    store.set_kind("cat", "subject")
    assert guess_kind(store, db, "cat") == "subject"
