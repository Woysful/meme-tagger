import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from memetagger import organizer as org
from memetagger.template import DEFAULT_TEMPLATE, PRESETS, LEGACY_STYLES, render, render_filename, validate

V = dict(original="bunny karate", text="Kick! 2x", description="A white rabbit in a karate pose", tags=["bunny", "rabbit", "anime"])
RND = dict(V, original="00IoBi56ZE8")


def test_presets_and_variables():
    r = lambda t, v=V: render(t, v)
    assert r("{original} [{tags}]") == "bunny karate [bunny rabbit anime]"
    assert r("{text|original} [{tags}]", RND) == "Kick! 2x [bunny rabbit anime]"
    assert r("{meaningful|text|original} [{tags}]") == "bunny karate [bunny rabbit anime]"
    assert r("{meaningful|text|original} [{tags}]", RND) == "Kick! 2x [bunny rabbit anime]"
    assert r("{meaningful}< - {text}> [{tags}]") == "bunny karate - Kick! 2x [bunny rabbit anime]"
    assert r("{meaningful}< - {text}> [{tags}]", RND) == "Kick! 2x [bunny rabbit anime]"           # ведущий разделитель убран
    assert r("{description} [{tags}]") == "A white rabbit in a karate pose [bunny rabbit anime]"
    assert r("[{tags}]") == "[bunny rabbit anime]"
    for name, t in PRESETS:
        assert not validate(t)[0], name
    assert set(LEGACY_STYLES) == {"keep", "text", "both", "tags"}


def test_fallbacks_when_no_text_use_description():
    notext = dict(RND, text="")
    assert render("{text|description|original} [{tags}]", notext) == "A white rabbit in a karate pose [bunny rabbit anime]"
    assert render("{text|description|original} [{tags}]", dict(notext, description="")) == "00IoBi56ZE8 [bunny rabbit anime]"
    assert render("{original}< - {text}> [{tags}]", notext) == "00IoBi56ZE8 [bunny rabbit anime]"      # пустая группа исчезла
    assert render("{text} [{tags}]", dict(notext, tags=[])) == "00IoBi56ZE8"                         # пусто -> исходное имя
    assert render("[{tags}]", dict(notext, tags=[])) == "00IoBi56ZE8"


def test_limits_cleanup_and_safety():
    assert render("{text:5} [{tags:2}]", V) == "Kick! [bunny rabbit]"
    assert render("{description:10}", V) == "A white"
    assert render("{text}/{description}", V).count("/") == 0                                          # слэши из шаблона убраны
    assert render("(){original} []", V) == "bunny karate"                                           # пустые скобки удалены
    assert render("{original} {text}", dict(V, original="Kick! 2x", text="kick 2x")) == "Kick! 2x"    # не дублируем текст из имени
    # длинное: хвост [теги] сохраняется, байты в пределах
    long = dict(V, text="Очень длинный текст " * 10, description="Подробное описание сцены " * 10, original="x" * 200)
    name = render_filename("{original} - {text} - {description} [{tags}]", type("I", (), dict(base=long["original"], text=long["text"], description=long["description"], tags=long["tags"]))(), ".jpg")
    assert len(name.encode()) <= 245 and name.endswith("[bunny rabbit anime].jpg"), (len(name.encode()), name)


def test_validation():
    assert validate("{original} [{tags}]") == ([], [])
    assert "Unknown variable: {foo}" in validate("{foo} [{tags}]")[0][0]
    assert validate("{text:x}")[0] and validate("{text")[0] and validate("<a<{text}>>")[0] and validate("   ")[0]
    assert validate("{text} {{tags}}")[0]
    assert "No {tags}" in validate("{text}")[1][0]
    assert "End the template with" in validate("{tags} {text}")[1][0]
    assert "only N tags" in validate("{text} [{tags:3}]")[1][0]
    assert render("{foo}", V) == render(DEFAULT_TEMPLATE, V)                                          # ошибка -> шаблон по умолчанию


def test_make_plan_with_template_roundtrip():
    for tpl in ("{text|description|original} [{tags}]", "{meaningful}< - {text}> [{tags}]", "{description} [{tags}]"):
        d = Path(tempfile.mkdtemp()); p = d / "bunny karate.jpg"; p.write_bytes(b"x")
        it = org._item(p, d); it.tags, it.text, it.description = ["bunny"], "Kick!", "A rabbit does karate"
        org.execute(org.make_plan([it], group_min=50, template=tpl), d / "j")
        again = org.collect([d]); assert again[0].tags == ["bunny"], (tpl, again[0].tags)          # теги читаются из имени
        again[0].text, again[0].description = "Kick!", "A rabbit does karate"                      # повторный анализ
        plan = org.make_plan(again, group_min=50, template=tpl)
        assert all(m.src == m.dst for m in plan.moves), (tpl, [(m.src.name, m.dst.name) for m in plan.moves])


if __name__ == "__main__":
    test_presets_and_variables(); test_fallbacks_when_no_text_use_description(); test_limits_cleanup_and_safety()
    test_validation(); test_make_plan_with_template_roundtrip(); print("OK template")
