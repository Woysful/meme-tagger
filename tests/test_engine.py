import base64
import io
import json
import pytest
from PIL import Image
from fake_ollama import FakeOllama, GOOD
from memetagger.characters import CharacterDB
from memetagger.engine import Engine, EngineConfig, TagOptions
from memetagger.tags import Tag
from memetagger.tagstore import TagStore
from memetagger.vlm import SCHEMA, SYSTEM, OllamaVLM, VLMError, build_schema, build_system, unload_models, words_for
from memetagger.wd14 import WdCandidate, WdResult

DB = CharacterDB([("rem_(re:zero)", "re:zero_kara_hajimeru_isekai_seikatsu", 8000), ("hirasawa_yui", "k-on!", 8000)])


@pytest.fixture
def ollama():
    f = FakeOllama(); yield f; f.close()


@pytest.fixture
def img(tmp_path):
    p = tmp_path / "a.png"; Image.new("RGBA", (3000, 1000), (255, 0, 0, 128)).save(p); return p


class StubOcr:
    def __init__(self, text=""): self.text, self.calls = text, 0
    def read(self, image): self.calls += 1; return self.text
    def flush(self): pass


class StubWd:
    def __init__(self, chars=()): self.chars, self.calls = list(chars), 0
    def __call__(self, img, *a, **k): self.calls += 1; return WdResult(self.chars)


class StubClassic:
    def tag(self, img, wd, *a):
        return [Tag("cat", "subject", 0.9, "classic"), Tag("Cats", "subject", 0.8, "classic"), Tag("photo", "medium", 0.7)], False


def engine(ollama=None, mode="vlm", ocr="", wd=(), **kw):
    cfg = EngineConfig(mode=mode, host=ollama.host if ollama else "127.0.0.1:1", ocr=True, wd=True)
    args = dict(ocr_reader=StubOcr(ocr), wd=wd if isinstance(wd, StubWd) else StubWd(wd), classic=StubClassic(), chardb=DB, folder_hints={}, seeds=[], parents={})
    args.update(kw)
    return Engine(cfg, TagStore(), **args)


def test_payload_prompt_and_tags(ollama, img):
    e = engine(ollama, wd=[WdCandidate("rem", "re_zero", 0.62)], ocr="Rem best girl")
    r = e.tag(img, {"name": "bunny karate", "folders": []})
    body = ollama.chat[0]
    assert body["think"] is False and body["stream"] is False and body["options"]["temperature"] == 0
    assert body["format"] == SCHEMA and body["messages"][0]["content"] == SYSTEM
    user = body["messages"][1]
    assert "rem (re_zero) 0.62" in user["content"] and "Rem best girl" in user["content"] and "'bunny karate'" in user["content"]
    assert "already uses" not in user["content"]                          # no vocabulary anchoring by default
    jpg = Image.open(io.BytesIO(base64.b64decode(user["images"][0]))); assert jpg.format == "JPEG" and max(jpg.size) <= 1344
    assert r.source == "vlm" and r.names == ["anime", "re_zero", "rem", "girl", "sadness"], r.names
    assert r.caption == "Я глупый кролик" and r.description == "Rem looks sad." and not ollama.verify


def test_vocabulary_is_optional(ollama, img):
    e = engine(ollama); e.store.commit([["re_zero"]]); e.store.set_kind("re_zero", "franchise")
    e.tag(img, opts=TagOptions(show_vocab=True))
    assert "already uses" in ollama.chat[0]["messages"][1]["content"] and "re_zero" in ollama.chat[0]["messages"][1]["content"]


def test_doubtful_claims_are_verified_in_one_query(ollama, img):
    ollama.answer = dict(GOOD, characters=[{"name": "Rem", "franchise": "Re:Zero", "confidence": "low"},
                                           {"name": "Yui Hirasawa", "franchise": "K-On!", "confidence": "low"}],
                         franchise=[{"name": "Re:Zero", "confidence": "low"}])
    ollama.verdicts = {"'rem'": "yes", "hirasawa": "no"}
    r = engine(ollama).tag(img)
    assert len(ollama.verify) == 1 and "c3" in ollama.verify[0], ollama.verify      # rem, yui, re_zero in one call
    assert "rem" in r.names and "re_zero" in r.names and "hirasawa_yui" not in r.names and "k-on" not in r.names
    assert any("hirasawa_yui" in n and "check said no" in n for n in r.notes)
    ollama.verify.clear()
    engine(ollama).tag(img, opts=TagOptions(verify=False))
    assert not ollama.verify


def test_retry_without_think_and_empty_answer_falls_back(ollama, img):
    ollama.mode = "think400"
    assert engine(ollama).tag(img).source == "vlm" and "think" not in ollama.chat[-1]
    ollama.mode = "empty"
    r = engine(ollama, mode="auto").tag(img)
    assert r.source == "classic" and any("vision model failed" in n for n in r.notes)
    assert r.names == ["photo", "cat"]                                          # cat/Cats collapse
    with pytest.raises(VLMError):
        engine(ollama, mode="vlm").tag(img)


def test_unreachable_server(img):
    e = engine(None, mode="auto")
    assert e.vlm is None and e.tag(img).source == "classic"
    with pytest.raises(VLMError, match="ollama pull"):
        engine(None, mode="vlm")


def test_generation_switches_skip_stages(ollama, img):
    wd, ocr = StubWd(), StubOcr("hello")
    r = engine(ollama, ocr_reader=ocr, wd=wd).tag(img, opts=TagOptions(want=frozenset({"text"})))
    body = ollama.chat[-1]
    assert body["format"]["required"] == ["name_text"] and r.names == [] and r.caption == "Я глупый кролик"
    assert wd.calls == 0 and ocr.calls == 1 and "Candidates" not in body["messages"][1]["content"]
    ocr2 = StubOcr("hello")
    r = engine(ollama, ocr_reader=ocr2).tag(img, opts=TagOptions(want=frozenset({"description"})))
    assert ocr2.calls == 0 and r.description and r.caption == "" and r.tags == []


def test_wd14_alone_in_classic_mode(img):
    e = engine(None, mode="classic", wd=[WdCandidate("hirasawa_yui", "k-on", 0.97)])
    class Art(StubClassic):
        def tag(self, img, wd, *a): return [Tag("anime", "medium", 0.8, "classic")], True
    e._classic = Art()
    assert e.tag(img).names == ["anime", "k-on", "hirasawa_yui"]


def test_folder_hints_with_kinds(ollama, img):
    ollama.answer = dict(GOOD, characters=[], franchise=[], subjects=["girl"], themes=[])
    e = engine(ollama, folder_hints={"gensh": ["genshin_impact"], "sad": ["sad"]},
               chardb=CharacterDB([("paimon_(genshin_impact)", "genshin_impact", 5000)]))
    r = e.tag(img, {"name": "x", "folders": ["Memes", "Gensh", "sad"]})
    assert r.names == ["anime", "genshin_impact", "girl"]
    assert "genshin_impact, sad" in ollama.chat[-1]["messages"][1]["content"]


def test_schema_and_limits(ollama, img):
    assert list(build_schema({"text"})["properties"]) == ["name_text"]
    assert list(build_schema()["properties"])[:3] == ["description", "name_text", "medium"]
    assert build_schema()["properties"]["characters"]["items"]["required"] == ["name", "franchise", "confidence"]
    only_text = build_system({"text"})
    assert "franchise" not in only_text and "Confidence" not in only_text and len(only_text) < len(SYSTEM) * 0.6
    lim = {"desc_chars": 100, "text_chars": 60}
    assert build_schema(limits=lim)["properties"]["description"]["maxLength"] == 150 and words_for(100) == 17
    assert "about 17 words" in build_system(limits=lim) and "at most 30 words" in SYSTEM
    ollama.reject_maxlen = True
    v = OllamaVLM(ollama.host)
    v.analyze("aGk=", "x", limits=lim)
    assert len(ollama.chat) == 2 and v.use_maxlen is False and "maxLength" not in json.dumps(ollama.chat[1]["format"])


def test_status_unload_pull(ollama):
    assert OllamaVLM(ollama.host, "qwen3-vl:8b").status()[0]
    ok, msg = OllamaVLM(ollama.host, "other:1b").status(); assert not ok and "ollama pull other:1b" in msg
    assert "cannot reach" in OllamaVLM("127.0.0.1:1", "x").status()[1]
    ollama.ps = [{"name": "qwen3-vl:8b"}, {"name": "gemma3:12b"}, {"name": "llava:latest"}]
    assert unload_models(ollama.host, ["qwen3-vl:8b", "llava"]) == ["qwen3-vl:8b", "llava:latest"]
    ollama.pull = [{"status": "pulling manifest"}, {"status": "pulling abc", "total": 1000, "completed": 250}, {"status": "success"}]
    seen = []
    assert OllamaVLM(ollama.host).pull("qwen3-vl:8b", lambda *a: seen.append(a)) and seen[1] == ("pulling abc", 250, 1000)
    ollama.pull = [{"status": "pulling manifest"}, {"error": "file does not exist"}]
    with pytest.raises(VLMError, match="does not exist"):
        OllamaVLM(ollama.host).pull("nope")
    assert OllamaVLM(ollama.host).list_models()[0] == {"name": "qwen3-vl:8b", "size": 0, "vision": True}


def test_ocr_reader_cache(tmp_path, img):
    from memetagger.imaging import ImageData
    from memetagger.ocr import OcrReader

    class Res:
        txts, scores = ("ПРИВЕТ мир", "мусор", ""), (0.95, 0.2, 0.9)

    class Eng:
        calls = 0
        def __call__(self, arr): Eng.calls += 1; return Res()
    cache = tmp_path / "c.json"
    r = OcrReader(cache_path=cache, engine=Eng())
    assert r.read(ImageData(img)) == "ПРИВЕТ мир" and r.read(ImageData(img)) == "ПРИВЕТ мир" and Eng.calls == 1
    r.flush()
    assert OcrReader(cache_path=cache, engine=Eng()).read(ImageData(img)) == "ПРИВЕТ мир" and Eng.calls == 1


def test_debug_report(ollama, tmp_path):
    from memetagger import debug
    d = tmp_path / "imgs"; d.mkdir(); Image.new("RGB", (64, 64)).save(d / "x.png")
    ollama.models = ["qwen3-vl:8b", "gemma3:12b"]
    cfg = EngineConfig(mode="vlm", host=ollama.host, ocr=False, wd=False)
    md = debug.to_markdown(debug.run([str(d)], ["qwen3-vl:8b", "gemma3:12b"], cfg, progress=lambda m: None), ["qwen3-vl:8b", "gemma3:12b"])
    assert "### x.png" in md and md.count("**anime re_zero rem") == 2 and "`gemma3:12b`" in md
