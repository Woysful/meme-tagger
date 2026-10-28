"""Custom WD-format taggers: resolving repos/folders, preprocessing, and the tagger with a fake ONNX session."""
import json
import numpy as np
import pytest
from PIL import Image
from memetagger import models
from memetagger.models import ModelError, resolve_wd
from memetagger.wd14 import Preprocess

TAGS = "tag_id,name,category,count\n0,1girl,0,0\n1,guitar,0,0\n2,hirasawa_yui,4,0\n3,k-on!,3,0\n4,rating:g,9,0\n"


def model_dir(tmp_path, onnx=("model.onnx",), tags=TAGS, pre=None):
    d = tmp_path / "my-tagger"; d.mkdir(parents=True)
    for n in onnx:
        (d / n).write_bytes(b"onnx")
    if tags is not None:
        (d / "selected_tags.csv").write_text(tags, "utf-8")
    if pre is not None:
        (d / "preprocess.json").write_text(json.dumps(pre), "utf-8")
    return d


def test_resolve_local_folder_and_file(tmp_path):
    d = model_dir(tmp_path, onnx=("model.onnx", "model_int8.onnx"), pre={"stages": []})
    f = resolve_wd(str(d))
    assert f.model == d / "model.onnx" and f.tags == d / "selected_tags.csv" and f.preprocess == d / "preprocess.json"
    assert resolve_wd(str(d / "model_int8.onnx")).model == d / "model_int8.onnx"


def test_resolve_errors(tmp_path):
    with pytest.raises(ModelError, match="several ONNX"):
        resolve_wd(str(model_dir(tmp_path, onnx=("a.onnx", "b.onnx"))))
    with pytest.raises(ModelError, match="selected_tags.csv"):
        resolve_wd(str(model_dir(tmp_path / "x", tags=None)))
    with pytest.raises(ModelError, match="neither"):
        resolve_wd("not a repo at all")


def test_resolve_hf_repo_downloads_into_its_own_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "user_data_dir", lambda: tmp_path)
    monkeypatch.setattr(models, "_repo_files", lambda repo: ["model.onnx", "model_fp16.onnx", "model_fp16.onnx.data",
                                                               "selected_tags.csv", "preprocess.json", "README.md"])
    got = []

    def fake_download(repo, fn, dest, progress=None):
        got.append(fn); dest.parent.mkdir(parents=True, exist_ok=True); dest.write_text("x")
        return dest
    monkeypatch.setattr(models, "_download", fake_download)
    f = resolve_wd("someone/tagger:model_fp16.onnx")
    assert sorted(got) == ["model_fp16.onnx", "model_fp16.onnx.data", "preprocess.json", "selected_tags.csv"]
    assert f.model == tmp_path / "models" / "custom" / "someone__tagger" / "model_fp16.onnx"
    got.clear(); resolve_wd("someone/tagger:model_fp16.onnx")
    assert got == []                                  # cached


def test_preprocess_wd_and_nchw():
    img = Image.new("RGB", (20, 10), (255, 0, 0))
    wd = Preprocess.for_model([1, 448, 448, 3])
    x = wd(img)
    assert x.shape == (1, 448, 448, 3) and x[0, 224, 224].tolist() == [0, 0, 255]       # BGR, 0..255
    assert x[0, 5, 224].tolist() == [255, 255, 255]                                      # padded with white
    nchw = Preprocess.for_model([1, 3, 64, 64])
    y = nchw(img)
    assert y.shape == (1, 3, 64, 64) and np.allclose(y[0, :, 32, 32], [1, -1, -1])       # RGB, mean/std 0.5


def test_preprocess_json_overrides(tmp_path):
    p = tmp_path / "preprocess.json"
    p.write_text(json.dumps({"stages": [{"type": "rescale_pad", "size": 32, "pad_value": 0},
                                        {"type": "normalize", "mean": [0.5] * 3, "std": [0.5] * 3}]}))
    pre = Preprocess.for_model([1, 3, "h", "w"], p)
    y = pre(Image.new("RGB", (20, 10), (255, 255, 255)))
    assert y.shape == (1, 3, 32, 32) and np.allclose(y[0, :, 2, 16], -1)                  # black padding


class FakeSession:
    def __init__(self, path, providers=None):
        self.path = path

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def get_inputs(self):
        return [type("I", (), {"name": "input", "shape": [1, 3, 32, 32]})()]

    def get_outputs(self):
        return [type("O", (), {"shape": [1, 5]})()]

    def run(self, _out, feed):
        assert feed["input"].shape == (1, 3, 32, 32)
        return [np.array([[3.0, 2.0, 4.0, -3.0, 0.0]], np.float32)]          # logits


def test_custom_tagger_with_fake_session(tmp_path, monkeypatch):
    ort = pytest.importorskip("onnxruntime")
    monkeypatch.setattr(ort, "InferenceSession", FakeSession)
    from memetagger.wd14 import Wd14
    wd = Wd14(str(model_dir(tmp_path)), device="cpu")
    r = wd(Image.new("RGB", (40, 40)), char_thr=0.5, general_thr=0.5)
    assert [c.tag for c in r.characters] == ["hirasawa_yui"] and r.characters[0].franchise == "k-on"
    assert r.characters[0].score == pytest.approx(1 / (1 + np.exp(-4)))           # logits become probabilities
    assert [g for g, _ in r.general] == ["guitar"]                                  # 1girl is blacklisted


def test_custom_tagger_validation(tmp_path, monkeypatch):
    ort = pytest.importorskip("onnxruntime")
    monkeypatch.setattr(ort, "InferenceSession", FakeSession)
    from memetagger.wd14 import Wd14
    with pytest.raises(ModelError, match="5 outputs"):
        Wd14(str(model_dir(tmp_path / "a", tags=TAGS + "5,extra,0,0\n")), device="cpu")
    with pytest.raises(ModelError, match="no character tags"):
        Wd14(str(model_dir(tmp_path / "b", tags=TAGS.replace(",4,", ",0,"))), device="cpu")
    with pytest.raises(ModelError, match="columns"):
        Wd14(str(model_dir(tmp_path / "c", tags="tag,kind\na,b\n")), device="cpu")
