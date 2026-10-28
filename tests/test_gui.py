"""Main window with real Qt (offscreen) and a fake analysis engine: queue, statuses, editing, plan, apply, undo."""
import os
import time
from pathlib import Path
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PIL import Image
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication, QMessageBox
from memetagger.engine import Prepared, Result
from memetagger.imaging import ImageData
from memetagger.tags import Tag
from memetagger.tagstore import TagStore
from memetagger.ui import common
from memetagger.ui.model import C_TAGS
from memetagger.vlm import OllamaVLM, VLMError

KON = [Tag("anime", "medium"), Tag("k-on", "franchise"), Tag("hirasawa_yui", "character")]


class FakeEngine:
    def __init__(self, answers):
        self.answers, self.progress, self.prepared = answers, None, []

    def warm_up(self, want):
        pass

    def flush(self):
        pass

    def prepare(self, path, opts):
        self.prepared.append(Path(path).name)
        return Prepared(ImageData(path))

    def analyze(self, prep, ctx, opts):
        a = self.answers.get(prep.image.path.stem)
        if isinstance(a, Exception):
            raise a
        tags = a if a is not None else [Tag("cat", "subject")]
        return Result(list(tags), f"about {prep.image.path.stem}", "", source="vlm", want=opts.want,
                      suggestions=[("akiyama_mio", "k-on", 0.4)])


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    common.apply_theme(app, "light")
    return app


@pytest.fixture(autouse=True)
def no_dialogs(monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    for name in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: None))

    def offline(self, *a, **k):
        raise VLMError("offline")
    monkeypatch.setattr(OllamaVLM, "_http", offline)


def images(folder: Path, names):
    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        Image.new("RGB", (64, 48), (200, 80, 80)).save(folder / n)
    return folder


def wait(app, cond, timeout=10):
    t = time.time()
    while not cond():
        app.processEvents(); time.sleep(0.01)
        if time.time() - t > timeout:
            raise AssertionError("timeout")
    app.processEvents()


def window(qapp, tmp_path, answers=None, **settings):
    from memetagger.ui.main import Main
    s = common.Settings(QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    for k, v in settings.items():
        s.set(k, v)
    eng = FakeEngine(answers or {})
    w = Main(s, TagStore(tmp_path / "store.json"), engine_factory=lambda cfg, store, progress: eng)
    w.fake = eng
    return w


def idle(w):
    return w.worker is None and not w.queue


def test_analysis_statuses_and_plan(qapp, tmp_path):
    src = images(tmp_path / "in", ["a.png", "b.png", "c.png", "bad.png", "d [old].png"])
    w = window(qapp, tmp_path, {"a": KON, "b": KON, "c": KON, "bad": RuntimeError("model failed")}, skip=True)
    w.add_paths([str(src)])
    wait(qapp, lambda: idle(w))
    st = {r.item.src.name: r.status for r in w.model.rows}
    assert st == {"a.png": "done", "b.png": "done", "c.png": "done", "bad.png": "error", "d [old].png": "skipped"}
    assert "d [old].png" not in w.fake.prepared                      # skipped files are never analyzed
    rows = {r.item.src.name: r for r in w.model.rows}
    assert rows["a.png"].item.kinds["hirasawa_yui"] == "character" and rows["a.png"].suggestions
    assert all(r.thumb is not None for r in w.model.rows)
    w.replan()
    assert rows["a.png"].dest == "k-on/hirasawa_yui/a [anime k-on hirasawa_yui].png"
    assert rows["bad.png"].dest == "stays as is" and not rows["bad.png"].dest_changed
    assert rows["d [old].png"].dest == "stays as is"
    w.close()


def test_threshold_setting_and_edits(qapp, tmp_path):
    src = images(tmp_path / "in", ["a.png", "b.png"])
    w = window(qapp, tmp_path, {"a": KON, "b": KON})
    w.add_paths([str(src)]); wait(qapp, lambda: idle(w))
    w.replan()
    assert all(r.dest == f"{r.item.base} [anime k-on hirasawa_yui].png" for r in w.model.rows)      # 2 < 3: no folder
    w.panel.min_count.setValue(2); w.replan()
    assert all(r.dest.startswith("k-on/hirasawa_yui/") for r in w.model.rows)
    assert w.s.get("min_count") == 2
    # edit tags in the table (clean_tag + dedupe), the row becomes "edited"
    idx = w.proxy.index(0, C_TAGS)
    assert w.proxy.setData(idx, "Funny Cat, cat k-on", Qt.ItemDataRole.EditRole)
    row = w.proxy.data(idx, Qt.ItemDataRole.UserRole + 2)
    assert row.item.tags == ["funny", "cat", "k-on"] and row.status == "edited"
    # preview: add to the current row and to a multi-selection
    w.table.selectRow(0); qapp.processEvents()
    w.preview.add_edit.setText("guitar"); w.preview._add_typed()
    assert w.current_row().item.tags[-1] == "guitar"
    w.table.selectAll(); qapp.processEvents()
    w.add_tags_to_selection(["meme_format_x"])
    assert all("meme_format_x" in r.item.tags for r in w.model.rows)
    w.close()


def test_filters(qapp, tmp_path):
    src = images(tmp_path / "in", ["a.png", "b.png", "bad.png"])
    w = window(qapp, tmp_path, {"a": KON, "b": [], "bad": RuntimeError("x")})
    w.add_paths([str(src)]); wait(qapp, lambda: idle(w))
    w.proxy.set(mode="untagged"); assert w.proxy.rowCount() == 2
    w.proxy.set(mode="error"); assert w.proxy.rowCount() == 1
    w.proxy.set(mode="all", query="hirasawa"); assert w.proxy.rowCount() == 1
    w.close()


def test_apply_updates_rows_commits_tags_and_undo(qapp, tmp_path):
    src = images(tmp_path / "in", ["a.png", "b.png", "c.png"])
    lib = tmp_path / "lib"; lib.mkdir()
    w = window(qapp, tmp_path, {"a": KON, "b": KON, "c": KON}, out=str(lib), metadata="sidecar")
    w.add_paths([str(src)]); wait(qapp, lambda: idle(w))
    w.apply(); wait(qapp, lambda: not w.tasks.tasks)
    moved = sorted(p.relative_to(lib).as_posix() for p in lib.rglob("*.png"))
    assert moved == [f"k-on/hirasawa_yui/{n} [anime k-on hirasawa_yui].png" for n in "abc"]
    assert all(r.status == "applied" and r.item.src.exists() and not r.item.changed for r in w.model.rows)
    assert w.store.count("hirasawa_yui") == 3                        # counted only when applied
    assert (lib / "k-on" / "hirasawa_yui" / "a [anime k-on hirasawa_yui].png.xmp").exists()
    w.replan()
    assert all(r.dest == "stays as is" for r in w.model.rows)        # applied files are settled
    w.undo()
    assert not w.model.rows and sorted(p.name for p in src.glob("*.png")) == ["a.png", "b.png", "c.png"]
    assert (src / "a.png.xmp").exists() and not list(lib.rglob("*.png"))      # sidecars travel back with their files
    w.close()


def test_tag_library_rename_in_files(qapp, tmp_path):
    lib = images(tmp_path / "lib", ["x [yui_hirasawa].png", "y [yui_hirasawa cat].png"])
    w = window(qapp, tmp_path, out=str(lib))
    assert w.apply_tag_mapping({"yui_hirasawa": "hirasawa_yui"}, in_files=True)
    assert sorted(p.name for p in lib.iterdir()) == ["x [hirasawa_yui].png", "y [hirasawa_yui cat].png"]
    w.store.rename("yui_hirasawa", "hirasawa_yui")
    assert w.store.canonical("yui_hirasawa") == "hirasawa_yui"
    assert w.apply_tag_mapping({"cat": None}, in_files=True, confirm=False)
    assert sorted(p.name for p in lib.iterdir()) == ["x [hirasawa_yui].png", "y [hirasawa_yui].png"]
    w.close()


def test_stop_and_pause(qapp, tmp_path):
    src = images(tmp_path / "in", [f"i{n}.png" for n in range(30)])
    w = window(qapp, tmp_path)
    w.add_paths([str(src)])
    wait(qapp, lambda: w.worker is not None)
    w.toggle_pause(); assert w.worker.is_paused()
    w.toggle_pause(); assert not w.worker.is_paused()
    w.stop_analysis(); wait(qapp, lambda: idle(w))
    statuses = {r.status for r in w.model.rows}
    assert statuses <= {"done", "idle"} and "analyzing" not in statuses and "queued" not in statuses
    w.close()


def test_settings_persist_and_widgets(qapp, tmp_path):
    w = window(qapp, tmp_path)
    w.panel.levels.setText("medium > franchise")
    w.panel.strictness.setCurrentIndex(w.panel.strictness.findData("strict"))
    w.panel.engine.setCurrentIndex(w.panel.engine.findData("classic"))
    assert w.panel.classic_group.isVisibleTo(w.panel)
    w.panel.levels.setText("franchise > nonsense")
    assert w.panel.level_spec() == "franchise > character" and "Unknown tag kind" in w.panel.levels_hint.text()
    w.panel.levels.setText("medium > franchise")
    w.close()
    w2 = window(qapp, tmp_path)
    assert w2.s.get("levels") == "medium > franchise" and w2.panel.levels.text() == "medium > franchise"
    assert w2.s.get("strictness") == "strict" and w2.panel.strictness.currentData() == "strict"
    opts = w2.tag_options()
    assert opts.strictness == "strict" and opts.max_tags == 8
    w2.close()
