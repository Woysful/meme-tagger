"""Логика контроллера GUI на заглушке Qt: очередь, пауза/стоп, список моделей Ollama, статус GPU, меню конфигов, фильтр."""
import os, sys, tempfile, threading, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
tmp = Path(tempfile.mkdtemp())
os.environ["XDG_DATA_HOME"] = str(tmp / "data"); os.environ["XDG_CONFIG_HOME"] = str(tmp / "config")
import fake_qt
fake_qt.install()
from PIL import Image
from memetagger import app
from memetagger.engine import Result
from memetagger.vlm import VLMError

DELAY = {"v": 0.0}
CALLS = []
LAST_OPTS = {}


class FakeEngine:
    def __init__(self, **kw): self.kw = kw
    def tag(self, path, context=None, want=None, **opts):
        want = frozenset(want or {"tags", "text", "description"})
        time.sleep(DELAY["v"]); CALLS.append((Path(path).name, context)); LAST_OPTS.update(opts, want=want, engine=self.kw)
        stem = Path(path).stem.split(" [")[0]
        return Result(["anime", stem] if "tags" in want and stem != "bad" else [], f"scene of {stem}" if "description" in want else "",
                      "vlm", "", "ocr text", f"text of {stem}" if "text" in want else "", want)
    def flush(self): pass


class FakeVLM:
    models, loaded_, fail = [], [], False
    def __init__(self, host="", model=""): pass
    def list_models(self):
        if FakeVLM.fail: raise VLMError("cannot reach Ollama")
        return FakeVLM.models
    def loaded(self): return FakeVLM.loaded_
    def status(self): return True, "OK"


app.Engine, app.OllamaVLM = FakeEngine, FakeVLM


def mkimgs(n, prefix="img", d=None):
    d = d or Path(tempfile.mkdtemp())
    for i in range(n):
        Image.new("RGB", (32, 32), (i * 20 % 255, 0, 0)).save(d / f"{prefix}{i}.png")
    return d


def wait(cond, timeout=8):
    t0 = time.time()
    while not cond():
        if time.time() - t0 > timeout: raise AssertionError("timeout")
        time.sleep(0.02)


def new_window(keep_settings=False):
    app._engine = None; fake_qt.QMessageBox.calls.clear(); CALLS.clear(); fake_qt.QMessageBox.answer = fake_qt._Enum("Yes")
    if not keep_settings: app.QSettings.store.clear()
    return app.Main()


def test_analyze_and_progress():
    DELAY["v"] = 0; w = new_window(); d = mkimgs(5)
    assert w.stack.currentIndex() == 0 and not w.apply_btn.isEnabled()
    w.add_paths([str(d)]); wait(lambda: w.worker is None and w.prog_done == 5)
    assert [it.tags[0] for it in w.items] == ["anime"] * 5 and w.table.item(0, 2).text().startswith("anime img0")
    assert w.progress.value() == 5 and w.stack.currentIndex() == 1 and w.apply_btn.isEnabled()
    assert CALLS[0][1]["folders"][0] == d.name                       # контекст (папки/имя) доходит до движка
    assert "OCR: ocr text" in w.table.item(0, 1).toolTip()


def test_pause_resume_stop():
    DELAY["v"] = 0.15; w = new_window(); d = mkimgs(8)
    w.add_paths([str(d)]); wait(lambda: w.prog_done >= 1)
    w.toggle_pause(); assert w.pause_btn.text() == "▶ Resume"
    time.sleep(0.4); frozen = w.prog_done; time.sleep(0.6)
    assert w.prog_done == frozen and frozen < 8, (frozen, w.prog_done)     # во время паузы прогресс стоит
    w.toggle_pause(); assert w.pause_btn.text() == "⏸ Pause"
    wait(lambda: w.prog_done >= frozen + 2)
    w.stop_analysis(); wait(lambda: w.worker is None and w.reanalyze_btn.isEnabled())
    assert w.stopped and not w.queue and w.pause_btn.isEnabled() is False and w.reanalyze_btn.isEnabled()
    untagged = sum(1 for it in w.items if not it.tags); assert 0 < untagged < 8
    DELAY["v"] = 0; w.reanalyze(); wait(lambda: w.worker is None and w.prog_done == 8)
    assert all(it.tags for it in w.items)


def test_add_while_running_and_clear_paused():
    DELAY["v"] = 0.1; w = new_window()
    w.add_paths([str(mkimgs(4, "a"))]); w.add_paths([str(mkimgs(3, "b"))])          # вторая пачка встаёт в очередь
    assert w.prog_total == 7
    wait(lambda: w.worker is None and w.prog_done == 7 and not w.queue)
    assert len(w.items) == 7 and all(it.tags for it in w.items)
    w.add_paths([str(mkimgs(6, "c"))]); w.toggle_pause(); time.sleep(0.3)
    w.clear()                                                                         # clear на паузе не зависает
    assert w.worker is None and not w.items and w.stack.currentIndex() == 0 and w.prog_total == 0


def test_ollama_models_and_gpu():
    w = new_window()
    FakeVLM.fail = False
    FakeVLM.models = [{"name": "qwen3-vl:8b", "size": 6.1e9, "vision": True}, {"name": "llava:7b", "size": 4.7e9, "vision": None},
                      {"name": "gemma3:12b", "size": 8.1e9, "vision": True}, {"name": "nomic-embed-text", "size": 2.7e8, "vision": False}]
    w.refresh_models(); wait(lambda: "online" in w.badge.text())
    assert w.model_box.itemTexts() == ["qwen3-vl:8b", "llava:7b", "gemma3:12b"], w.model_box.itemTexts()   # embed-модель скрыта
    assert "3 vision" in w.badge.text() and "1 text-only hidden" in w.badge.text()
    assert w.model_hint.text().startswith("Installed · 6.1 GB")
    w.model_box.setCurrentText("foo:1b"); assert "ollama pull foo:1b" in w.model_hint.text()
    w.model_box.setCurrentText("nomic-embed-text"); assert "text-only" in w.model_hint.text()
    assert w.badge.property("state") == "ok" and w.model_hint.property("state") == "warn"      # цвета через состояния темы
    w.model_box.setCurrentText("qwen3-vl:8b"); assert w.model_hint.property("state") == ""
    FakeVLM.fail = True; w.refresh_models(); wait(lambda: "offline" in w.badge.text()); assert w.badge.property("state") == "bad"
    FakeVLM.loaded_ = [{"name": "qwen3-vl:8b", "size": 6e9, "vram": 3e9}]
    w.poll_ps(); wait(lambda: "50% in VRAM" in w.gpu_label.text()); assert "partly on CPU" in w.gpu_label.text() and w.gpu_label.property("state") == "warn"
    FakeVLM.loaded_ = [{"name": "qwen3-vl:8b", "size": 6e9, "vram": 6e9}]
    w.poll_ps(); wait(lambda: "100% in VRAM" in w.gpu_label.text()); assert "CPU" not in w.gpu_label.text() and w.gpu_label.property("state") == ""
    FakeVLM.loaded_ = []; w.poll_ps(); wait(lambda: "no model loaded" in w.gpu_label.text())


def test_config_menu_and_reload():
    w = new_window(); fake_qt.QDesktopServices.opened.clear()
    w.open_config("folder_hints.json"); w.open_config("vocab.txt"); w.open_config("aliases.json"); w.open_config("seed_tags.txt")
    cfg = tmp / "config" / "memetagger"
    assert all((cfg / n).exists() for n in ("folder_hints.json", "vocab.txt", "aliases.json", "seed_tags.txt"))   # дефолты созданы
    assert [Path(u).name for u in fake_qt.QDesktopServices.opened] == ["folder_hints.json", "vocab.txt", "aliases.json", "seed_tags.txt"]
    w.open_path(cfg); assert fake_qt.QDesktopServices.opened[-1] == str(cfg)
    app._engine = object(); w.reload_config(); assert app._engine is None


def test_banned_filter_edit():
    DELAY["v"] = 0; w = new_window(); w.add_paths([str(mkimgs(3, "x"))]); wait(lambda: w.worker is None and w.prog_done == 3)
    w.banned_edit.setText("anime, nothing"); w.on_banned_changed()
    assert all("anime" not in it.tags for it in w.items) and w.table.item(0, 2).text() == "x0"
    w.table.item(1, 2).setText("Cat, DOG  k-on!"); assert w.items[1].tags == ["cat", "dog", "k-on"]     # ручная правка
    w.filter_edit.setText("dog"); w.apply_filter(); assert [w.table.isRowHidden(r) for r in range(3)] == [True, False, True]
    w.filter_edit.setText(""); w.untagged_cb.setChecked(True); w.items[2].tags = []; w.apply_filter()
    assert [w.table.isRowHidden(r) for r in range(3)] == [True, True, False]


def test_theme_switch_creates_check_icon():
    w = new_window()
    for mode in ("dark", "light", "system"):
        w.theme_box.setCurrentText({"dark": "Theme: dark", "light": "Theme: light", "system": "Theme: system"}[mode]); w.on_theme()
    icon = tmp / "data" / "memetagger" / "ui" / "check.png"
    assert icon.exists() and Image.open(icon).size == (16, 16)


def test_template_description_and_columns():
    from memetagger.template import LEGACY_STYLES, PRESETS
    DELAY["v"] = 0; w = new_window(); d = mkimgs(2, "pic"); w.add_paths([str(d)]); wait(lambda: w.worker is None and w.prog_done == 2)
    assert w.table.item(0, 3).text() == "text of pic0" and w.items[0].text == "text of pic0"
    assert w.table.item(0, 4).text() == "scene of pic0" and w.items[0].description == "scene of pic0"       # колонка описания
    assert LAST_OPTS["desc_lang"] == "English" and w.tpl_preview.text().startswith("Preview: ")
    assert w.template_edit.text() == "{original} [{tags}]" and w.preset_box.currentIndex() == 0
    w.replan(); assert w.table.item(0, 5).text() == "pic0 [anime pic0].png"
    w.template_edit.setText("{text|original} [{tags}]")                                                       # правка шаблона
    assert w.preset_box.currentIndex() == 1; w.replan(); assert w.table.item(0, 5).text() == "text of pic0 [anime pic0].png"
    w.template_edit.setText("{description} [{tags}]"); w.replan()
    assert w.table.item(0, 5).text() == "scene of pic0 [anime pic0].png"
    w.preset_box.setCurrentIndex(3); w.on_preset(3); assert w.template_edit.text() == PRESETS[3][1]
    w.template_edit.setText("{text} / {tags}")                                                                # свой шаблон
    assert w.preset_box.currentIndex() == len(PRESETS) and w.tpl_warn.property("state") == "warn" and "End the template" in w.tpl_warn.text()
    w.template_edit.setText("{foo} [{tags}]"); w.replan()                                                      # ошибка -> безопасный шаблон
    assert "Unknown variable" in w.tpl_warn.text() and w.table.item(0, 5).text() == "pic0 [anime pic0].png"
    assert w.current_template() == "{original} [{tags}]"
    w.template_edit.setText("{description|text} [{tags}]")
    w.table.item(1, 4).setText("  edited scene "); assert w.items[1].description == "edited scene"            # ручная правка описания
    w.replan(); assert w.table.item(1, 5).text() == "edited scene [anime pic1].png"
    w.filter_edit.setText("edited scene"); w.apply_filter(); assert [w.table.isRowHidden(r) for r in range(2)] == [True, False]
    w.template_edit.setText("{text} [{tags}]"); assert w.tpl_warn.text() == ""                               # без предупреждений
    w.desc_lang.setCurrentText("Русский"); w.reanalyze(); wait(lambda: w.worker is None and w.prog_done == 2)
    assert LAST_OPTS["desc_lang"] == "Russian"                                                                 # язык описания уходит в движок


def test_legacy_name_style_migrates_to_template():
    from memetagger.template import LEGACY_STYLES
    app._engine = None; app.QSettings.store.clear(); app.QSettings.store["name_style"] = "both"
    w = app.Main(); assert w.template_edit.text() == LEGACY_STYLES["both"] and w.preset_box.currentIndex() == 3
    app.QSettings.store.clear(); app.QSettings.store.update({"name_style": "both", "template": "[{tags}]"})
    assert app.Main().template_edit.text() == "[{tags}]"                                                       # новый ключ главнее


def test_generation_checkboxes():
    DELAY["v"] = 0; w = new_window()
    assert w.want() == {"tags", "text", "description"} and w.table.header[2] == w.COLS[2]
    d = Path(tempfile.mkdtemp())
    for n in ("a [anime old].png", "b [cat].png"): Image.new("RGB", (16, 16)).save(d / n)
    w.add_paths([str(d)]); wait(lambda: w.worker is None and w.prog_done == 2)
    assert len(CALLS) == 0 and [it.tags for it in w.items] == [["anime", "old"], ["cat"]]                # теги уже есть -> пропуск
    w.clear(); CALLS.clear()
    w.gen_tags.setChecked(False)                                                                         # теги выключены
    assert w.want() == {"text", "description"} and "generation off" in w.table.header[2] and app.QSettings.store["gen_tags"] is False
    w.add_paths([str(d)]); wait(lambda: w.worker is None and w.prog_done == 2)
    assert len(CALLS) == 2 and LAST_OPTS["want"] == {"text", "description"} and LAST_OPTS["engine"]["want_tags"] is False
    assert [it.tags for it in w.items] == [["anime", "old"], ["cat"]]                                    # прежние теги не затёрты
    assert w.items[0].text == "text of a" and w.table.item(0, 4).text() == "scene of a"
    w.gen_text.setChecked(False); w.gen_desc.setChecked(False)                                           # всё выключили -> теги вернулись
    assert w.gen_tags.isChecked() and "At least one" in w.info.text()
    w.gen_tags.setChecked(True); w.gen_text.setChecked(True); w.gen_desc.setChecked(False)
    w.template_edit.setText("{description} [{tags}]"); w.items[0].description = ""; w.items[1].description = ""
    w.update_template_ui(); assert "description generation is off" in w.tpl_warn.text()
    w.items[0].description = "filled by hand"; w.update_template_ui(); assert w.tpl_warn.text() == ""    # вручную заполнено -> без предупреждения
    w.reanalyze(); wait(lambda: w.worker is None and w.prog_done == 2)
    assert LAST_OPTS["want"] == {"tags", "text"} and w.items[0].description == "filled by hand"          # описание не затёрто


def test_all_settings_persist_between_sessions():
    w = new_window()
    changes = {
        "template_edit": ("setText", "{text|description} [{tags}] x"), "out": ("setText", "/tmp/lib"), "host_edit": ("setText", "127.0.0.1:9999"),
        "banned_edit": ("setText", "foo bar"), "gmin": ("setValue", 25), "depth": ("setValue", 2), "text_max": ("setValue", 45),
        "desc_max": ("setValue", 70), "maxtags": ("setValue", 9), "cthr": ("setValue", 0.4), "gthr": ("setValue", 0.5),
    }
    for name, (meth, val) in changes.items(): getattr(getattr(w, name), meth)(val)
    boxes = ("gen_text", "copy", "skip", "ocr_cb", "verify_cb", "fallback", "unload_exit", "limit_model")
    for name in boxes:                                                                # заглушка Qt отвечает на любой атрибут: проверяем, что это реальный чекбокс
        assert isinstance(getattr(w, name), fake_qt.QCheckBox), name
        getattr(w, name).setChecked(not getattr(w, name).isChecked())
    w.merge.setChecked(False); w.desc_lang.setCurrentText("Русский"); w.engine_box.setCurrentText("Ollama vision model only (best quality)")
    w.model_box.setCurrentText("gemma3:12b"); w.theme_box.setCurrentText("Theme: dark"); w.tabs.setCurrentIndex(1); w.split.setSizes([700, 500])
    assert w.save_timer.started                                                       # изменение запускает отложенное сохранение
    w.closeEvent(fake_qt.Stub())                                                      # …и закрытие сохраняет без запуска анализа
    w2 = new_window(keep_settings=True)                                               # «новая сессия»
    for name, (meth, val) in changes.items():
        got = getattr(w2, name).text() if meth == "setText" else getattr(w2, name).value()
        assert got == val, (name, got, val)
    for name, expect in (("gen_text", False), ("copy", True), ("skip", False), ("ocr_cb", False), ("verify_cb", False), ("fallback", False),
                         ("unload_exit", False), ("limit_model", False), ("merge", False), ("gen_tags", True), ("gen_desc", True)):
        assert getattr(w2, name).isChecked() == expect, name
    assert w2.desc_lang.currentData() == "Russian" and w2.engine_box.currentData() == "vlm" and w2.model_box.currentText() == "gemma3:12b"
    assert w2.theme_box.currentData() == "dark" and w2.tabs.currentIndex() == 1 and w2.split.sizes() == [700, 500]
    assert w2.template_edit.text() == "{text|description} [{tags}] x" and w2.want() == {"tags", "description"}


def test_model_side_limits_option():
    DELAY["v"] = 0; w = new_window(); w.add_paths([str(mkimgs(1, "l"))]); wait(lambda: w.worker is None and w.prog_done == 1)
    assert LAST_OPTS["limits"] == {"desc_chars": 100, "text_chars": 60}              # по умолчанию включено; описание до ~100 символов
    w.desc_max.setValue(40); w.limit_model.setChecked(False); w.reanalyze(); wait(lambda: w.worker is None and w.prog_done == 1)
    assert LAST_OPTS["limits"] is None                                                # режим «только обрезка в имени»
    w.limit_model.setChecked(True); w.reanalyze(); wait(lambda: w.worker is None and w.prog_done == 1)
    assert LAST_OPTS["limits"] == {"desc_chars": 40, "text_chars": 60}


def test_unload_button_and_exit():
    DELAY["v"] = 0; w = new_window(); calls = []
    app.unload_models = lambda host, names, timeout=4: calls.append((host, set(names), timeout)) or sorted(names)
    w.used_models = {"qwen3-vl:8b"}; w.model_box.setCurrentText("gemma3:12b"); w.host_edit.setText("h:1")
    w.unload_model(); wait(lambda: calls and "Unloaded" in w.info.text())
    assert calls[0][:2] == ("h:1", {"qwen3-vl:8b", "gemma3:12b"}) and "gemma3:12b" in w.info.text()
    app.unload_models = lambda host, names, timeout=4: (_ for _ in ()).throw(RuntimeError("ollama down"))
    w.unload_model(); wait(lambda: "Unload failed" in w.info.text())
    # при выходе: только модели, которыми пользовались; ошибки не мешают закрытию
    calls.clear(); app.unload_models = lambda host, names, timeout=4: calls.append((host, set(names), timeout)) or []
    w.unload_exit.setChecked(True); w.used_models = {"x:1"}; w.closeEvent(fake_qt.Stub())
    assert calls == [("h:1", {"x:1"}, 3)]
    calls.clear(); w.unload_exit.setChecked(False); w.closeEvent(fake_qt.Stub()); assert calls == []
    w.unload_exit.setChecked(True); w.used_models = set(); w.closeEvent(fake_qt.Stub()); assert calls == []        # ничего не использовали
    app.unload_models = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")); w.used_models = {"x"}; w.closeEvent(fake_qt.Stub())
    # кнопка недоступна, пока идёт анализ; модель попадает в used_models
    DELAY["v"] = 0.2; w2 = new_window(); w2.add_paths([str(mkimgs(2, "u"))])
    assert not w2.unload_btn.isEnabled() and "qwen3-vl:8b" in w2.used_models
    wait(lambda: w2.worker is None and w2.unload_btn.isEnabled()); DELAY["v"] = 0
    w3 = new_window(); w3.engine_box.setCurrentText("Classic CLIP+WD14 only (fast, weaker)")
    w3.add_paths([str(mkimgs(1, "c"))]); wait(lambda: w3.worker is None); assert w3.used_models == set()
    import memetagger.vlm as vlm_mod; app.unload_models = vlm_mod.unload_models


class FakePuller(fake_qt.QThread):
    progress = fake_qt.Signal(); done = fake_qt.Signal(); failed = fake_qt.Signal()
    script = []
    gate = threading.Event()                    # поток начинает «скачивание» только когда тест откроет ворота
    def __init__(self, host, name): super().__init__(); self.host, self.name = host, name
    def cancel(self): self.requestInterruption()
    def run(self):
        FakePuller.gate.wait(5)
        for kind, *a in FakePuller.script:
            if self.isInterruptionRequested(): return
            if kind == "progress": self.progress.emit(*a); time.sleep(0.05)
            elif kind == "done": self.done.emit(self.name)
            elif kind == "fail": self.failed.emit(a[0]); return


def test_pull_from_ui():
    w = new_window(); real = app.PullWorker; app.PullWorker = FakePuller
    FakeVLM.fail = False; FakeVLM.models = [{"name": "gemma3:12b", "size": 8.1e9, "vision": True}]
    seen = []; orig = w.on_pull_progress; w.on_pull_progress = lambda *a: (seen.append(a), orig(*a))
    try:
        w.model_box.setCurrentText("bad name"); w.toggle_pull()                                   # имя с пробелом -> подсказка, загрузки нет
        assert w.puller is None and fake_qt.QMessageBox.calls[-1][0] == "information"
        w.model_box.setCurrentText("gemma3:12b"); fake_qt.QMessageBox.answer = fake_qt._Enum("No"); w.toggle_pull()
        assert w.puller is None                                                                    # отказ в подтверждении
        fake_qt.QMessageBox.answer = fake_qt._Enum("Yes")
        FakePuller.script = [("progress", "pulling manifest", None, None), ("progress", "pulling abc123", 4_050_000_000, 8_100_000_000),
                             ("progress", "pulling abc123", 8_100_000_000, 8_100_000_000), ("done",)]
        FakePuller.gate.clear(); w.toggle_pull()
        assert w.pull_btn.text() == "✕ Cancel pull" and w.pull_label.text().startswith("Starting")
        FakePuller.gate.set(); wait(lambda: w.puller is None)
        assert seen[1][1:] == (4_050_000_000, 8_100_000_000) and w.pull_bar.value() == 100 and "8.10 / 8.10 GB" in w.pull_label.text()
        assert fake_qt.QMessageBox.calls[-1][0] == "information" and "installed" in fake_qt.QMessageBox.calls[-1][1][1]
        assert w.pull_btn.text() == "⬇ Pull model" and w.model_box.currentText() == "gemma3:12b" and "gemma3:12b" in w.installed
        # ошибка загрузки
        FakePuller.script = [("progress", "pulling manifest", None, None), ("fail", "pull model manifest: file does not exist")]
        n = len(fake_qt.QMessageBox.calls); w.toggle_pull(); wait(lambda: w.puller is None)
        assert fake_qt.QMessageBox.calls[n][0] == "question" and fake_qt.QMessageBox.calls[-1][0] == "warning" and "does not exist" in fake_qt.QMessageBox.calls[-1][1][1]
        # отмена: повторное нажатие кнопки
        FakePuller.script = [("progress", "pulling abc", i, 100) for i in range(1, 60)] + [("done",)]
        n = len(fake_qt.QMessageBox.calls); w.toggle_pull(); time.sleep(0.15); w.toggle_pull()
        assert w.pull_label.text() == "Cancelling…"; wait(lambda: w.puller is None)
        assert not any("installed" in str(c[1]) for c in fake_qt.QMessageBox.calls[n + 1:]) and w.pull_btn.text() == "⬇ Pull model"
    finally:
        app.PullWorker = real


def test_engine_failure_message():
    w = new_window()
    class Boom:
        def __init__(self, **kw): raise VLMError("cannot reach Ollama")
    app.Engine = Boom
    w.add_paths([str(mkimgs(2, "f"))]); wait(lambda: w.worker is None)
    assert fake_qt.QMessageBox.calls and fake_qt.QMessageBox.calls[0][0] == "critical"
    app.Engine = FakeEngine


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"): fn(); print("ok", name)
