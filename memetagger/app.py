import re, sys, threading, time
from pathlib import Path
from PIL import Image
from PySide6.QtCore import QSettings, QSize, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import (QAction, QBrush, QColor, QDesktopServices, QIcon, QImage, QPalette, QPixmap)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QFrame, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMenu,
                               QMessageBox, QProgressBar, QPushButton, QSpinBox, QSplitter, QStackedWidget,
                               QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)
from . import engine as eng_mod
from . import organizer as org
from .engine import Engine, INSTALL_HELP, Result, config_file
from .models import user_config_dir, user_data_dir
from .tagmemory import TagMemory
from .template import DEFAULT_TEMPLATE, LEGACY_STYLES, PRESETS, render_filename, validate
from .theme import THEMES, build_style, make_check_icon
from .vlm import ALL_WANT, OllamaVLM, VLMError, unload_models

_engine, _engine_key = None, None          # движок (и модели) создаётся один раз на набор настроек
MEMORY = TagMemory(user_data_dir() / "tag_memory.json")
DEFAULT_MODEL = "qwen3-vl:8b"
_SYSTEM_DARK = False


def detect_system_dark(app) -> bool:
    """Тёмная ли системная тема (Qt >= 6.5 сообщает схему; иначе смотрим на яркость окна)."""
    try:
        return app.styleHints().colorScheme() == Qt.ColorScheme.Dark
    except Exception:
        pass
    try:
        return app.palette().color(QPalette.ColorRole.Window).lightness() < 128
    except Exception:
        return False


def make_palette(t: dict) -> QPalette:
    R, G = QPalette.ColorRole, QPalette.ColorGroup
    roles = {R.Window: t["window"], R.WindowText: t["text"], R.Base: t["card"], R.AlternateBase: t["alt"],
             R.ToolTipBase: t["card"], R.ToolTipText: t["text"], R.Text: t["text"], R.Button: t["raised"],
             R.ButtonText: t["text"], R.Highlight: t["accent"], R.HighlightedText: t["on_accent"], R.Mid: t["border"],
             R.Midlight: t["hover"], R.Dark: t["border"], R.Light: t["raised"], R.PlaceholderText: t["muted"],
             R.Link: t["accent"]}
    p = QPalette()
    for role, c in roles.items():
        p.setColor(role, QColor(c))
    for role in (R.WindowText, R.Text, R.ButtonText):
        p.setColor(G.Disabled, role, QColor(t["disabled"]))
    return p


def apply_theme(app, mode: str):
    """Всегда применяем СВОЮ палитру и стили (а не системные роли): иначе на части тем элементы сливаются с фоном.
    «system» только выбирает светлый или тёмный вариант."""
    dark = _SYSTEM_DARK if mode not in ("dark", "light") else mode == "dark"
    t = THEMES["dark" if dark else "light"]
    app.setPalette(make_palette(t))
    app.setStyleSheet(build_style(t, make_check_icon(user_data_dir() / "ui" / "check.png")))


def set_state(widget, state: str):
    """Переключает состояние стиля (QLabel#badge[state="ok"] и т.п.); цвета берутся из текущей темы."""
    widget.setProperty("state", state)
    widget.style().unpolish(widget); widget.style().polish(widget)


def make_thumb(path) -> QImage:
    try:
        im = Image.open(path); im.thumbnail((128, 128)); im = im.convert("RGBA")
        return QImage(im.tobytes(), im.width, im.height, QImage.Format.Format_RGBA8888).copy()
    except Exception:
        return QImage()


def fmt_eta(sec: float) -> str:
    sec = int(sec)
    return f"{sec}s" if sec < 90 else (f"{sec // 60}m {sec % 60:02d}s" if sec < 3600 else f"{sec // 3600}h {sec % 3600 // 60:02d}m")


class Task(QThread):
    """Выполняет функцию в фоне и возвращает (результат, ошибка); колбэк вызывается в главном потоке."""
    done = Signal(object, object)

    def __init__(self, fn, cb=None):
        super().__init__()
        self.fn, self.cb = fn, cb

    def run(self):
        try:
            self.done.emit(self.fn(), None)
        except Exception as e:
            self.done.emit(None, f"{e}")


class Worker(QThread):
    row_done = Signal(int, object, QImage)   # row, Result или None, миниатюра
    row_error = Signal(int, str)
    status = Signal(str)
    failed = Signal(str)

    def __init__(self, jobs, opts):
        super().__init__()
        self.jobs, self.opts = jobs, opts
        self._go = threading.Event(); self._go.set()

    def pause(self): self._go.clear()
    def resume(self): self._go.set()
    def is_paused(self): return not self._go.is_set()

    def cancel(self):
        self.requestInterruption(); self._go.set()

    def run(self):
        global _engine, _engine_key
        key = tuple(sorted(self.opts["engine"].items()))
        try:
            if (_engine is None or _engine_key != key) and any(j[2] for j in self.jobs):
                _engine = Engine(**self.opts["engine"], memory=MEMORY, progress=self.status.emit)
                _engine_key = key
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            self.failed.emit(msg if isinstance(e, VLMError) else msg + "\n\nThe classic engine needs internet once to download its models.")
            return
        for row, path, analyze, ctx in self.jobs:
            while not self._go.wait(0.2):                 # пауза: ждём между изображениями
                if self.isInterruptionRequested():
                    break
            if self.isInterruptionRequested():
                break
            res = None
            if analyze:
                try:
                    res = _engine.tag(path, context=ctx, **self.opts["tag"])
                except Exception as e:
                    res = Result([], "", "error", want=frozenset()); self.row_error.emit(row, f"{type(e).__name__}: {e}")
            self.row_done.emit(row, res, make_thumb(path))
        if _engine is not None:
            _engine.flush()


class PullWorker(QThread):
    """`ollama pull` с прогрессом; отмена закрывает поток (Ollama продолжит с места при следующей загрузке)."""
    progress = Signal(str, object, object)      # статус, скачано, всего (байты; могут быть None)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, host, name):
        super().__init__()
        self.host, self.name = host, name

    def cancel(self):
        self.requestInterruption()

    def run(self):
        try:
            OllamaVLM(self.host, self.name).pull(self.name, lambda st, c, t: self.progress.emit(st, c, t),
                                                 self.isInterruptionRequested)
            self.done.emit(self.name)
        except VLMError as e:
            if e.code != "cancel":
                self.failed.emit(str(e))
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


def _card(layout) -> QFrame:
    f = QFrame(); f.setObjectName("card"); f.setLayout(layout)
    return f


class Main(QMainWindow):
    COLS = ["", "File", "Tags (double-click to edit)", "Text on the image (editable)", "Scene description (editable)", "New location"]
    CONFIG_FILES = [("vocab.txt", "classic CLIP labels (vocab.txt)"), ("folder_hints.json", "folder → tags (folder_hints.json)"),
                    ("seed_tags.txt", "starting tag vocabulary (seed_tags.txt)"), ("aliases.json", "tag aliases (aliases.json)")]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MemeTagger"); self.resize(1280, 800); self.setAcceptDrops(True)
        self.settings = QSettings("memetagger", "memetagger")
        self.items, self.queue, self.worker, self._busy = [], [], None, False
        self.row_errors, self.errors, self.installed, self._tasks = {}, 0, {}, set()
        self.prog_total = self.prog_done = 0
        self.avg_dt, self.t_last, self.stopped = None, None, False
        self.replan_timer = QTimer(singleShot=True, interval=250, timeout=self.replan)
        self.ps_timer = QTimer(interval=4000, timeout=self.poll_ps)
        self.save_timer = QTimer(singleShot=True, interval=400, timeout=self.save_settings)
        self.used_models, self.puller = set(), None

        root = QWidget(); outer = QVBoxLayout(root); outer.setContentsMargins(20, 16, 20, 14); outer.setSpacing(12)
        outer.addLayout(self._build_header())
        outer.addLayout(self._build_toolbar())
        split = self.split = QSplitter(Qt.Orientation.Horizontal); split.setChildrenCollapsible(False)
        split.addWidget(self._build_table_area()); split.addWidget(self._build_settings())
        split.setStretchFactor(0, 3); split.setStretchFactor(1, 1); split.setSizes([860, 400])
        outer.addWidget(split, 1)
        outer.addLayout(self._build_statusbar())
        self.setCentralWidget(root)
        self.update_controls(); self.update_template_ui(); self.update_headers()
        self.restore_window(); self._wire_settings()
        QTimer.singleShot(300, self.refresh_models)
        QTimer.singleShot(600, self.poll_ps)

    # ---------- построение интерфейса ----------
    def S(self, key, default, typ):
        return self.settings.value(key, default, typ)

    def _btn(self, text, slot, name=""):
        b = QPushButton(text); b.clicked.connect(slot)
        if name: b.setObjectName(name)
        return b

    def _build_header(self):
        row = QHBoxLayout()
        title = QLabel("MemeTagger"); title.setObjectName("title")
        sub = QLabel("Local AI tagging & sorting for your meme library"); sub.setObjectName("muted")
        col = QVBoxLayout(); col.setSpacing(0); col.addWidget(title); col.addWidget(sub)
        self.badge = QLabel("● checking Ollama…"); self.badge.setObjectName("badge")
        self.theme_box = QComboBox()
        for label, key in (("Theme: system", "system"), ("Theme: dark", "dark"), ("Theme: light", "light")):
            self.theme_box.addItem(label, key)
        self.theme_box.setCurrentIndex(max(0, self.theme_box.findData(self.S("theme", "system", str))))
        self.theme_box.currentIndexChanged.connect(self.on_theme)
        row.addLayout(col); row.addStretch(1); row.addWidget(self.badge); row.addWidget(self.theme_box)
        return row

    def _build_toolbar(self):
        row = QHBoxLayout(); row.setSpacing(8)
        self.add_files_btn = self._btn("＋ Add files", self.pick_files)
        self.add_dir_btn = self._btn("Add folder", self.pick_dir)
        self.pause_btn = self._btn("⏸ Pause", self.toggle_pause)
        self.stop_btn = self._btn("■ Stop", self.stop_analysis, "danger")
        self.reanalyze_btn = self._btn("↻ Re-analyze", self.reanalyze)
        self.clear_btn = self._btn("Clear", self.clear, "flat")
        self.filter_edit = QLineEdit(); self.filter_edit.setPlaceholderText("Filter by name or tag…")
        self.filter_edit.setClearButtonEnabled(True); self.filter_edit.setMinimumWidth(200)
        self.filter_edit.textChanged.connect(self.apply_filter)
        self.untagged_cb = QCheckBox("Only untagged"); self.untagged_cb.toggled.connect(self.apply_filter)
        self.config_btn = QPushButton("⚙ Config"); self.config_btn.setMenu(self._build_config_menu())
        self.undo_btn = self._btn("↩ Undo", self.undo)
        self.apply_btn = self._btn("Apply", self.apply, "primary")
        for w in (self.add_files_btn, self.add_dir_btn, self.pause_btn, self.stop_btn, self.reanalyze_btn, self.clear_btn):
            row.addWidget(w)
        row.addStretch(1)
        for w in (self.filter_edit, self.untagged_cb, self.config_btn, self.undo_btn, self.apply_btn):
            row.addWidget(w)
        return row

    def _build_config_menu(self):
        menu = QMenu(self)

        def act(text, fn):
            a = QAction(text, self); a.triggered.connect(fn); menu.addAction(a)
        act("Open config folder", lambda checked=False: self.open_path(user_config_dir()))
        menu.addSeparator()
        for name, label in self.CONFIG_FILES:
            act("Edit " + label, lambda checked=False, n=name: self.open_config(n))
        menu.addSeparator()
        act("Open data folder (models, tag memory, undo journal)", lambda checked=False: self.open_path(user_data_dir()))
        act("Reload configuration", lambda checked=False: self.reload_config())
        return menu

    def _build_table_area(self):
        self.stack = QStackedWidget()
        drop = QLabel("Drop images or folders here\n\nor use “Add files” / “Add folder”")
        drop.setObjectName("drop"); drop.setAlignment(Qt.AlignmentFlag.AlignCenter)
        holder = QWidget(); hl = QVBoxLayout(holder); hl.setContentsMargins(40, 40, 40, 40); hl.addWidget(drop)
        self.table = QTableWidget(0, 6); self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.setIconSize(QSize(72, 72)); self.table.verticalHeader().setDefaultSectionSize(84)
        self.table.verticalHeader().hide(); self.table.setShowGrid(False); self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setWordWrap(True)
        h = self.table.horizontalHeader(); h.setSectionResizeMode(QHeaderView.ResizeMode.Interactive); h.setStretchLastSection(True)
        self.table.setColumnWidth(0, 92); self.table.setColumnWidth(1, 180); self.table.setColumnWidth(2, 230)
        self.table.setColumnWidth(3, 200); self.table.setColumnWidth(4, 280)
        self.table.itemChanged.connect(self.on_edit)
        self.table.itemSelectionChanged.connect(self.update_template_ui)
        lay = QVBoxLayout(); lay.setContentsMargins(4, 4, 4, 4); lay.addWidget(self.table)
        self.stack.addWidget(holder); self.stack.addWidget(_card(lay))
        return self.stack

    @staticmethod
    def _tab(form):
        w = QWidget(); outer = QVBoxLayout(w); outer.setContentsMargins(12, 14, 12, 8); outer.addLayout(form); outer.addStretch(1)
        return w

    def _build_settings(self):
        tabs = self.tabs = QTabWidget(); tabs.setDocumentMode(True); tabs.tabBar().setExpanding(False); tabs.tabBar().setDrawBase(False)
        # --- Engine ---
        self.engine_box = QComboBox()
        self.engine_box.addItem("Auto: Ollama vision model, else CLIP+WD14", "auto")
        self.engine_box.addItem("Ollama vision model only (best quality)", "vlm")
        self.engine_box.addItem("Classic CLIP+WD14 only (fast, weaker)", "classic")
        self.engine_box.setCurrentIndex(max(0, self.engine_box.findData(self.S("engine", "auto", str))))
        self.model_box = QComboBox(); self.model_box.setEditable(True)
        self.model_box.setCurrentText(self.S("model", DEFAULT_MODEL, str))
        self.model_box.currentTextChanged.connect(self.update_model_hint)
        refresh = self._btn("⟳", self.refresh_models, "flat"); refresh.setToolTip("Reload the list of installed Ollama models")
        mrow = QHBoxLayout(); mrow.addWidget(self.model_box, 1); mrow.addWidget(refresh)
        self.model_hint = QLabel(""); self.model_hint.setObjectName("hint"); self.model_hint.setWordWrap(True)
        self.pull_btn = self._btn("⬇ Pull model", self.toggle_pull)
        self.pull_btn.setToolTip("Download the model named above from the Ollama library (like `ollama pull`)")
        self.unload_btn = self._btn("⏏ Unload from memory", self.unload_model)
        self.unload_btn.setToolTip("Free the GPU memory right now (like `ollama stop <model>`). The next analysis reloads the model.")
        mbtn = QHBoxLayout(); mbtn.addWidget(self.pull_btn); mbtn.addWidget(self.unload_btn); mbtn.addStretch(1)
        self.pull_label = QLabel(""); self.pull_label.setObjectName("hint"); self.pull_label.setWordWrap(True)
        self.pull_bar = QProgressBar(); self.pull_bar.setTextVisible(False); self.pull_bar.setRange(0, 100)
        self.pull_label.setVisible(False); self.pull_bar.setVisible(False)
        self.unload_exit = QCheckBox("Unload the model from memory when closing the app")
        self.unload_exit.setChecked(self.S("unload_exit", True, bool))
        self.unload_exit.setToolTip("Only models this app used in this session are unloaded.")
        self.host_edit = QLineEdit(self.S("host", "", str)); self.host_edit.setPlaceholderText("empty = $OLLAMA_HOST or http://127.0.0.1:11434")
        self.host_edit.editingFinished.connect(self.refresh_models)
        hrow = QHBoxLayout(); hrow.addWidget(self.host_edit, 1); hrow.addWidget(self._btn("Test", self.test_ollama))
        self.gen_tags = QCheckBox("Tags"); self.gen_tags.setChecked(self.S("gen_tags", True, bool))
        self.gen_tags.setToolTip("Used for folders and search. Off: skips WD14 character hints, verification queries and the tag vocabulary in the prompt.")
        self.gen_text = QCheckBox("Text from the image"); self.gen_text.setChecked(self.S("gen_text", True, bool))
        self.gen_text.setToolTip("For file names. Off: skips the text field (and OCR, unless tags still use it as a hint).")
        self.gen_desc = QCheckBox("Scene description"); self.gen_desc.setChecked(self.S("gen_desc", True, bool))
        self.gen_desc.setToolTip("One or two sentences about the scene. Off: the model's answer gets shorter.")
        gen_row = QHBoxLayout(); gen_row.addWidget(self.gen_tags); gen_row.addWidget(self.gen_text); gen_row.addWidget(self.gen_desc); gen_row.addStretch(1)
        self.gen_hint = QLabel("Fewer items = faster: unneeded steps are skipped, not just hidden. "
                               "Existing values in the table are kept when an item is off.")
        self.gen_hint.setObjectName("hint"); self.gen_hint.setWordWrap(True)
        for w in (self.gen_tags, self.gen_text, self.gen_desc):
            w.toggled.connect(self.on_want_changed)
        self.ocr_cb = QCheckBox("Read text on images (OCR, CPU)"); self.ocr_cb.setChecked(self.S("ocr", True, bool))
        self.verify_cb = QCheckBox("Verify franchise/character guesses (extra short query)"); self.verify_cb.setChecked(self.S("verify", True, bool))
        self.fallback = QCheckBox("Always assign at least one tag (best guess)"); self.fallback.setChecked(self.S("fallback", True, bool))
        self.maxtags = QSpinBox(); self.maxtags.setRange(1, 15); self.maxtags.setValue(self.S("maxtags", 6, int))
        self.cthr = QDoubleSpinBox(); self.cthr.setRange(0.05, 0.95); self.cthr.setSingleStep(0.05); self.cthr.setValue(self.S("cthr", 0.30, float))
        self.cthr.setToolTip("Classic engine: higher = fewer, more certain tags. Applies to the next analysis.")
        self.gthr = QDoubleSpinBox(); self.gthr.setRange(0.1, 0.95); self.gthr.setSingleStep(0.05); self.gthr.setValue(self.S("gthr", 0.35, float))
        self.gpu_label = QLabel("GPU: —"); self.gpu_label.setObjectName("hint"); self.gpu_label.setWordWrap(True)
        f = QFormLayout(); f.setVerticalSpacing(10)
        f.addRow("Engine", self.engine_box); f.addRow("Model", mrow); f.addRow("", self.model_hint); f.addRow("", mbtn); f.addRow(self.pull_label); f.addRow(self.pull_bar)
        f.addRow("Ollama URL", hrow)
        f.addRow("Generate", gen_row); f.addRow(self.gen_hint)
        f.addRow(self.ocr_cb); f.addRow(self.verify_cb); f.addRow(self.fallback)
        f.addRow("Max tags per file", self.maxtags); f.addRow("Classic: concept confidence", self.cthr)
        f.addRow("Classic: WD14 confidence", self.gthr); f.addRow(self.gpu_label); f.addRow(self.unload_exit)
        tabs.addTab(self._tab(f), "Engine")
        # --- Naming & folders ---
        self.out = QLineEdit(self.S("out", "", str)); self.out.setPlaceholderText("empty = rename in place")
        orow = QHBoxLayout(); orow.addWidget(self.out, 1); orow.addWidget(self._btn("…", self.pick_out))
        self.preset_box = QComboBox()
        for name, _t in PRESETS:
            self.preset_box.addItem(name)
        self.preset_box.addItem("Custom")
        tpl = self.S("template", "", str) or LEGACY_STYLES.get(self.S("name_style", "keep", str), DEFAULT_TEMPLATE)
        self.template_edit = QLineEdit(tpl); self.template_edit.setPlaceholderText(DEFAULT_TEMPLATE)
        self.tpl_preview = QLabel(""); self.tpl_preview.setObjectName("hint"); self.tpl_preview.setWordWrap(True)
        self.tpl_warn = QLabel(""); self.tpl_warn.setObjectName("hint"); self.tpl_warn.setWordWrap(True)
        self.tpl_help = QLabel("Variables: {original} {meaningful} {text} {description} {tags}\n"
                               "Fallback: {text|description|original}  ·  Limit: {text:40}, {tags:3}\n"
                               "Optional part (dropped if a variable inside is empty): < - {text}>")
        self.tpl_help.setObjectName("hint"); self.tpl_help.setWordWrap(True)
        self.text_max = QSpinBox(); self.text_max.setRange(15, 120); self.text_max.setValue(self.S("text_max", 60, int))
        self.text_max.setToolTip("Default length of {text}; longer text is cut at a word boundary. The AI already shortens long texts.")
        self.desc_max = QSpinBox(); self.desc_max.setRange(15, 160); self.desc_max.setValue(self.S("desc_max", 100, int))
        self.desc_max.setToolTip("Default length of {description} in the file name (the table always keeps the full description).")
        self.limit_model = QCheckBox("Ask the model to respect these lengths (shorter, faster answers)")
        self.limit_model.setChecked(self.S("limit_model", True, bool))
        self.limit_model.setToolTip("On: the length is given to the model as a target in WORDS (no character counting, no line wrapping) "
                                    "plus a hard safety limit in the answer format, so descriptions come out shorter and quicker. "
                                    "Off: the model writes freely and long values are only trimmed in file names.")
        self.desc_lang = QComboBox()
        for label, key in (("English", "English"), ("Русский", "Russian"), ("Українська", "Ukrainian")):
            self.desc_lang.addItem(label, key)
        self.desc_lang.setCurrentIndex(max(0, self.desc_lang.findData(self.S("desc_lang", "English", str))))
        self.desc_lang.setToolTip("Language of the scene description written by the AI model (tags stay English, text keeps its language).")
        self.sync_preset()
        self.preset_box.currentIndexChanged.connect(self.on_preset)
        self.template_edit.textChanged.connect(self.on_template_changed)
        self.copy = QCheckBox("Copy instead of move (library folder set)"); self.copy.setChecked(self.S("copy", False, bool))
        self.merge = QCheckBox("Re-organize files already in the library"); self.merge.setChecked(self.S("merge", True, bool))
        self.skip = QCheckBox("Skip files that already have [tags]"); self.skip.setChecked(self.S("skip", True, bool))
        self.skip.setToolTip("Applies while tags are being generated. If tags are off, files are analyzed for the other items.")
        self.gmin = QSpinBox(); self.gmin.setRange(2, 1000); self.gmin.setValue(self.S("gmin", 12, int))
        self.depth = QSpinBox(); self.depth.setRange(0, 10); self.depth.setValue(self.S("depth", 4, int))
        f2 = QFormLayout(); f2.setVerticalSpacing(10)
        f2.addRow("Library folder", orow); f2.addRow("File name preset", self.preset_box); f2.addRow("Template", self.template_edit)
        f2.addRow(self.tpl_preview); f2.addRow(self.tpl_warn); f2.addRow(self.tpl_help)
        f2.addRow("Max text length", self.text_max); f2.addRow("Max description length", self.desc_max); f2.addRow(self.limit_model)
        f2.addRow("Description language", self.desc_lang); f2.addRow(self.copy)
        f2.addRow(self.merge); f2.addRow(self.skip)
        f2.addRow("Folder threshold (images per tag)", self.gmin); f2.addRow("Max folder depth", self.depth)
        tabs.addTab(self._tab(f2), "Naming & folders")
        for w in (self.copy, self.merge, self.skip):
            w.toggled.connect(self.replan_timer.start)
        for w in (self.gmin, self.depth, self.text_max, self.desc_max):
            w.valueChanged.connect(self.replan_timer.start)
            w.valueChanged.connect(self.update_template_ui)
        self.out.textChanged.connect(self.replan_timer.start)
        # --- Tags ---
        self.banned_edit = QLineEdit(self.S("banned", "", str))
        self.banned_edit.setPlaceholderText("space/comma separated, e.g.  photo  text_meme")
        self.banned_edit.editingFinished.connect(self.on_banned_changed)
        note = QLabel("Banned tags are never assigned and are removed from files already in the list. "
                      "Use ⚙ Config to edit folder hints, the starting vocabulary and aliases, then “Reload configuration”.")
        note.setObjectName("hint"); note.setWordWrap(True)
        f3 = QFormLayout(); f3.setVerticalSpacing(10)
        f3.addRow("Banned tags", self.banned_edit); f3.addRow(note); f3.addRow(self._btn("Reload configuration", self.reload_config))
        tabs.addTab(self._tab(f3), "Tags")
        lay = QVBoxLayout(); lay.addWidget(tabs)
        card = _card(lay); card.setMinimumWidth(340)
        return card

    def _build_statusbar(self):
        row = QHBoxLayout(); row.setSpacing(12)
        self.info = QLabel("Ready"); self.info.setObjectName("muted")
        self.eta_label = QLabel(""); self.eta_label.setObjectName("muted")
        self.progress = QProgressBar(); self.progress.setTextVisible(False); self.progress.setRange(0, 1); self.progress.setValue(0)
        self.progress.setMinimumWidth(240)
        row.addWidget(self.info, 1); row.addWidget(self.eta_label); row.addWidget(self.progress)
        return row

    # ---------- тема, конфиги ----------
    def on_theme(self, *_):
        self.settings.setValue("theme", self.theme_box.currentData())
        apply_theme(QApplication.instance(), self.theme_box.currentData())

    def open_path(self, path):
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
            QMessageBox.information(self, "Open", f"Could not open it automatically. Path:\n{path}")

    def open_config(self, name):
        self.open_path(config_file(name))      # создаёт файл из дефолта, если его ещё нет

    def reload_config(self, *_):
        global _engine
        if self.worker is not None:
            QMessageBox.information(self, "MemeTagger", "Pause or stop the analysis first (or wait until it finishes)."); return
        _engine = None; eng_mod.reset_shared()
        self.info.setText("Configuration will be reloaded on the next analysis")

    # ---------- фоновые задачи, Ollama: модели и GPU ----------
    def run_task(self, fn, callback):
        t = Task(fn, callback); self._tasks.add(t)
        t.done.connect(self._task_done); t.finished.connect(self._task_finished)
        t.start()

    def _task_done(self, res, err):
        cb = getattr(self.sender(), "cb", None)
        if cb: cb(res, err)

    def _task_finished(self):
        self._tasks = {t for t in self._tasks if t.isRunning()}

    def _host(self):
        return self.host_edit.text().strip()

    def refresh_models(self, *_):
        host = self._host()
        self.run_task(lambda: OllamaVLM(host, "").list_models(), self.on_models)

    def on_models(self, models, err):
        if err:
            self.installed = {}
            self.badge.setText("● Ollama offline"); set_state(self.badge, "bad")
            self.model_hint.setText(f"Could not reach Ollama ({err}).\n" + INSTALL_HELP.splitlines()[0]); set_state(self.model_hint, "warn")
            return
        self.installed = {m["name"]: m for m in models}
        usable = [m["name"] for m in models if m["vision"] is not False]
        hidden = len(models) - len(usable)
        current = self.model_box.currentText().strip()
        if not current and usable:
            current = next((n for n in usable if "qwen3-vl" in n), usable[0])
        self.model_box.blockSignals(True)
        self.model_box.clear(); self.model_box.addItems(usable); self.model_box.setCurrentText(current or DEFAULT_MODEL)
        self.model_box.blockSignals(False)
        self.badge.setText(f"● Ollama online · {len(usable)} vision model(s)" + (f" · {hidden} text-only hidden" if hidden else ""))
        set_state(self.badge, "ok")
        self.update_model_hint()

    def update_model_hint(self, *_):
        name = self.model_box.currentText().strip()
        if not self.installed:
            return
        m = self.installed.get(name) or self.installed.get(name + ":latest")
        if m:
            extra = " · text-only model, it cannot see images" if m["vision"] is False else ""
            self.model_hint.setText(f"Installed · {m['size'] / 1e9:.1f} GB{extra}"); set_state(self.model_hint, "warn" if extra else "")
        else:
            self.model_hint.setText(f"Not installed. Run:  ollama pull {name}" if name else "Pick or type a model name")
            set_state(self.model_hint, "warn")

    def poll_ps(self, *_):
        host = self._host()
        self.run_task(lambda: OllamaVLM(host, "").loaded(), self.on_ps)

    def on_ps(self, loaded, err):
        if err:
            return
        if not loaded:
            self.gpu_label.setText("GPU: no model loaded right now"); set_state(self.gpu_label, ""); return
        parts, partial = [], False
        for m in loaded:
            pct = int(100 * m["vram"] / m["size"]) if m["size"] else 0
            partial = partial or pct < 100
            parts.append(f"{m['name']} · {m['size'] / 1e9:.1f} GB · {pct}% in VRAM")
        self.gpu_label.setText("GPU: " + "; ".join(parts) + ("  ⚠ partly on CPU: expect slow speed" if partial else ""))
        set_state(self.gpu_label, "warn" if partial else "")

    def test_ollama(self, *_):
        host, model = self._host(), self.model_box.currentText()

        def done(res, err):
            ok, msg = res if res else (False, err)
            (QMessageBox.information if ok else QMessageBox.warning)(self, "Ollama", msg if ok else f"{msg}\n\n{INSTALL_HELP}")
        self.run_task(lambda: OllamaVLM(host, model).status(), done)
        self.refresh_models()

    # ---------- ввод ----------
    def pick_files(self, *_):
        fs, _ = QFileDialog.getOpenFileNames(self, "Images", "", "Images (*.jpg *.jpeg *.png *.webp *.gif *.bmp *.tif *.tiff *.avif *.jfif)")
        self.add_paths(fs)

    def pick_dir(self, *_):
        d = QFileDialog.getExistingDirectory(self, "Folder with images")
        if d: self.add_paths([d])

    def pick_out(self, *_):
        d = QFileDialog.getExistingDirectory(self, "Library folder")
        if d: self.out.setText(d)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()

    def dropEvent(self, e):
        self.add_paths([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])

    def clear(self, *_):
        self.queue.clear()
        if self.worker is not None:
            self.worker.cancel(); self.worker.wait()
            QApplication.processEvents()                 # доставляем «хвост» сигналов, пока список items ещё прежний
            self.worker = None
        self.items.clear(); self.row_errors.clear(); self.errors = 0; self.stopped = False
        self.prog_total = self.prog_done = 0; self.table.setRowCount(0)
        self.info.setText("Ready"); self.eta_label.setText(""); self.progress.setRange(0, 1); self.progress.setValue(0)
        self.update_controls()

    def want(self) -> frozenset:
        w = {k for k, cb in (("tags", self.gen_tags), ("text", self.gen_text), ("description", self.gen_desc)) if cb.isChecked()}
        return frozenset(w) or ALL_WANT

    def on_want_changed(self, *_):
        if not (self.gen_tags.isChecked() or self.gen_text.isChecked() or self.gen_desc.isChecked()):
            self.gen_tags.setChecked(True)                       # хотя бы что-то должно генерироваться
            self.info.setText("At least one of tags / text / description must stay on"); return
        for k, cb in (("gen_tags", self.gen_tags), ("gen_text", self.gen_text), ("gen_desc", self.gen_desc)):
            self.settings.setValue(k, cb.isChecked())
        self.update_headers(); self.update_template_ui()

    def update_headers(self):
        w = self.want(); labels = list(self.COLS)
        for col, key in ((2, "tags"), (3, "text"), (4, "description")):
            if key not in w:
                labels[col] += "  (generation off)"
        self.table.setHorizontalHeaderLabels(labels)

    # ---------- настройки: сохраняются сразу при изменении и при выходе ----------
    def setting_values(self) -> dict:
        cb = lambda w: w.isChecked()
        return {"engine": self.engine_box.currentData(), "model": self.model_box.currentText().strip(), "host": self._host(),
                "ocr": cb(self.ocr_cb), "verify": cb(self.verify_cb), "fallback": cb(self.fallback), "gen_tags": cb(self.gen_tags),
                "gen_text": cb(self.gen_text), "gen_desc": cb(self.gen_desc), "unload_exit": cb(self.unload_exit),
                "maxtags": self.maxtags.value(), "cthr": self.cthr.value(), "gthr": self.gthr.value(),
                "out": self.out.text(), "template": self.template_edit.text(), "text_max": self.text_max.value(),
                "desc_max": self.desc_max.value(), "desc_lang": self.desc_lang.currentData(), "limit_model": cb(self.limit_model),
                "copy": cb(self.copy), "merge": cb(self.merge), "skip": cb(self.skip), "gmin": self.gmin.value(),
                "depth": self.depth.value(), "banned": self.banned_edit.text(), "theme": self.theme_box.currentData(),
                "tab": self.tabs.currentIndex()}

    def save_settings(self, *_):
        for k, v in self.setting_values().items():
            self.settings.setValue(k, v)
        try:
            self.settings.setValue("splitter", list(self.split.sizes())); self.settings.setValue("geometry", self.saveGeometry())
        except Exception:
            pass
        self.settings.sync()

    def schedule_save(self, *_):
        self.save_timer.start()

    def _wire_settings(self):
        """Любое изменение любой настройки запускает отложенное сохранение (раньше сохранялось только при запуске анализа)."""
        for w in (self.ocr_cb, self.verify_cb, self.fallback, self.gen_tags, self.gen_text, self.gen_desc, self.unload_exit,
                  self.limit_model, self.copy, self.merge, self.skip):
            w.toggled.connect(self.schedule_save)
        for w in (self.engine_box, self.desc_lang, self.theme_box, self.preset_box):
            w.currentIndexChanged.connect(self.schedule_save)
        self.model_box.currentTextChanged.connect(self.schedule_save)
        for w in (self.host_edit, self.out, self.template_edit, self.banned_edit):
            w.textChanged.connect(self.schedule_save)
        for w in (self.maxtags, self.cthr, self.gthr, self.text_max, self.desc_max, self.gmin, self.depth):
            w.valueChanged.connect(self.schedule_save)
        self.tabs.currentChanged.connect(self.schedule_save)
        self.split.splitterMoved.connect(self.schedule_save)

    def restore_window(self):
        try:
            g = self.settings.value("geometry")
            if g: self.restoreGeometry(g)
            sizes = self.settings.value("splitter")
            if sizes: self.split.setSizes([int(x) for x in sizes])
            self.tabs.setCurrentIndex(int(self.S("tab", 0, int)))
        except Exception:
            pass

    def limits(self):
        """Лимиты длины для модели (в словах + страховка maxLength) или None, если отключено."""
        if not self.limit_model.isChecked():
            return None
        return {"desc_chars": self.desc_max.value(), "text_chars": self.text_max.value()}

    # ---------- модель: выгрузка и загрузка ----------
    def unload_model(self, *_):
        names, host = set(self.used_models) | {self.model_box.currentText().strip()}, self._host()

        def done(res, err):
            self.info.setText(f"Unload failed: {err}" if err else (f"Unloaded from memory: {', '.join(res)}" if res
                                                                    else "The model was not loaded"))
            self.poll_ps()
        self.run_task(lambda: unload_models(host, names), done)

    def unload_sync(self):
        """При выходе: выгружаем только то, чем пользовались в этой сессии; ошибки не мешают закрытию."""
        if not self.used_models:
            return
        try:
            unload_models(self._host(), set(self.used_models), timeout=3)
        except Exception:
            pass

    def toggle_pull(self, *_):
        if self.puller is not None:                                  # идёт загрузка -> кнопка работает как «Отмена»
            self.puller.cancel(); self.pull_label.setText("Cancelling…"); return
        name = self.model_box.currentText().strip()
        if not name or any(c.isspace() for c in name):
            QMessageBox.information(self, "Pull model", "Type the model name in the “Model” box first, e.g. qwen3-vl:8b"); return
        again = name in self.installed or name + ":latest" in self.installed
        q = f"{'Update' if again else 'Download'} “{name}” from the Ollama library?\nIt can take several GB and a few minutes."
        if QMessageBox.question(self, "Pull model", q) != QMessageBox.StandardButton.Yes:
            return
        self.puller = PullWorker(self._host(), name)
        self.puller.progress.connect(self.on_pull_progress); self.puller.done.connect(self.on_pull_done)
        self.puller.failed.connect(self.on_pull_failed); self.puller.finished.connect(self.on_pull_finished)
        self.pull_label.setText(f"Starting download of {name}…"); self.pull_bar.setRange(0, 0)
        self.pull_label.setVisible(True); self.pull_bar.setVisible(True)
        self.pull_btn.setText("✕ Cancel pull")
        self.puller.start()

    def on_pull_progress(self, status, completed, total):
        if total:
            self.pull_bar.setRange(0, 100); self.pull_bar.setValue(int(100 * (completed or 0) / total))
            self.pull_label.setText(f"{status[:36]} · {(completed or 0) / 1e9:.2f} / {total / 1e9:.2f} GB")
        else:
            self.pull_bar.setRange(0, 0); self.pull_label.setText(status)

    def on_pull_done(self, name):
        QMessageBox.information(self, "Pull model", f"“{name}” is installed.")
        self.refresh_models(); self.model_box.setCurrentText(name)

    def on_pull_failed(self, msg):
        QMessageBox.warning(self, "Pull failed", msg)

    def on_pull_finished(self, *_):
        w, self.puller = self.puller, None
        if w is not None:
            w.wait()
        self.pull_label.setVisible(False); self.pull_bar.setVisible(False); self.pull_btn.setText("⬇ Pull model")

    def banned(self) -> set:
        return {t for t in map(org.clean_tag, re.split(r"[,\s]+", self.banned_edit.text())) if t}

    def add_paths(self, paths):
        known = {i.src for i in self.items}
        new = [i for i in org.collect(paths) if i.src not in known]
        if not new: return
        if self.worker is None and not self.queue:        # простой -> новый «забег», прогресс с нуля
            self.prog_total = self.prog_done = 0
        self.stopped = False
        ban = self.banned()
        for it in new:
            it.tags = [t for t in it.tags if t not in ban]
            MEMORY.observe(it.tags)
        start = len(self.items); self.items += new
        self._busy = True; self.table.setRowCount(len(self.items))
        for k, it in enumerate(new):
            r = start + k
            cells = [QTableWidgetItem(), QTableWidgetItem(it.src.name), QTableWidgetItem(" ".join(it.tags)), QTableWidgetItem(it.text),
                     QTableWidgetItem(it.description), QTableWidgetItem("…")]
            for c, cell in enumerate(cells):
                self.table.setItem(r, c, cell)
                if c not in (2, 3, 4): cell.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            cells[1].setToolTip(str(it.src))
            self.mark_empty(r)
            skip_it = bool(it.tags) and self.skip.isChecked() and "tags" in self.want()
            self.queue.append((r, it.src, not skip_it, org.item_context(it)))
        self._busy = False
        self.prog_total += len(new)
        self.start_worker(); self.update_controls(); self.apply_filter()

    # ---------- анализ: запуск, пауза, стоп ----------
    def start_worker(self):
        if self.worker is not None or not self.queue:
            return
        jobs, self.queue = self.queue, []
        self.save_settings()
        if self.engine_box.currentData() != "classic":
            self.used_models.add(self.model_box.currentText().strip())
        opts = {"engine": dict(mode=self.engine_box.currentData(), host=self._host(), model=self.model_box.currentText().strip(),
                               ocr=self.ocr_cb.isChecked(), verify=self.verify_cb.isChecked(),
                               want_tags="tags" in self.want()),
                "tag": dict(max_tags=self.maxtags.value(), general_thr=self.gthr.value(), concept_thr=self.cthr.value(),
                            banned=frozenset(self.banned()), fallback=self.fallback.isChecked(),
                            desc_lang=self.desc_lang.currentData(), want=self.want(),
                            limits=self.limits())}
        self.worker = Worker(jobs, opts)
        self.worker.row_done.connect(self.on_row); self.worker.row_error.connect(self.on_row_error)
        self.worker.status.connect(self.info.setText); self.worker.failed.connect(self.on_fail)
        self.worker.finished.connect(self.on_worker_finished)
        self.t_last = time.time(); self.avg_dt = None
        self.worker.start(); self.ps_timer.start(); self.update_controls()

    def on_worker_finished(self, *_):
        w, self.worker = self.worker, None
        if w is not None:
            w.wait()
        self.ps_timer.stop(); self.poll_ps(); self.eta_label.setText("")
        self.replan_timer.start()
        self.start_worker(); self.update_controls()

    def toggle_pause(self, *_):
        if self.worker is None: return
        if self.worker.is_paused():
            self.worker.resume(); self.t_last = time.time()
        else:
            self.worker.pause()
        self.update_controls(); self.update_progress()

    def stop_analysis(self, *_):
        if self.worker is None and not self.queue: return
        self.queue.clear(); self.stopped = True
        if self.worker is not None:
            self.worker.cancel()
        self.prog_total = self.prog_done
        self.update_controls()

    def reanalyze(self, *_):
        if self.worker is not None: return
        self.row_errors.clear(); self.errors = 0; self.stopped = False
        self.queue = [(r, it.src, True, org.item_context(it)) for r, it in enumerate(self.items)]
        self.prog_done, self.prog_total = 0, len(self.queue)
        self.start_worker(); self.update_controls()

    def update_controls(self):
        running = self.worker is not None
        paused = running and self.worker.is_paused()
        self.pause_btn.setText("▶ Resume" if paused else "⏸ Pause")
        self.pause_btn.setEnabled(running)
        self.stop_btn.setEnabled(running or bool(self.queue))
        self.reanalyze_btn.setEnabled(bool(self.items) and not running)
        self.apply_btn.setEnabled(bool(self.items) and not running)
        self.clear_btn.setEnabled(bool(self.items))
        self.unload_btn.setEnabled(not running)
        self.stack.setCurrentIndex(1 if self.items else 0)

    def update_progress(self):
        total = max(self.prog_total, 1); done = min(self.prog_done, total)
        self.progress.setRange(0, total); self.progress.setValue(done)
        if self.worker is not None:
            paused = self.worker.is_paused()
            self.info.setText(f"{'Paused' if paused else 'Analyzing'}: {done}/{self.prog_total}")
            left = self.prog_total - self.prog_done
            self.eta_label.setText("" if paused or self.avg_dt is None else f"~{fmt_eta(self.avg_dt * left)} left")

    def on_fail(self, msg):
        self.prog_total = self.prog_done
        QMessageBox.critical(self, "Could not start the analysis engine", msg)
        self.info.setText("Engine failed to start")

    # ---------- таблица ----------
    def on_row(self, row, res, thumb):
        if row >= len(self.items): return
        self._busy = True
        if not thumb.isNull(): self.table.item(row, 0).setIcon(QIcon(QPixmap.fromImage(thumb)))
        if res is not None:
            if "tags" in res.want:                               # выключенное не затираем: в таблице остаются прежние значения
                self.items[row].tags = res.tags; self.table.item(row, 2).setText(" ".join(res.tags))
            if "text" in res.want:
                self.items[row].text = res.caption; self.table.item(row, 3).setText(res.caption)
            if "description" in res.want:
                self.items[row].description = res.description; self.table.item(row, 4).setText(res.description)
            err = self.row_errors.get(row, "")
            self.table.item(row, 2).setToolTip(f"Analysis failed: {err}" if err else "")
            tip = f"[{res.source}] {res.description}".strip() + (f"\nOCR: {res.text}" if res.text else "")
            self.table.item(row, 1).setToolTip(f"{self.items[row].src}\n\n{tip}" + (f"\n({res.note})" if res.note else ""))
            self.mark_empty(row)
            now = time.time()
            if self.t_last is not None:
                dt = now - self.t_last
                self.avg_dt = dt if self.avg_dt is None else 0.8 * self.avg_dt + 0.2 * dt
            self.t_last = now
        self._busy = False
        self.prog_done += 1
        self.update_progress()
        self.replan_timer.start()

    def on_row_error(self, row, msg):
        self.row_errors[row] = msg
        self.errors = len(self.row_errors)

    def mark_empty(self, row):
        self.table.item(row, 2).setBackground(QBrush(QColor(255, 170, 0, 70)) if not self.items[row].tags else QBrush())

    def on_edit(self, cell):
        if self._busy or cell.column() not in (2, 3, 4): return
        if cell.column() in (3, 4):                         # ручная правка текста / описания
            setattr(self.items[cell.row()], "text" if cell.column() == 3 else "description", cell.text().strip())
            self.replan_timer.start(); self.update_template_ui(); return
        tags = [t for t in map(org.clean_tag, re.split(r"[,\s]+", cell.text())) if t]
        self.items[cell.row()].tags = tags
        self._busy = True; cell.setText(" ".join(tags)); self.mark_empty(cell.row()); self._busy = False
        self.replan_timer.start()

    def on_banned_changed(self, *_):
        self.settings.setValue("banned", self.banned_edit.text())
        ban = self.banned(); self._busy = True
        for r, it in enumerate(self.items):
            kept = [t for t in it.tags if t not in ban]
            if kept != it.tags:
                it.tags = kept; self.table.item(r, 2).setText(" ".join(kept)); self.mark_empty(r)
        self._busy = False
        self.replan_timer.start()

    def apply_filter(self, *_):
        q = self.filter_edit.text().strip().lower(); only = self.untagged_cb.isChecked()
        for r, it in enumerate(self.items):
            hay = (it.src.name + " " + " ".join(it.tags) + " " + it.text + " " + it.description).lower()
            self.table.setRowHidden(r, bool((q and q not in hay) or (only and it.tags)))

    def current_template(self) -> str:
        tpl = self.template_edit.text().strip()
        return tpl if not validate(tpl)[0] else DEFAULT_TEMPLATE      # при ошибке в шаблоне — безопасный шаблон по умолчанию

    def current_plan(self):
        return org.make_plan(self.items, self.out.text().strip() or None, self.gmin.value(), self.depth.value(),
                             self.copy.isChecked(), self.merge.isChecked(), text_max=self.text_max.value(),
                             template=self.current_template(), desc_max=self.desc_max.value())

    # ---------- шаблон имени ----------
    def sync_preset(self):
        tpl = self.template_edit.text().strip()
        idx = next((i for i, (_n, t) in enumerate(PRESETS) if t == tpl), len(PRESETS))
        self.preset_box.blockSignals(True); self.preset_box.setCurrentIndex(idx); self.preset_box.blockSignals(False)

    def on_preset(self, index=None, *_):
        i = self.preset_box.currentIndex() if not isinstance(index, int) else index
        if 0 <= i < len(PRESETS):
            self.template_edit.setText(PRESETS[i][1])

    def on_template_changed(self, *_):
        self.sync_preset(); self.update_template_ui(); self.replan_timer.start()

    def preview_item(self):
        r = self.table.currentRow()
        if isinstance(r, int) and 0 <= r < len(self.items):
            return self.items[r]
        if self.items:
            return self.items[0]
        return org.Item(Path("/tmp/00IoBi56ZE8.jpg"), Path("/tmp"), "00IoBi56ZE8", ["anime", "bunny"], "I'm stupid",
                        "A white rabbit stares at the camera, captioned “I'm stupid”")

    def update_template_ui(self, *_):
        tpl = self.template_edit.text().strip()
        errors, warnings = validate(tpl)
        it = self.preview_item()
        name = render_filename(DEFAULT_TEMPLATE if errors else tpl, it, it.src.suffix or ".jpg",
                               self.text_max.value(), self.desc_max.value())
        self.tpl_preview.setText("Preview: " + name)
        if not errors:
            used = {n.strip().lower() for m in re.findall(r"\{([^{}]*)\}", tpl) for n in m.split(":")[0].split("|")}
            for var, key, label in (("text", "text", "text"), ("description", "description", "description")):
                if var in used and key not in self.want() and not any(getattr(i, var) for i in self.items):
                    warnings.append(f"The template uses {{{var}}}, but {label} generation is off: it stays empty unless you fill the column by hand.")
        msg = (" · ".join(errors) + "  → the default template is used") if errors else "\n".join(warnings)
        self.tpl_warn.setText(msg); set_state(self.tpl_warn, "warn" if msg else "")

    def replan(self):
        plan = self.current_plan(); dst = {m.src: m.dst for m in plan.moves}
        self._busy = True
        for r, it in enumerate(self.items):
            d = dst.get(it.src)
            base = Path(self.out.text().strip()) if self.out.text().strip() else it.root
            txt = "(unchanged)" if d == it.src else (str(d.relative_to(base)) if d and base in d.parents else str(d))
            self.table.item(r, 5).setText(txt)
        self._busy = False
        self.apply_filter(); self.update_template_ui()
        if self.worker is None:
            n = sum(1 for m in plan.moves if m.src != m.dst)
            empty = sum(1 for it in self.items if not it.tags)
            self.info.setText(f"{len(self.items)} images · {empty} without tags (orange)"
                              + (" · analysis stopped, use Re-analyze" if self.stopped and empty else "")
                              + (f" · {self.errors} analysis errors (hover the tag cell)" if self.errors else "")
                              + f" · {n} file operations pending")

    # ---------- применение ----------
    def apply(self, *_):
        if not self.items or self.worker is not None: return
        plan = self.current_plan(); n = sum(1 for m in plan.moves if m.src != m.dst)
        if not n: QMessageBox.information(self, "MemeTagger", "Nothing to change."); return
        if QMessageBox.question(self, "Apply", f"{n} file operation(s) will be performed.\nYou can revert them with “Undo”.") != QMessageBox.StandardButton.Yes:
            return
        done, errors = org.execute(plan, user_data_dir() / "journal")
        MEMORY.commit([it.tags for it in self.items])
        self.clear(); self.info.setText(f"Done: {done} file(s)")
        if errors: QMessageBox.warning(self, "Errors", "\n".join(errors[:20]))

    def undo(self, *_):
        done, errors = org.undo_last(user_data_dir() / "journal")
        self.info.setText(f"Undone: {done} file(s)")
        if errors: QMessageBox.warning(self, "Undo", "\n".join(errors[:20]))

    def closeEvent(self, e):
        self.ps_timer.stop()
        if self.puller is not None:
            self.puller.cancel(); self.puller.wait(3000)
        if self.worker is not None:
            self.worker.cancel(); self.worker.wait(3000)
        self.save_settings()
        if self.unload_exit.isChecked():
            self.unload_sync()
        super().closeEvent(e)


def main():
    global _SYSTEM_DARK
    app = QApplication(sys.argv); app.setApplicationName("MemeTagger"); app.setStyle("Fusion")
    _SYSTEM_DARK = detect_system_dark(app)
    apply_theme(app, QSettings("memetagger", "memetagger").value("theme", "system", str))
    w = Main(); w.show()
    paths = [a for a in sys.argv[1:] if Path(a).exists()]
    if paths: w.add_paths(paths)
    sys.exit(app.exec())
