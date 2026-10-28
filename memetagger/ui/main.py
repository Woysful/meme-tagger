"""Main window: file list, preview, settings, analysis control and applying the plan."""
from __future__ import annotations
import re
import sys
import time
from pathlib import Path
from PySide6.QtCore import QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QFileDialog, QHBoxLayout, QHeaderView,
                               QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox, QProgressBar, QPushButton, QSplitter,
                               QStackedWidget, QTableView, QVBoxLayout, QWidget)
from .. import organizer as org
from ..characters import default_db
from ..engine import INSTALL_HELP, EngineConfig, TagOptions, guess_kind
from ..paths import config_file, user_config_dir, user_data_dir
from ..tags import clean_tag
from ..tagstore import TagStore
from ..vlm import ALL_WANT, OllamaVLM, normalize_host, unload_models
from .common import Settings, TaskRunner, apply_theme, card, fmt_eta, remember_system_scheme, set_state
from .library import TagLibraryDialog, ask_tag
from .model import (C_DESC, C_DEST, C_FILE, C_STATUS, C_TAGS, C_TEXT, C_THUMB, FILTERS, ROW_ROLE, ItemsModel, Row,
                    RowFilter)
from .preview import PreviewPanel
from .settings import SettingsPanel
from .widgets import TagsDelegate, ThumbDelegate
from .workers import AnalysisWorker, ImportWorker, Job, PullWorker, drop_engine

IMAGE_FILTER = "Images (*.jpg *.jpeg *.png *.webp *.gif *.bmp *.tif *.tiff *.avif *.jfif)"
REPLAN_KEYS = {"out", "copy", "levels", "min_count", "gather", "template", "text_max", "desc_max", "metadata"}


class Main(QMainWindow):
    def __init__(self, settings: Settings | None = None, store: TagStore | None = None, engine_factory=None):
        super().__init__()
        self.setWindowTitle("MemeTagger"); self.resize(1480, 900); self.setAcceptDrops(True)
        self.s = settings or Settings()
        self.store = store if store is not None else TagStore(user_data_dir() / "tag_memory.json")
        self.db = default_db()
        self.engine_factory = engine_factory
        self.tasks = TaskRunner()
        self.worker: AnalysisWorker | None = None
        self.importer: ImportWorker | None = None
        self.import_added = 0
        self._known: set = set()
        self.puller: PullWorker | None = None
        self.queue: list[Job] = []
        self.installed: dict = {}
        self.used_models: set[str] = set()
        self.ollama_online: bool | None = None
        self.prog_total = self.prog_done = 0
        self.avg_dt = self.t_last = None
        self.stopped = False
        self._words: list[str] | None = None
        self._library: tuple[str, list] | None = None
        self.plan: org.Plan | None = None

        self.model = ItemsModel(self.kind_of, self)
        self.proxy = RowFilter(self); self.proxy.setSourceModel(self.model)
        self.replan_timer = QTimer(self, singleShot=True, interval=250, timeout=self.replan)
        self.ps_timer = QTimer(self, interval=4000, timeout=self.poll_ps)

        root = QWidget(); outer = QVBoxLayout(root); outer.setContentsMargins(18, 14, 18, 12); outer.setSpacing(10)
        outer.addLayout(self._build_header())
        outer.addLayout(self._build_toolbar())
        self.split = QSplitter(Qt.Orientation.Horizontal); self.split.setChildrenCollapsible(False)
        self.split.addWidget(self._build_table())
        self.preview = PreviewPanel(self.tag_words, self.kind_of)
        self.preview.changed.connect(self.on_preview_changed)
        self.preview.add_to_selection.connect(self.add_tags_to_selection)
        pv = card(self.preview); pv.layout().setContentsMargins(10, 10, 4, 10); pv.setMinimumWidth(300)
        self.preview_card = pv
        self.split.addWidget(pv)
        self.panel = SettingsPanel(self.s, self.preview_item, self.want)
        st = card(self.panel); st.layout().setContentsMargins(10, 10, 4, 6); st.setMinimumWidth(380)
        self.settings_card = st
        self.split.addWidget(st)
        self.split.setStretchFactor(0, 5); self.split.setStretchFactor(1, 2); self.split.setStretchFactor(2, 2)
        self.split.setSizes([720, 340, 420])
        outer.addWidget(self.split, 1)
        outer.addLayout(self._build_statusbar())
        self.setCentralWidget(root)
        self._wire_panel()
        self.model.edited.connect(self.on_table_edit)
        self.restore_window()
        self.update_headers(); self.update_controls()
        QShortcut(QKeySequence.StandardKey.Find, self, activated=lambda: self.filter_edit.setFocus())
        QShortcut(QKeySequence.StandardKey.Delete, self.table, activated=self.remove_selected)
        QTimer.singleShot(300, self.refresh_models)
        QTimer.singleShot(600, self.poll_ps)

    # ---------- tag helpers ----------
    def kind_of(self, tag: str) -> str | None:
        return guess_kind(self.store, self.db, tag)

    def tag_words(self) -> list[str]:
        """Completion list: the user's tags first (most used first), then known franchises and characters."""
        if self._words is None:
            seen, words = set(), []
            for t, _k, _n in self.store.all_tags():
                seen.add(t); words.append(t)
            for r in self.model.rows:
                for t in r.item.tags:
                    if t not in seen:
                        seen.add(t); words.append(t)
            extra = sorted(((max(p for _f, p in opts), t) for t, opts in self.db.by_tag.items() if t not in seen), reverse=True)
            words += sorted(f for f in self.db.franchises if f not in seen)
            words += [t for _p, t in extra]
            self._words = words
        return self._words

    def banned(self) -> set[str]:
        return {t for t in map(clean_tag, re.split(r"[,\s]+", self.s.get("banned"))) if t}

    def want(self) -> frozenset:
        w = {k for k, key in (("tags", "gen_tags"), ("text", "gen_text"), ("description", "gen_desc")) if self.s.get(key)}
        return frozenset(w) or ALL_WANT

    # ---------- layout ----------
    def _button(self, text, slot, name="", tip=""):
        b = QPushButton(text); b.clicked.connect(slot)
        if name:
            b.setObjectName(name)
        if tip:
            b.setToolTip(tip)
        return b

    def _build_header(self):
        row = QHBoxLayout()
        title = QLabel("MemeTagger"); title.setObjectName("title")
        sub = QLabel("Local AI tagging and sorting for your meme library"); sub.setObjectName("muted")
        col = QVBoxLayout(); col.setSpacing(0); col.addWidget(title); col.addWidget(sub)
        self.badge = QLabel("● checking Ollama…"); self.badge.setObjectName("badge")
        self.theme_box = QComboBox()
        for label, key in (("Theme: system", "system"), ("Theme: dark", "dark"), ("Theme: light", "light")):
            self.theme_box.addItem(label, key)
        self.theme_box.setCurrentIndex(max(0, self.theme_box.findData(self.s.get("theme"))))
        self.theme_box.currentIndexChanged.connect(self.on_theme)
        row.addLayout(col); row.addStretch(1); row.addWidget(self.badge); row.addWidget(self.theme_box)
        return row

    def _build_toolbar(self):
        row = QHBoxLayout(); row.setSpacing(8)
        self.add_files_btn = self._button("＋ Add files", self.pick_files)
        self.add_dir_btn = self._button("Add folder", self.pick_dir)
        self.pause_btn = self._button("⏸ Pause", self.toggle_pause)
        self.stop_btn = self._button("■ Stop", self.stop_analysis, "danger")
        self.again_btn = self._button("↻ Analyze again", self.reanalyze, tip="Selected files, or all when nothing is selected")
        self.clear_btn = self._button("Clear list", self.clear, "flat")
        self.filter_edit = QLineEdit(); self.filter_edit.setPlaceholderText("Search name, tag or text…  (Ctrl+F)")
        self.filter_edit.setClearButtonEnabled(True); self.filter_edit.setMinimumWidth(220)
        self.filter_edit.textChanged.connect(lambda q: self.proxy.set(query=q))
        self.filter_mode = QComboBox()
        for label, key in FILTERS:
            self.filter_mode.addItem(label, key)
        self.filter_mode.currentIndexChanged.connect(lambda _i: self.proxy.set(mode=self.filter_mode.currentData()))
        self.library_btn = self._button("Tag library", self.open_library, tip="Rename, merge and clean up tags")
        self.settings_btn = self._button("⚙ Settings", self.toggle_settings, "toggle")
        self.settings_btn.setCheckable(True)
        self.undo_btn = self._button("↩ Undo", self.undo, tip="Revert the last Apply")
        self.apply_btn = self._button("Apply", self.apply, "primary", "Rename and move the files as shown in “New location”")
        for w in (self.add_files_btn, self.add_dir_btn, self.pause_btn, self.stop_btn, self.again_btn, self.clear_btn):
            row.addWidget(w)
        row.addStretch(1)
        for w in (self.filter_edit, self.filter_mode, self.library_btn, self.settings_btn, self.undo_btn, self.apply_btn):
            row.addWidget(w)
        return row

    def _build_table(self):
        self.stack = QStackedWidget()
        drop = QLabel("Drop images or folders here\n\nor use “Add files” / “Add folder”")
        drop.setObjectName("drop"); drop.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_label, self.drop_text = drop, drop.text()
        holder = QWidget(); hl = QVBoxLayout(holder); hl.setContentsMargins(40, 40, 40, 40); hl.addWidget(drop)
        t = self.table = QTableView()
        t.setModel(self.proxy)
        t.setItemDelegateForColumn(C_TAGS, TagsDelegate(self.tag_words, t))
        t.setItemDelegateForColumn(C_THUMB, ThumbDelegate(t))
        t.setIconSize(QSize(72, 72)); t.verticalHeader().setDefaultSectionSize(84); t.verticalHeader().hide()
        t.setShowGrid(False); t.setAlternatingRowColors(True); t.setWordWrap(True)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed)
        t.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        t.setSortingEnabled(True); t.sortByColumn(-1, Qt.SortOrder.AscendingOrder)
        h = t.horizontalHeader(); h.setSectionResizeMode(QHeaderView.ResizeMode.Interactive); h.setStretchLastSection(True)
        h.setHighlightSections(False)
        for col, w in ((C_THUMB, 88), (C_STATUS, 104), (C_FILE, 160), (C_TAGS, 240), (C_TEXT, 160), (C_DESC, 220), (C_DEST, 220)):
            t.setColumnWidth(col, w)
        t.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        t.customContextMenuRequested.connect(self.table_menu)
        t.selectionModel().selectionChanged.connect(self.on_selection)
        t.selectionModel().currentRowChanged.connect(self.on_selection)
        c = card(t); c.layout().setContentsMargins(4, 4, 4, 4)
        self.stack.addWidget(holder); self.stack.addWidget(c)
        return self.stack

    def _build_statusbar(self):
        row = QHBoxLayout(); row.setSpacing(12)
        self.info = QLabel("Ready"); self.info.setObjectName("muted")
        self.eta_label = QLabel(""); self.eta_label.setObjectName("muted")
        self.progress = QProgressBar(); self.progress.setTextVisible(False); self.progress.setRange(0, 1); self.progress.setValue(0)
        self.progress.setMinimumWidth(240)
        row.addWidget(self.info, 1); row.addWidget(self.eta_label); row.addWidget(self.progress)
        return row

    def _wire_panel(self):
        p = self.panel
        p.changed.connect(self.on_setting)
        p.refresh_btn.clicked.connect(self.refresh_models)
        p.model.currentTextChanged.connect(self.update_model_hint)
        p.test_btn.clicked.connect(self.test_ollama)
        p.pull_btn.clicked.connect(self.toggle_pull)
        p.unload_btn.clicked.connect(self.unload_model)
        p.out_btn.clicked.connect(self.pick_out)
        p.wd_custom_btn.clicked.connect(self.pick_wd_folder)
        p.library_btn.clicked.connect(self.open_library)
        for name, btn in p.config_btns.items():
            btn.clicked.connect(lambda _=False, n=name: self.open_path(config_file(n)))
        p.open_config_btn.clicked.connect(lambda: self.open_path(user_config_dir()))
        p.open_data_btn.clicked.connect(lambda: self.open_path(user_data_dir()))
        p.reload_btn.clicked.connect(self.reload_config)

    # ---------- window state ----------
    def restore_window(self):
        try:
            if g := self.s.raw("geometry"):
                self.restoreGeometry(g)
            if sizes := self.s.raw("splitter"):
                self.split.setSizes([int(x) for x in sizes])
        except Exception:
            pass
        on = self.s.get("show_settings")
        self.settings_card.setVisible(on); self.settings_btn.setChecked(on)

    def save_window(self):
        self.s.set("geometry", self.saveGeometry())
        self.s.set("splitter", list(self.split.sizes()))
        self.s.sync()

    def toggle_settings(self, *_):
        on = not self.settings_card.isVisible()
        self.settings_card.setVisible(on); self.settings_btn.setChecked(on); self.s.set("show_settings", on)

    def on_theme(self, *_):
        mode = self.theme_box.currentData()
        self.s.set("theme", mode)
        apply_theme(QApplication.instance(), mode)
        self.model.refresh(); self.show_current()

    # ---------- settings reactions ----------
    def on_setting(self, key: str):
        if key in ("out",):
            self._library = None
        if key in REPLAN_KEYS:
            self.replan_timer.start()
        if key in ("gen_tags", "gen_text", "gen_desc"):
            if not any(self.s.get(k) for k in ("gen_tags", "gen_text", "gen_desc")):
                self.panel.gen_tags.setChecked(True)            # something must be generated
                self.info.setText("At least one of tags / text / description stays on")
            self.update_headers(); self.panel.update_template()
        if key == "banned":
            self.apply_banned()
        if key == "engine":
            self.update_classic_visibility()
        if key == "host":
            self.refresh_models()

    def update_headers(self):
        w = self.want()
        self.model.disabled_cols = {c for c, k in ((C_TAGS, "tags"), (C_TEXT, "text"), (C_DESC, "description")) if k not in w}
        self.model.headerDataChanged.emit(Qt.Orientation.Horizontal, 0, C_DEST)

    def update_classic_visibility(self):
        mode = self.s.get("engine")
        self.panel.set_classic_visible(mode == "classic" or (mode == "auto" and self.ollama_online is False))

    def engine_config(self) -> EngineConfig:
        return EngineConfig(mode=self.s.get("engine"), host=self.s.get("host"), model=self.s.get("model"),
                            ocr=self.s.get("ocr"), wd=self.s.get("wd"), wd_model=self.panel.wd_spec(),
                            wd_device=self.s.get("wd_device"), want_tags="tags" in self.want())

    def tag_options(self) -> TagOptions:
        limits = {"desc_chars": self.s.get("desc_max"), "text_chars": self.s.get("text_max")} if self.s.get("limit_model") else None
        return TagOptions(want=self.want(), max_tags=self.s.get("max_tags"), banned=frozenset(self.banned()),
                          desc_lang=self.s.get("desc_lang"), limits=limits, strictness=self.s.get("strictness"),
                          verify=self.s.get("verify"), show_vocab=self.s.get("show_vocab"),
                          folder_tags_all=self.s.get("folder_tags_all"), concept_thr=self.s.get("cthr"),
                          general_thr=self.s.get("gthr"), fallback=self.s.get("fallback"))

    def reload_config(self, *_):
        if self.worker is not None:
            QMessageBox.information(self, "MemeTagger", "Stop the analysis first (or wait until it finishes)."); return
        drop_engine(); self._words = None
        self.info.setText("Configuration will be reloaded on the next analysis")

    def open_path(self, path):
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
            QMessageBox.information(self, "Open", f"Could not open it automatically. Path:\n{path}")

    # ---------- Ollama ----------
    def host(self) -> str:
        return self.s.get("host")

    def refresh_models(self, *_):
        host = self.host()
        self.tasks.run(lambda: OllamaVLM(host, "").list_models(), self.on_models)

    def on_models(self, models, err):
        p = self.panel
        if err:
            self.installed, self.ollama_online = {}, False
            self.badge.setText("● Ollama offline"); set_state(self.badge, "bad")
            p.model_hint.setText(f"Ollama is not reachable at {normalize_host(self.host())}.\n" + INSTALL_HELP.splitlines()[0])
            p.model_hint.setToolTip(str(err)); set_state(p.model_hint, "warn")
            self.update_classic_visibility()
            return
        self.ollama_online = True
        self.installed = {m["name"]: m for m in models}
        usable = [m["name"] for m in models if m["vision"] is not False]
        hidden = len(models) - len(usable)
        current = p.model.currentText().strip()
        if not current and usable:
            current = next((n for n in usable if "qwen3-vl" in n), usable[0])
        p.model.blockSignals(True)
        p.model.clear(); p.model.addItems(usable); p.model.setCurrentText(current or Settings.DEFAULTS["model"])
        p.model.blockSignals(False)
        self.badge.setText(f"● Ollama online · {len(usable)} vision model(s)" + (f" · {hidden} text-only hidden" if hidden else ""))
        set_state(self.badge, "ok")
        self.update_model_hint(); self.update_classic_visibility()

    def update_model_hint(self, *_):
        p = self.panel
        name = p.model.currentText().strip()
        if not self.installed:
            return
        m = self.installed.get(name) or self.installed.get(name + ":latest")
        if m:
            extra = " · text-only model, it cannot see images" if m["vision"] is False else ""
            p.model_hint.setText(f"Installed · {m['size'] / 1e9:.1f} GB{extra}"); set_state(p.model_hint, "warn" if extra else "")
        else:
            p.model_hint.setText(f"Not installed. Use “Download model” or run:  ollama pull {name}" if name else "Pick or type a model name")
            set_state(p.model_hint, "warn")

    def poll_ps(self, *_):
        host = self.host()
        self.tasks.run(lambda: OllamaVLM(host, "").loaded(), self.on_ps)

    def on_ps(self, loaded, err):
        lab = self.panel.gpu_label
        if err:
            return
        if not loaded:
            lab.setText("GPU: no model loaded right now"); set_state(lab, ""); return
        parts, partial = [], False
        for m in loaded:
            pct = int(100 * m["vram"] / m["size"]) if m["size"] else 0
            partial = partial or pct < 100
            parts.append(f"{m['name']} · {m['size'] / 1e9:.1f} GB · {pct}% in VRAM")
        lab.setText("GPU: " + "; ".join(parts) + ("\n⚠ partly on the CPU: expect slow speed (try a smaller model)" if partial else ""))
        set_state(lab, "warn" if partial else "")

    def test_ollama(self, *_):
        host, model = self.host(), self.s.get("model")

        def done(res, err):
            ok, msg = res if res else (False, err)
            (QMessageBox.information if ok else QMessageBox.warning)(self, "Ollama", msg if ok else f"{msg}\n\n{INSTALL_HELP}")
        self.tasks.run(lambda: OllamaVLM(host, model).status(), done)
        self.refresh_models()

    def unload_model(self, *_):
        names, host = set(self.used_models) | {self.s.get("model")}, self.host()

        def done(res, err):
            self.info.setText(f"Unload failed: {err}" if err else
                              (f"Unloaded from memory: {', '.join(res)}" if res else "The model was not loaded"))
            self.poll_ps()
        self.tasks.run(lambda: unload_models(host, names), done)

    def toggle_pull(self, *_):
        p = self.panel
        if self.puller is not None:
            self.puller.cancel(); p.pull_label.setText("Cancelling…"); return
        name = p.model.currentText().strip()
        if not name or any(c.isspace() for c in name):
            QMessageBox.information(self, "Download model", "Type the model name first, e.g. qwen3-vl:8b"); return
        again = name in self.installed or name + ":latest" in self.installed
        q = f"{'Update' if again else 'Download'} “{name}” from the Ollama library?\nIt can take several GB and a few minutes."
        if QMessageBox.question(self, "Download model", q) != QMessageBox.StandardButton.Yes:
            return
        self.puller = PullWorker(self.host(), name)
        self.puller.progress.connect(self.on_pull_progress)
        self.puller.done.connect(lambda n: (QMessageBox.information(self, "Download model", f"“{n}” is installed."),
                                            self.refresh_models(), p.model.setCurrentText(n)))
        self.puller.failed.connect(lambda msg: QMessageBox.warning(self, "Download failed", msg))
        self.puller.finished.connect(self.on_pull_finished)
        p.pull_label.setText(f"Starting download of {name}…"); p.pull_bar.setRange(0, 0)
        p.pull_label.setVisible(True); p.pull_bar.setVisible(True); p.pull_btn.setText("✕ Cancel")
        self.puller.start()

    def on_pull_progress(self, status, completed, total):
        p = self.panel
        if total:
            p.pull_bar.setRange(0, 100); p.pull_bar.setValue(int(100 * (completed or 0) / total))
            p.pull_label.setText(f"{status[:36]} · {(completed or 0) / 1e9:.2f} / {total / 1e9:.2f} GB")
        else:
            p.pull_bar.setRange(0, 0); p.pull_label.setText(status)

    def on_pull_finished(self, *_):
        w, self.puller = self.puller, None
        if w is not None:
            w.wait()
        p = self.panel
        p.pull_label.setVisible(False); p.pull_bar.setVisible(False); p.pull_btn.setText("⬇ Download")

    # ---------- adding files ----------
    def pick_files(self, *_):
        fs, _ = QFileDialog.getOpenFileNames(self, "Images", "", IMAGE_FILTER)
        self.add_paths(fs)

    def pick_dir(self, *_):
        d = QFileDialog.getExistingDirectory(self, "Folder with images")
        if d:
            self.add_paths([d])

    def pick_out(self, *_):
        d = QFileDialog.getExistingDirectory(self, "Library folder", self.s.get("out"))
        if d:
            self.panel.out.setText(d)

    def pick_wd_folder(self, *_):
        d = QFileDialog.getExistingDirectory(self, "Folder with model.onnx and selected_tags.csv", self.s.get("wd_custom"))
        if d:
            self.panel.wd_custom.setText(d); self.s.set("wd_custom", d)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        self.add_paths([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])

    def add_paths(self, paths):
        """Folders are scanned in the background: the window stays responsive and rows appear in batches."""
        paths = [str(p) for p in paths if p]
        if not paths:
            return
        if self.importer is not None:
            self.importer.add(paths); return
        if self.worker is None and not self.queue:
            self.prog_total = self.prog_done = 0
        self.stopped = False
        self.import_added = 0
        self._known = {r.item.src for r in self.model.rows}
        self._start_import(paths)

    def _start_import(self, paths):
        w = self.importer = ImportWorker(paths, read_meta=self.s.get("metadata") != "off")
        w.batch.connect(self.add_items)
        w.progress.connect(self.on_import_progress)
        w.finished.connect(self.on_import_finished)
        w.start()
        self.update_controls(); self.update_progress()

    def add_items(self, items):
        w = self.sender()
        if w is not self.importer or w.isInterruptionRequested():     # in flight from an import that was cleared or stopped
            return
        new = [it for it in items if it.src not in self._known]
        if not new:
            return
        self._known.update(it.src for it in new)
        self.import_added += len(new)
        ban, skip = self.banned(), self.s.get("skip") and "tags" in self.want()
        rows, jobs = [], []
        for it in new:
            it.tags = [t for t in it.tags if t not in ban]
            analyze = not (skip and it.tags)
            row = Row(it, "queued" if analyze else "skipped")
            rows.append(row); jobs.append(Job(row.id, it.src, analyze, org.item_context(it)))
        self.model.add(rows)
        self._words = None
        self.enqueue(jobs)
        if not self.table.currentIndex().isValid():
            self.table.selectRow(0)
        self.update_controls()
        self.replan_timer.start()

    def on_import_progress(self, found, folder):
        if self.importer is not None:
            self.update_progress()

    def on_import_finished(self):
        w = self.sender()
        if w is None or w is not self.importer:          # cleared: clear() has already waited for it
            return
        self.importer = None
        w.wait()
        cancelled = w.isInterruptionRequested()
        rest = [] if cancelled else w.leftover()
        if rest:                                         # added right when the scan was finishing
            self._start_import(rest); return
        self.update_controls(); self.update_progress()
        self.replan_timer.start()
        if self.worker is None:
            if self.import_added or cancelled:
                self.show_summary()
            else:
                self.info.setText("No new images found")
    # ---------- analysis ----------
    def enqueue(self, jobs: list[Job]):
        self.prog_total += sum(j.analyze for j in jobs)
        if self.worker is not None and self.worker.add(jobs):
            self.update_progress(); return
        self.queue += jobs
        self.start_worker()

    def start_worker(self):
        if self.worker is not None or not self.queue:
            return
        jobs, self.queue = self.queue, []
        if any(j.analyze for j in jobs) and self.s.get("engine") != "classic":
            self.used_models.add(self.s.get("model"))
        kw = {"engine_factory": self.engine_factory} if self.engine_factory else {}
        self.worker = AnalysisWorker(self.engine_config(), self.tag_options(), self.store, jobs, **kw)
        self.worker.row_started.connect(self.on_row_started)
        self.worker.row_done.connect(self.on_row_done)
        self.worker.row_error.connect(self.on_row_error)
        self.worker.status.connect(self.info.setText)
        self.worker.failed.connect(self.on_engine_failed)
        self.worker.finished.connect(self.on_worker_finished)
        self.t_last, self.avg_dt = time.time(), None
        self.worker.start()
        if self.worker.needs_engine:
            self.ps_timer.start()
        self.update_controls()

    def on_worker_finished(self):
        w, self.worker = self.worker, None
        if w is not None:
            w.wait()
            waiting = {j.rid for j in self.queue}         # stopped: the rest goes back to "not analyzed"
            for r in self.model.rows:
                if r.status in ("queued", "analyzing") and r.id not in waiting:
                    r.status = "idle"; self.model.refresh(r.id)
        self.ps_timer.stop(); self.poll_ps(); self.eta_label.setText("")
        self.start_worker(); self.update_controls(); self.update_progress()
        self.replan_timer.start()
        if self.worker is None and self.importer is None:
            self.show_summary()

    def on_row_started(self, rid):
        if (r := self.model.row(rid)) is not None and r.status == "queued":
            r.status = "analyzing"; self.model.refresh(rid, (C_STATUS,))

    def _set_thumb(self, row: Row, thumb):
        if not thumb.isNull():
            row.thumb = QPixmap.fromImage(thumb)

    def on_row_done(self, rid, res, thumb):
        row = self.model.row(rid)
        if row is None:
            return
        self._set_thumb(row, thumb)
        it = row.item
        if res is not None:
            if "tags" in res.want:
                it.tags = res.names
                it.kinds = {**it.kinds, **res.kinds}
            if "text" in res.want:
                it.text = res.caption
            if "description" in res.want:
                it.description = res.description
            it.touched = True
            row.status, row.error = "done", ""
            row.source, row.notes, row.suggestions, row.ocr = res.source, res.notes, res.suggestions, res.ocr_text
            self.tick()
        elif row.status == "queued":
            row.status = "idle"
        self.model.refresh(rid)
        if res is not None and "tags" in res.want:
            self._words = None
        if self.current_row() is row:
            self.show_current()
        self.replan_timer.start()

    def on_row_error(self, rid, msg, thumb):
        row = self.model.row(rid)
        if row is None:
            return
        self._set_thumb(row, thumb)
        row.status, row.error = "error", msg
        self.tick()
        self.model.refresh(rid)
        if self.current_row() is row:
            self.show_current()

    def tick(self):
        self.prog_done += 1
        now = time.time()
        if self.t_last is not None:
            dt = now - self.t_last
            self.avg_dt = dt if self.avg_dt is None else 0.8 * self.avg_dt + 0.2 * dt
        self.t_last = now
        self.update_progress()

    def on_engine_failed(self, msg):
        self.prog_total = self.prog_done
        QMessageBox.critical(self, "Could not start the analysis", msg)
        self.info.setText("The analysis engine failed to start")

    def toggle_pause(self, *_):
        if self.worker is None:
            return
        if self.worker.is_paused():
            self.worker.resume(); self.t_last = time.time()
        else:
            self.worker.pause()
        self.update_controls(); self.update_progress()

    def stop_analysis(self, *_):
        if self.worker is None and not self.queue and self.importer is None:
            return
        if self.importer is not None:
            self.importer.cancel()
        for j in self.queue:
            if (r := self.model.row(j.rid)) is not None and r.status == "queued":
                r.status = "idle"
        self.queue.clear(); self.stopped = True
        if self.worker is not None:
            self.worker.cancel()
        self.prog_total = self.prog_done
        self.model.refresh(); self.update_controls()

    def reanalyze(self, *_):
        rows = self.selected_rows() or self.model.rows
        rows = [r for r in rows if r.status not in ("queued", "analyzing")]
        if not rows:
            return
        if self.worker is None and not self.queue:
            self.prog_total = self.prog_done = 0
        self.stopped = False
        for r in rows:
            r.status, r.error = "queued", ""
        self.model.refresh()
        self.enqueue([Job(r.id, r.item.src, True, org.item_context(r.item)) for r in rows])
        self.update_controls()

    def update_controls(self):
        running = self.worker is not None
        paused = running and self.worker.is_paused()
        busy = running or bool(self.queue) or self.importer is not None
        self.pause_btn.setText("▶ Resume" if paused else "⏸ Pause")
        self.pause_btn.setEnabled(running)
        self.stop_btn.setEnabled(busy)
        self.again_btn.setEnabled(bool(self.model.rows))
        self.apply_btn.setEnabled(bool(self.model.rows) and not busy)
        self.clear_btn.setEnabled(bool(self.model.rows))
        self.panel.unload_btn.setEnabled(not running)
        self.stack.setCurrentIndex(1 if self.model.rows else 0)
        self.drop_label.setText("Looking for images…" if self.importer is not None else self.drop_text)
        self.preview_card.setVisible(bool(self.model.rows))

    def update_progress(self):
        importing = f"Importing: {self.importer.found} files found…" if self.importer is not None else ""
        total = max(self.prog_total, 1); done = min(self.prog_done, total)
        if importing and (self.worker is None or not self.prog_total):
            self.progress.setRange(0, 0); self.info.setText(importing); return     # busy indicator
        self.progress.setRange(0, total); self.progress.setValue(done)
        if self.worker is not None:
            paused = self.worker.is_paused()
            self.info.setText(f"{'Paused' if paused else 'Analyzing'}: {done} of {self.prog_total}"
                              + (f" · {importing}" if importing else ""))
            left = self.prog_total - self.prog_done
            self.eta_label.setText("" if paused or self.avg_dt is None or left <= 0 else f"about {fmt_eta(self.avg_dt * left)} left")

    def show_summary(self):
        rows = self.model.rows
        if not rows:
            self.info.setText("Ready"); return
        empty = sum(1 for r in rows if not r.item.tags)
        errors = sum(1 for r in rows if r.status == "error")
        ops = len(self.plan.operations) if self.plan else 0
        bits = [f"{len(rows)} files", f"{ops} changes pending"]
        if empty:
            bits.append(f"{empty} without tags")
        if errors:
            bits.append(f"{errors} errors (filter: Errors)")
        if self.stopped and any(r.status == "idle" for r in rows):
            bits.append("analysis stopped, use Analyze again")
        self.info.setText(" · ".join(bits))

    # ---------- table & selection ----------
    def selected_rows(self) -> list[Row]:
        idx = self.table.selectionModel().selectedRows()
        return [self.proxy.data(i, ROW_ROLE) for i in idx]

    def current_row(self) -> Row | None:
        i = self.table.currentIndex()
        return self.proxy.data(i, ROW_ROLE) if i.isValid() else None

    def preview_item(self):
        r = self.current_row() or (self.model.rows[0] if self.model.rows else None)
        if r is not None:
            return r.item
        return org.Item(Path("/tmp/00IoBi56ZE8.jpg"), Path("/tmp"), "00IoBi56ZE8", ["k-on", "hirasawa_yui", "guitar"],
                        "When the solo hits", "Yui plays the guitar on stage")

    def on_selection(self, *_):
        self.show_current()
        self.panel.update_template()

    def show_current(self):
        n = len(self.table.selectionModel().selectedRows()) if self.table.selectionModel() else 0
        self.preview.show_row(self.current_row(), max(n, 1))

    def on_preview_changed(self, rid, what):
        self.model.refresh(rid)
        if what == "tags":
            self._words = None
        self.replan_timer.start()

    def on_table_edit(self, rid):
        if self.current_row() is not None and self.current_row().id == rid:
            self.show_current()
        self.replan_timer.start()

    def add_tags_to_selection(self, tags):
        for r in self.selected_rows():
            new = [t for t in tags if t not in r.item.tags]
            if new:
                r.item.tags += new; r.item.touched = True
                if r.status not in ("queued", "analyzing"):
                    r.status = "edited"
        self.model.refresh(); self.show_current(); self.replan_timer.start()

    def remove_tags_from_selection(self):
        rows = self.selected_rows()
        tag = ask_tag(self, "Remove tag", f"Tag to remove from {len(rows)} selected files:", "", self.tag_words)
        if not tag:
            return
        for r in rows:
            if tag in r.item.tags:
                r.item.tags = [t for t in r.item.tags if t != tag]; r.item.touched = True
                if r.status not in ("queued", "analyzing"):
                    r.status = "edited"
        self.model.refresh(); self.show_current(); self.replan_timer.start()

    def remove_selected(self):
        rows = self.selected_rows()
        if not rows:
            return
        ids = {r.id for r in rows}
        if self.worker is not None:
            self.worker.drop(ids)
        self.queue = [j for j in self.queue if j.rid not in ids]
        self.model.remove(ids)
        self.update_controls(); self.show_current(); self.replan_timer.start()

    def table_menu(self, pos):
        rows = self.selected_rows()
        if not rows:
            return
        m = QMenu(self)

        def act(text, fn, enabled=True):
            a = QAction(text, m); a.triggered.connect(lambda *_: fn()); a.setEnabled(enabled); m.addAction(a)
        n = len(rows)
        act(f"Analyze again ({n})" if n > 1 else "Analyze again", self.reanalyze)
        act("Add tags…", lambda: self.add_tags_dialog(rows))
        act("Remove a tag…", self.remove_tags_from_selection)
        m.addSeparator()
        act("Open file", lambda: self.open_path(rows[0].item.src), n == 1)
        act("Show in folder", lambda: self.open_path(rows[0].item.src.parent), n == 1)
        m.addSeparator()
        act(f"Remove from list ({n})" if n > 1 else "Remove from list", self.remove_selected)
        m.exec(self.table.viewport().mapToGlobal(pos))

    def add_tags_dialog(self, rows):
        tag = ask_tag(self, "Add tag", f"Tag to add to {len(rows)} file(s):", "", self.tag_words)
        if tag:
            self.add_tags_to_selection([tag])

    def apply_banned(self):
        ban = self.banned()
        for r in self.model.rows:
            kept = [t for t in r.item.tags if t not in ban]
            if kept != r.item.tags:
                r.item.tags = kept
        self.model.refresh(); self.show_current(); self.replan_timer.start()

    def clear(self, *_):
        if self.importer is not None:
            w, self.importer = self.importer, None
            w.cancel(); w.wait()
        self.queue.clear()
        if self.worker is not None:
            self.worker.cancel(); self.worker.wait()
            QApplication.processEvents()
            self.worker = None
        self.model.clear(); self.plan = None
        self.stopped = False; self.prog_total = self.prog_done = 0
        self.info.setText("Ready"); self.eta_label.setText(""); self.progress.setRange(0, 1); self.progress.setValue(0)
        self.show_current(); self.update_controls()

    # ---------- plan ----------
    def out_dir(self) -> Path | None:
        o = self.s.get("out").strip()
        return Path(o).expanduser() if o else None

    def library_items(self) -> list:
        out = self.out_dir()
        if out is None or not out.is_dir():
            return []
        if self._library is None or self._library[0] != str(out):
            self._library = (str(out), org.collect([out]))
        return self._library[1]

    def library_dirs(self) -> list[Path]:
        out = self.out_dir()
        if out is not None:
            return [out] if out.is_dir() else []
        return sorted({r.item.root for r in self.model.rows})

    def make_plan(self) -> org.Plan:
        p = self.panel
        return org.make_plan(self.model.items(), out_dir=self.out_dir(), levels=p.level_spec(), min_count=self.s.get("min_count"),
                             copy=self.s.get("copy"), template=p.template_text(), text_max=self.s.get("text_max"),
                             desc_max=self.s.get("desc_max"), library=self.library_items(), gather=self.s.get("gather"),
                             kind_of=self.kind_of, metadata_mode=self.s.get("metadata"))

    def replan(self):
        if not self.model.rows:
            self.plan = None; return
        self.plan = self.make_plan()
        dst = {m.src: m.dst for m in self.plan.moves}
        out = self.out_dir()
        for r in self.model.rows:
            d = dst.get(r.item.src)
            base = out or r.item.root
            r.dest_changed = d is not None and d != r.item.src
            if d is None or d == r.item.src:
                r.dest = "stays as is"
            else:
                r.dest = str(d.relative_to(base)) if base in d.parents else str(d)
        self.model.refresh_column(C_DEST)
        cur = self.current_row()
        if cur is not None:
            self.preview.dest.setText(cur.dest)
        self.panel.update_template()
        if self.worker is None and self.importer is None:
            self.show_summary()

    # ---------- apply / undo ----------
    def apply(self, *_):
        if not self.model.rows or self.worker is not None or self.importer is not None:
            return
        plan = self.make_plan()
        ops = plan.operations
        if not ops:
            QMessageBox.information(self, "MemeTagger", "Nothing to change."); return
        session = {r.item.src for r in self.model.rows}
        moves = [m for m in ops if m.moves]
        older = sum(1 for m in moves if m.src not in session)
        folders = {m.dst.parent for m in moves if not m.dst.parent.exists()}
        meta = sum(1 for m in ops if m.meta is not None)
        lines = [f"{len(moves) - older} file(s) will be {'copied' if self.s.get('copy') else 'renamed or moved'}"]
        if older:
            lines.append(f"{older} older file(s) will join new folders")
        if folders:
            lines.append(f"{len(folders)} new folder(s) will be created")
        if meta:
            lines.append(f"tags will be written into {meta} file(s)")
        if QMessageBox.question(self, "Apply", "\n".join(lines) + "\n\nYou can revert this with “Undo”.") \
                != QMessageBox.StandardButton.Yes:
            return
        self.apply_btn.setEnabled(False); self.info.setText("Applying…")
        rows = list(self.model.rows)

        def done(res, err):
            if err:
                QMessageBox.warning(self, "Apply", err); self.update_controls(); return
            n, errors = res
            self.after_apply(plan, rows)
            self.info.setText(f"Done: {n} file(s)" + (f", {len(errors)} errors" if errors else ""))
            if errors:
                QMessageBox.warning(self, "Some files failed", "\n".join(errors[:20]))
        self.tasks.run(lambda: org.execute(plan, user_data_dir() / "journal"), done)

    def after_apply(self, plan: org.Plan, rows: list[Row]):
        self.store.commit([r.item.tags for r in rows if r.item.touched or r.item.changed])
        dst = {m.src: m.dst for m in plan.moves if m.moves and not m.copy}
        out = self.out_dir()
        for r in rows:
            it = r.item
            if it.src in dst:
                it.src = dst[it.src]
                if out is not None:
                    it.root = out
            it.base = org.split_name(it.src.stem)[0]
            it.orig_tags = list(it.tags); it.touched = False
            if r.status in ("done", "edited", "skipped", "idle"):
                r.status = "applied"
        self._library = None; self._words = None
        self.model.refresh(); self.show_current(); self.update_controls(); self.replan()

    def undo(self, *_):
        if self.worker is not None:
            return
        if self.model.rows and QMessageBox.question(
                self, "Undo", "Undo the last Apply? The file list will be cleared, because files go back to their "
                              "old places.") != QMessageBox.StandardButton.Yes:
            return
        n, errors = org.undo_last(user_data_dir() / "journal")
        self.clear(); self._library = None
        self.info.setText(f"Undone: {n} file(s)")
        if errors:
            QMessageBox.warning(self, "Undo", "\n".join(errors[:20]))

    # ---------- tag library ----------
    def open_library(self, *_):
        TagLibraryDialog(self).exec()

    def apply_tag_mapping(self, mapping: dict, in_files: bool, confirm: bool = True) -> bool:
        """Applies a rename/merge/delete to the list and (optionally) to files in the library folders."""
        session = {r.item.src for r in self.model.rows}
        plan = None
        if in_files and (dirs := self.library_dirs()):
            items = [it for it in org.collect(dirs) if it.src not in session]
            plan = org.plan_tag_edit(items, mapping, self.kind_of, self.s.get("metadata"))
            n = len(plan.operations)
            if n and confirm and QMessageBox.question(
                    self, "Tag library", f"{n} file(s) in the library will be renamed.\nYou can revert it with “Undo”.") \
                    != QMessageBox.StandardButton.Yes:
                return False
        for r in self.model.rows:
            if any(t in mapping for t in r.item.tags):
                new = []
                for t in r.item.tags:
                    t = mapping.get(t, t)
                    if t and t not in new:
                        new.append(t)
                r.item.tags, r.item.touched = new, True
                if r.status not in ("queued", "analyzing"):
                    r.status = "edited"
        if plan is not None and plan.operations:
            n, errors = org.execute(plan, user_data_dir() / "journal")
            self.info.setText(f"Tag library: {n} file(s) updated")
            if errors:
                QMessageBox.warning(self, "Some files failed", "\n".join(errors[:20]))
            self._library = None
        return True

    def after_vocabulary_change(self):
        self._words = None
        self.model.refresh(); self.show_current(); self.replan_timer.start()

    # ---------- exit ----------
    def closeEvent(self, e):
        self.ps_timer.stop(); self.replan_timer.stop()
        if self.puller is not None:
            self.puller.cancel(); self.puller.wait(3000)
        if self.importer is not None:
            self.importer.cancel(); self.importer.wait(5000)
        if self.worker is not None:
            self.worker.cancel(); self.worker.wait(5000)
        self.tasks.wait_all(3000)
        self.save_window()
        self.store.save()
        if self.s.get("unload_exit") and self.used_models:
            try:
                unload_models(self.host(), set(self.used_models), timeout=3)
            except Exception:
                pass
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv); app.setApplicationName("MemeTagger"); app.setStyle("Fusion")
    remember_system_scheme(app)
    s = Settings()
    apply_theme(app, s.get("theme"))
    w = Main(s); w.show()
    paths = [a for a in sys.argv[1:] if Path(a).exists()]
    if paths:
        w.add_paths(paths)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
