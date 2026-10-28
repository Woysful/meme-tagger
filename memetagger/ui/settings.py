"""Settings side panel. Every control is bound to a Settings key: it loads, saves and announces changes by itself."""
from __future__ import annotations
import re
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLayout, QLineEdit,
                               QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QTabWidget, QVBoxLayout, QWidget)
from ..models import CUSTOM, DEFAULT_WD, WD_MODELS
from ..naming import PRESETS, render_filename, validate
from ..organizer import LEVEL_PRESETS, parse_levels
from ..tags import KIND_LABELS
from .common import Settings, WrapCheckBox, hint, set_state, vbox

CONFIG_FILES = [("folder_hints.json", "Folder name → tags"), ("seed_tags.txt", "Starting tag vocabulary"),
                ("aliases.json", "Character → franchise aliases"), ("vocab.txt", "Classic engine labels")]
STRICTNESS = [("Lenient", "lenient"), ("Balanced (recommended)", "balanced"), ("Strict", "strict")]
STRICT_HINT = {
    "lenient": "More tags, more mistakes. Accepts the vision model's franchise/character guesses unless it is unsure itself.",
    "balanced": "Confident guesses are kept, medium ones are double-checked with a short extra question, "
                "and WD14 or text on the image counts as confirmation.",
    "strict": "Only confirmed franchises and characters. Even confident guesses are double-checked; unconfirmed low-confidence guesses are dropped.",
}


def form() -> QFormLayout:
    f = QFormLayout()
    f.setVerticalSpacing(8); f.setHorizontalSpacing(10)
    f.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
    f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    f.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
    return f


def group(title: str, f) -> QGroupBox:
    g = QGroupBox(title); g.setLayout(f)
    return g


def page(*groups) -> QScrollArea:
    w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(4, 0, 10, 8); lay.setSpacing(4)
    lay.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)     # follow the panel width, never scroll sideways
    w.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    for g in groups:
        lay.addWidget(g)
    lay.addStretch(1)
    s = QScrollArea(); s.setWidgetResizable(True); s.setWidget(w)
    s.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    return s


def hrow(*widgets, stretch_first=True) -> QHBoxLayout:
    r = QHBoxLayout(); r.setSpacing(6)
    for i, w in enumerate(widgets):
        r.addWidget(w, 1 if (i == 0 and stretch_first) else 0)
    return r


def shrinkable(c: QComboBox) -> QComboBox:
    """Long item texts must not widen the whole settings panel (they are elided instead)."""
    c.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    c.setMinimumContentsLength(10)
    return c


def combo(options) -> QComboBox:
    c = shrinkable(QComboBox())
    for label, key in options:
        c.addItem(label, key)
        c.setItemData(c.count() - 1, label, Qt.ItemDataRole.ToolTipRole)
    c.view().setMinimumWidth(c.view().sizeHintForColumn(0) + 32)      # the open list shows full texts
    return c


def check(text: str) -> WrapCheckBox:
    return WrapCheckBox(text)


class SettingsPanel(QWidget):
    changed = Signal(str)

    def __init__(self, settings: Settings, preview_item_fn=None, want_fn=None, parent=None):
        super().__init__(parent)
        self.s, self.preview_item_fn, self.want_fn = settings, preview_item_fn, want_fn
        self.tabs = QTabWidget(); self.tabs.setDocumentMode(True)
        self.tabs.tabBar().setExpanding(False); self.tabs.tabBar().setDrawBase(False)
        self.tabs.addTab(self._analysis_tab(), "Analysis")
        self.tabs.addTab(self._library_tab(), "Library")
        self.tabs.addTab(self._tags_tab(), "Tags")
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.addWidget(self.tabs)
        self.tabs.setCurrentIndex(max(0, min(2, self.s.get("tab"))))
        self.tabs.currentChanged.connect(lambda i: self.s.set("tab", i))
        self.on_engine(); self.on_strictness(); self.on_levels(); self.update_template(); self.on_wd()

    # ---- binding ----
    def bind(self, w, key: str, on_edit_finished: bool = False):
        v = self.s.get(key)
        emit = lambda *_: self.changed.emit(key)
        if isinstance(w, QCheckBox):
            w.setChecked(bool(v)); w.toggled.connect(lambda x: (self.s.set(key, x), emit()))
        elif isinstance(w, QComboBox) and w.isEditable():
            w.setCurrentText(v); w.currentTextChanged.connect(lambda x: (self.s.set(key, x.strip()), emit()))
        elif isinstance(w, QComboBox):
            w.setCurrentIndex(max(0, w.findData(v)))
            w.currentIndexChanged.connect(lambda _i: (self.s.set(key, w.currentData()), emit()))
        elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
            w.setValue(v); w.valueChanged.connect(lambda x: (self.s.set(key, x), emit()))
        elif isinstance(w, QLineEdit):
            w.setText(v)
            if on_edit_finished:
                w.editingFinished.connect(lambda: (self.s.set(key, w.text().strip()), emit()))
            else:
                w.textChanged.connect(lambda x: (self.s.set(key, x.strip()), emit()))
        return w

    # ---- tabs ----
    def _analysis_tab(self):
        b = self.bind
        self.engine = b(combo([("Auto (Ollama, else classic)", "auto"), ("Vision model only", "vlm"),
                               ("Classic only (no Ollama)", "classic")]), "engine")
        self.engine.currentIndexChanged.connect(self.on_engine)
        self.model = shrinkable(QComboBox()); self.model.setEditable(True); b(self.model, "model")
        self.refresh_btn = QPushButton("⟳"); self.refresh_btn.setObjectName("flat")
        self.refresh_btn.setToolTip("Reload the list of installed Ollama models")
        self.model_hint = hint("")
        self.pull_btn = QPushButton("⬇ Download"); self.pull_btn.setToolTip("Download the model above (like `ollama pull`)")
        self.unload_btn = QPushButton("⏏ Free VRAM")
        self.unload_btn.setToolTip("Unload the model now (like `ollama stop`). The next analysis loads it again.")
        self.pull_label = hint(""); self.pull_bar = QProgressBar(); self.pull_bar.setTextVisible(False)
        self.pull_label.setVisible(False); self.pull_bar.setVisible(False)
        self.host = b(QLineEdit(), "host", on_edit_finished=True)
        self.host.setPlaceholderText("default: $OLLAMA_HOST or 127.0.0.1:11434")
        self.test_btn = QPushButton("Test")
        self.gpu_label = hint("GPU: —")
        self.unload_exit = b(check("Free GPU memory when closing the app"), "unload_exit")
        f = form()
        f.addRow("Engine", self.engine)
        f.addRow("Model", hrow(self.model, self.refresh_btn))
        f.addRow(self.model_hint)
        f.addRow(hrow(self.pull_btn, self.unload_btn, stretch_first=False))
        f.addRow(self.pull_label); f.addRow(self.pull_bar)
        f.addRow("Ollama URL", hrow(self.host, self.test_btn))
        f.addRow(self.gpu_label); f.addRow(self.unload_exit)
        vlm = group("Vision model (Ollama)", f)

        self.wd = b(check("Recognize anime characters with WD14"), "wd")
        self.wd_model = b(combo([(label, key) for key, (_repo, label) in WD_MODELS.items()] + [("Custom model…", CUSTOM)]),
                          "wd_model")
        self.wd_custom = b(QLineEdit(), "wd_custom", on_edit_finished=True)
        self.wd_custom.setPlaceholderText("owner/repo  or  owner/repo:file.onnx  or a folder")
        self.wd_custom_btn = QPushButton("…"); self.wd_custom_btn.setToolTip("Choose a folder with model.onnx and selected_tags.csv")
        self.wd_custom_box = vbox(hrow(self.wd_custom, self.wd_custom_btn), hint(
            "Any ONNX tagger in the WD format from Hugging Face or a local folder: model.onnx + selected_tags.csv "
            "(categories 4 = character, 3 = copyright). Optional preprocess.json. Gated repos need HF_TOKEN."), spacing=4)
        self.wd_device = b(combo([("Auto: GPU if available", "auto"), ("GPU (CUDA)", "cuda"), ("CPU", "cpu")]), "wd_device")
        self.wd.toggled.connect(self.on_wd)
        self.wd_model.currentIndexChanged.connect(self.on_wd)
        f = form()
        f.addRow(self.wd); f.addRow("Model", self.wd_model); f.addRow(self.wd_custom_box); f.addRow("Device", self.wd_device)
        f.addRow(hint("Its character guesses with probabilities are given to the vision model and confirm its answers. "
                      "The GPU needs onnxruntime-gpu; the model is downloaded once."))
        wd = group("Character recognition", f)

        self.gen_tags = b(check("Tags"), "gen_tags")
        self.gen_text = b(check("Text on the image (for file names)"), "gen_text")
        self.gen_desc = b(check("Scene description"), "gen_desc")
        f = form()
        for w in (self.gen_tags, self.gen_text, self.gen_desc):
            f.addRow(w)
        f.addRow(hint("Switched-off items are skipped (faster), existing values are kept."))
        gen = group("Generate", f)

        self.strictness = b(combo(STRICTNESS), "strictness")
        self.strictness.currentIndexChanged.connect(self.on_strictness)
        self.strict_hint = hint("")
        self.verify = b(check("Double-check doubtful franchise/character guesses"), "verify")
        self.verify.setToolTip("One short extra question per image, only when something needs checking.")
        self.max_tags = b(QSpinBox(), "max_tags"); self.max_tags.setRange(3, 20)
        self.ocr = b(check("Read text on images (OCR) as a hint"), "ocr")
        self.show_vocab = b(check("Show my frequent tags to the model"), "show_vocab")
        self.show_vocab.setToolTip("More consistent tag names, but the model may pick a known tag that does not fit.")
        f = form()
        f.addRow("Strictness", self.strictness); f.addRow(self.strict_hint)
        f.addRow(self.verify); f.addRow("Max tags per file", self.max_tags); f.addRow(self.ocr); f.addRow(self.show_vocab)
        quality = group("Tag quality", f)

        self.cthr = b(QDoubleSpinBox(), "cthr"); self.cthr.setRange(0.05, 0.95); self.cthr.setSingleStep(0.05)
        self.gthr = b(QDoubleSpinBox(), "gthr"); self.gthr.setRange(0.10, 0.95); self.gthr.setSingleStep(0.05)
        self.fallback = b(check("Always assign at least one tag"), "fallback")
        f = form()
        f.addRow("Concept confidence", self.cthr); f.addRow("WD14 tag confidence", self.gthr); f.addRow(self.fallback)
        f.addRow(hint("Higher = fewer, more certain tags."))
        self.classic_group = group("Classic engine (CLIP + WD14)", f)
        return page(vlm, wd, gen, quality, self.classic_group)

    def _library_tab(self):
        b = self.bind
        self.out = b(QLineEdit(), "out"); self.out.setPlaceholderText("empty: sort inside the added folder")
        self.out_btn = QPushButton("…"); self.out_btn.setToolTip("Choose the library folder")
        self.copy = b(check("Copy files into the library instead of moving"), "copy")
        f = form()
        f.addRow("Library folder", hrow(self.out, self.out_btn)); f.addRow(self.copy)
        lib = group("Library", f)

        self.level_preset = combo([(n, s) for n, s in LEVEL_PRESETS] + [("Custom", None)])
        self.levels = b(QLineEdit(), "levels"); self.levels.setPlaceholderText("e.g. franchise > character")
        self.levels_hint = hint("")
        self.min_count = b(QSpinBox(), "min_count"); self.min_count.setRange(1, 500)
        self.min_count.setSuffix(" images")
        self.gather = b(check("Move older files into a folder once it appears"), "gather")
        self.gather.setToolTip("When a franchise reaches the threshold, files of it waiting in the parent folder join "
                               "the new folder. Other files that are already in place are never touched.")
        self.level_preset.currentIndexChanged.connect(self.on_level_preset)
        self.levels.textChanged.connect(self.on_levels)
        f = form()
        f.addRow("Structure", self.level_preset); f.addRow("Levels", self.levels); f.addRow(self.levels_hint)
        f.addRow("Create a folder from", self.min_count); f.addRow(self.gather)
        f.addRow(hint("Files already in the library keep their place unless their tags change. "
                      "Theme tags (cat, war…) never become folders; search them instead."))
        folders = group("Folders", f)

        self.name_preset = combo([(n, t) for n, t in PRESETS] + [("Custom", None)])
        self.template = b(QLineEdit(), "template")
        self.tpl_preview = hint(""); self.tpl_warn = hint("")
        tpl_help = hint("Variables: {original} {meaningful} {text} {description} {tags}\n"
                        "Fallback {text|original} · limit {text:40} {tags:3} · optional part < - {text}>")
        self.text_max = b(QSpinBox(), "text_max"); self.text_max.setRange(15, 120)
        self.desc_max = b(QSpinBox(), "desc_max"); self.desc_max.setRange(15, 160)
        self.limit_model = b(check("Ask the model for these lengths (faster answers)"), "limit_model")
        self.desc_lang = b(combo([("English", "English"), ("Русский", "Russian"), ("Українська", "Ukrainian")]), "desc_lang")
        self.name_preset.currentIndexChanged.connect(self.on_name_preset)
        self.template.textChanged.connect(self.update_template)
        for w in (self.text_max, self.desc_max):
            w.valueChanged.connect(self.update_template)
        f = form()
        f.addRow("Preset", self.name_preset); f.addRow("Template", self.template)
        f.addRow(self.tpl_preview); f.addRow(self.tpl_warn); f.addRow(tpl_help)
        f.addRow("Max text length", self.text_max); f.addRow("Max description length", self.desc_max)
        f.addRow(self.limit_model); f.addRow("Description language", self.desc_lang)
        names = group("File names", f)

        self.metadata = b(combo([("Into the file (XMP)", "embed"), ("Sidecar file (.xmp)", "sidecar"), ("Off", "off")]),
                          "metadata")
        f = form()
        f.addRow("Tags & description", self.metadata)
        f.addRow(hint("Searchable in digiKam, Dolphin/Baloo, Lightroom… Tag kinds are kept, so re-adding a file "
                      "restores them. Formats that cannot hold XMP get a sidecar."))
        meta = group("Metadata", f)
        return page(lib, folders, names, meta)

    def _tags_tab(self):
        b = self.bind
        self.skip = b(check("Skip files that already have [tags] in the name"), "skip")
        self.banned = b(QLineEdit(), "banned", on_edit_finished=True)
        self.banned.setPlaceholderText("e.g.  photo  text_meme")
        self.folder_tags_all = b(check("Folder names may add any kind of tag"), "folder_tags_all")
        self.folder_tags_all.setToolTip("Off: folder hints only add franchises and characters. On: also themes like "
                                        "'Sad' → sad, which then lands on every file of that folder.")
        f = form()
        f.addRow(self.skip); f.addRow("Never use these tags", self.banned); f.addRow(self.folder_tags_all)
        rules = group("Rules", f)

        self.library_btn = QPushButton("Open tag library…")
        f = form()
        f.addRow(self.library_btn)
        f.addRow(hint("All known tags with their kind and usage: rename, merge (hirasawa_yui + yui_hirasawa), "
                      "change the kind or delete, also in files already in the library."))
        lib = group("Tag library", f)

        self.config_btns = {}
        f = form()
        for name, label in CONFIG_FILES:
            btn = QPushButton("Edit"); self.config_btns[name] = btn
            f.addRow(label, btn)
        self.open_config_btn = QPushButton("Open config folder")
        self.open_data_btn = QPushButton("Open data folder")
        self.reload_btn = QPushButton("Reload configuration")
        f.addRow(hrow(self.open_config_btn, self.open_data_btn, stretch_first=False)); f.addRow(self.reload_btn)
        cfg = group("Configuration files", f)
        return page(rules, lib, cfg)

    # ---- dependent widgets ----
    def on_wd(self, *_):
        on = self.wd.isChecked()
        for w in (self.wd_model, self.wd_device, self.wd_custom_box):
            w.setEnabled(on)
        self.wd_custom_box.setVisible(self.wd_model.currentData() == CUSTOM)

    def wd_spec(self) -> str:
        """Registry key, or the custom repo/folder (the default model while the custom field is empty)."""
        if self.wd_model.currentData() == CUSTOM:
            return self.wd_custom.text().strip() or DEFAULT_WD
        return self.wd_model.currentData()

    def on_engine(self, *_):
        self.set_classic_visible(self.engine.currentData() == "classic")

    def set_classic_visible(self, on: bool):
        self.classic_group.setVisible(on)

    def on_strictness(self, *_):
        self.strict_hint.setText(STRICT_HINT.get(self.strictness.currentData(), ""))

    def on_level_preset(self, i):
        spec = self.level_preset.itemData(i)
        if spec is not None and spec != self.levels.text().strip():
            self.levels.setText(spec)

    def on_levels(self, *_):
        spec = self.levels.text().strip()
        idx = next((i for i, (_n, s) in enumerate(LEVEL_PRESETS) if s == spec), len(LEVEL_PRESETS))
        self.level_preset.blockSignals(True); self.level_preset.setCurrentIndex(idx); self.level_preset.blockSignals(False)
        try:
            lv = parse_levels(spec)
        except ValueError as e:
            self.levels_hint.setText(str(e)); set_state(self.levels_hint, "warn"); return
        if not lv:
            self.levels_hint.setText("No folders: files are only renamed.")
        else:
            names = [" or ".join(KIND_LABELS[k].lower() for k in level) for level in lv]
            example = {"franchise": "k-on", "character": "hirasawa_yui", "meme_format": "drake", "medium": "anime",
                       "subject": "cat", "theme": "war", "other": "misc"}
            path = "/".join(example[level[0]] for level in lv)
            self.levels_hint.setText(f"{' › '.join(names)}  ·  e.g. {path}/")
        set_state(self.levels_hint, "")

    def level_spec(self) -> str:
        try:
            parse_levels(self.levels.text())
            return self.levels.text().strip()
        except ValueError:
            return self.s.DEFAULTS["levels"]

    def on_name_preset(self, i):
        t = self.name_preset.itemData(i)
        if t is not None and t != self.template.text().strip():
            self.template.setText(t)

    def template_text(self) -> str:
        t = self.template.text().strip()
        return t if not validate(t)[0] else PRESETS[0][1]

    def update_template(self, *_):
        tpl = self.template.text().strip()
        idx = next((i for i, (_n, t) in enumerate(PRESETS) if t == tpl), len(PRESETS))
        self.name_preset.blockSignals(True); self.name_preset.setCurrentIndex(idx); self.name_preset.blockSignals(False)
        errors, warnings = validate(tpl)
        it = self.preview_item_fn() if self.preview_item_fn else None
        if it is not None:
            name = render_filename(self.template_text(), it, it.src.suffix or ".jpg", self.text_max.value(), self.desc_max.value())
            self.tpl_preview.setText("Preview: " + name)
        if not errors and self.want_fn is not None:
            used = {n.strip().lower() for m in re.findall(r"\{([^{}]*)\}", tpl) for n in m.split(":")[0].split("|")}
            want = self.want_fn()
            for var in ("text", "description"):
                if var in used and var not in want:
                    warnings.append(f"{{{var}}} is used, but its generation is off: it stays empty unless you type it.")
        msg = (" · ".join(errors) + "  → the default template is used") if errors else "\n".join(warnings)
        self.tpl_warn.setText(msg); self.tpl_warn.setVisible(bool(msg)); set_state(self.tpl_warn, "warn" if msg else "")
