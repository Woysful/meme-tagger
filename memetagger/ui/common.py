"""Shared UI helpers: theme application, settings with defaults, background tasks, small widgets."""
from __future__ import annotations
from PIL import Image
from PySide6.QtCore import QObject, QSettings, Qt, QThread, Signal
from PySide6.QtGui import QColor, QImage, QPalette
from PySide6.QtWidgets import QApplication, QFrame, QLabel, QVBoxLayout, QWidget
from ..paths import user_data_dir
from ..theme import KIND_COLORS, THEMES, build_style, make_check_icon

_SYSTEM_DARK = [False]


def detect_system_dark(app) -> bool:
    try:
        return app.styleHints().colorScheme() == Qt.ColorScheme.Dark
    except Exception:
        pass
    try:
        return app.palette().color(QPalette.ColorRole.Window).lightness() < 128
    except Exception:
        return False


def remember_system_scheme(app):
    _SYSTEM_DARK[0] = detect_system_dark(app)


def current_theme(mode: str) -> dict:
    dark = _SYSTEM_DARK[0] if mode not in ("dark", "light") else mode == "dark"
    return THEMES["dark" if dark else "light"]


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


def apply_theme(app, mode: str) -> dict:
    """The app always uses its own palette and stylesheet; 'system' only picks the light or dark variant
    (system palette roles made some widgets blend into the background on several desktop themes)."""
    t = current_theme(mode)
    app.setPalette(make_palette(t))
    app.setStyleSheet(build_style(t, make_check_icon(user_data_dir() / "ui" / "check.png")))
    app.setProperty("mt_theme", t)
    return t


def theme() -> dict:
    app = QApplication.instance()
    return (app.property("mt_theme") if app else None) or THEMES["light"]


def kind_color(kind: str | None) -> QColor:
    return QColor(KIND_COLORS.get(kind or "other", KIND_COLORS["other"]))


def set_state(widget, state: str):
    """Switches a style state (QLabel#hint[state="warn"] etc.); colors come from the current theme."""
    widget.setProperty("state", state)
    widget.style().unpolish(widget); widget.style().polish(widget)


def pil_to_qimage(im: Image.Image | None) -> QImage:
    if im is None:
        return QImage()
    im = im.convert("RGBA")
    return QImage(im.tobytes(), im.width, im.height, QImage.Format.Format_RGBA8888).copy()


def fmt_eta(sec: float) -> str:
    sec = int(sec)
    if sec < 90:
        return f"{sec}s"
    return f"{sec // 60}m {sec % 60:02d}s" if sec < 3600 else f"{sec // 3600}h {sec % 3600 // 60:02d}m"


def hint(text: str = "", wrap: bool = True) -> QLabel:
    lab = QLabel(text); lab.setObjectName("hint"); lab.setWordWrap(wrap)
    return lab


def card(*widgets, layout=None, name: str = "card") -> QFrame:
    f = QFrame(); f.setObjectName(name)
    lay = layout or QVBoxLayout()
    for w in widgets:
        lay.addWidget(w)
    f.setLayout(lay)
    return f


class Settings:
    """QSettings with one table of defaults and types, so every value is read the same way everywhere."""
    DEFAULTS = {
        # analysis
        "engine": "auto", "model": "qwen3-vl:8b", "host": "", "unload_exit": True,
        "wd": True, "wd_model": "eva02-large", "wd_device": "auto",
        "gen_tags": True, "gen_text": True, "gen_desc": True, "ocr": True, "verify": True,
        "strictness": "balanced", "max_tags": 8, "show_vocab": False,
        "cthr": 0.30, "gthr": 0.35, "fallback": True,
        # library
        "out": "", "copy": False, "levels": "franchise > character", "min_count": 3, "gather": True,
        "template": "{original} [{tags}]", "text_max": 60, "desc_max": 100, "limit_model": True, "desc_lang": "English",
        "metadata": "embed",
        # tags
        "skip": True, "banned": "", "folder_tags_all": False,
        # window
        "theme": "system", "tab": 0, "show_settings": True,
    }

    def __init__(self, qs: QSettings | None = None):
        self.qs = qs or QSettings("memetagger", "memetagger")

    def get(self, key: str):
        d = self.DEFAULTS[key]
        try:
            return self.qs.value(key, d, type(d))
        except (TypeError, ValueError):
            return d

    def set(self, key: str, value):
        self.qs.setValue(key, value)

    def raw(self, key: str, default=None):
        return self.qs.value(key, default)

    def sync(self):
        self.qs.sync()


class Task(QThread):
    """Runs fn() in the background; callback(result, error) is called in the GUI thread."""
    done = Signal(object, object)

    def __init__(self, fn, callback=None, parent: QObject | None = None):
        super().__init__(parent)
        self.fn, self.callback = fn, callback
        self.done.connect(self._deliver)

    def run(self):
        try:
            self.done.emit(self.fn(), None)
        except Exception as e:
            self.done.emit(None, f"{e}" or type(e).__name__)

    def _deliver(self, res, err):
        if self.callback:
            self.callback(res, err)


class TaskRunner:
    def __init__(self):
        self.tasks: set[Task] = set()

    def run(self, fn, callback=None):
        t = Task(fn, callback)
        self.tasks.add(t)
        t.finished.connect(lambda: self.tasks.discard(t))
        t.start()
        return t

    def wait_all(self, ms: int = 3000):
        for t in list(self.tasks):
            t.wait(ms)


def vbox(*items, margins=(0, 0, 0, 0), spacing=8) -> QWidget:
    w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(*margins); lay.setSpacing(spacing)
    for it in items:
        if isinstance(it, QWidget):
            lay.addWidget(it)
        else:
            lay.addLayout(it)
    return w
