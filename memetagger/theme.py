"""Colors and QSS without Qt imports (so contrast can be checked by tests). No palette(...) roles in the stylesheet:
every color is explicit, otherwise widgets blend into the background on some system themes."""
from __future__ import annotations
from pathlib import Path

# Tag kinds: solid dots/borders; chip backgrounds use the same colors at low alpha, so both themes stay readable.
KIND_COLORS = {"character": "#e0607e", "franchise": "#4f86e8", "meme_format": "#e39a2d", "medium": "#9b7fd6",
               "subject": "#2fae7f", "theme": "#c464c9", "other": "#8a8fa8"}

THEMES = {
    "dark": dict(window="#1b1c23", card="#252733", alt="#2a2c3a", input="#1c1d26", raised="#34374a", hover="#40445c",
                 border="#6a6f8e", text="#eceef7", muted="#aeb3cb", disabled="#8a8fa8", accent="#6c5ce8",
                 accent_hover="#5c4cdc", on_accent="#ffffff", sel="rgba(108, 92, 232, 100)", ok="#4cd6a8", bad="#ff8f7a",
                 track="#3b3f55", handle="#7a7f9c"),
    "light": dict(window="#e9ebf3", card="#ffffff", alt="#f4f5fa", input="#f5f6fb", raised="#e6e8f2", hover="#d7dbeb",
                  border="#8087a3", text="#1b1e2a", muted="#515a73", disabled="#6b7189", accent="#5646d0",
                  accent_hover="#6a5bdc", on_accent="#ffffff", sel="rgba(86, 70, 208, 60)", ok="#0a7357", bad="#b83224",
                  track="#cfd3e4", handle="#8e94ad"),
}


def _lin(c: float) -> float:
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def _rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def luminance(h: str) -> float:
    r, g, b = (_lin(v / 255) for v in _rgb(h))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def blend(fg_rgba: str, bg: str) -> str:
    """'rgba(r, g, b, a)' поверх #rrggbb -> #rrggbb (для проверки контраста выделения)."""
    r, g, b, a = [float(x) for x in fg_rgba[fg_rgba.index("(") + 1:-1].split(",")]
    br, bg_, bb = _rgb(bg)
    mix = lambda f, bgc: round(f * a / 255 + bgc * (1 - a / 255))
    return "#%02x%02x%02x" % (mix(r, br), mix(g, bg_), mix(b, bb))


def make_check_icon(path: Path) -> str:
    """Белая «галочка» 16x16 для отмеченных чекбоксов (рисуется один раз)."""
    try:
        if not path.exists():
            from PIL import Image, ImageDraw
            path.parent.mkdir(parents=True, exist_ok=True)
            big = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
            ImageDraw.Draw(big).line([(14, 34), (26, 46), (50, 18)], fill=(255, 255, 255, 255), width=9, joint="curve")
            big.resize((16, 16), Image.LANCZOS).save(path)
        return str(path).replace("\\", "/")
    except Exception:
        return ""


def build_style(t: dict, check_icon: str = "") -> str:
    tick = f'image: url("{check_icon}");' if check_icon else ""
    return f"""
* {{ font-size: 13px; }}
QMainWindow, QDialog, QMessageBox {{ background: {t['window']}; }}
QLabel {{ color: {t['text']}; background: transparent; }}
QLabel#title {{ font-size: 20px; font-weight: 700; }}
QLabel#muted, QLabel#hint {{ color: {t['muted']}; }}
QLabel#hint[state="warn"] {{ color: {t['bad']}; }}
QLabel#badge {{ padding: 5px 12px; border-radius: 12px; background: {t['raised']}; border: 1px solid {t['border']}; color: {t['text']}; }}
QLabel#badge[state="ok"] {{ color: {t['ok']}; }}
QLabel#badge[state="bad"] {{ color: {t['bad']}; }}
QFrame#card {{ background: {t['card']}; border: 1px solid {t['border']}; border-radius: 12px; }}
QFrame#card QLabel {{ background: transparent; }}
QLabel#drop {{ border: 2px dashed {t['border']}; border-radius: 16px; padding: 48px; font-size: 16px; color: {t['muted']}; }}
QCheckBox {{ color: {t['text']}; spacing: 8px; background: transparent; }}
QCheckBox:disabled {{ color: {t['disabled']}; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid {t['border']}; border-radius: 4px; background: {t['input']}; }}
QCheckBox::indicator:hover {{ border-color: {t['accent']}; }}
QCheckBox::indicator:checked {{ background: {t['accent']}; border-color: {t['accent']}; {tick} }}
QCheckBox::indicator:disabled {{ background: {t['raised']}; border-color: {t['disabled']}; }}
QPushButton {{ padding: 7px 14px; border-radius: 8px; border: 1px solid {t['border']}; background: {t['raised']}; color: {t['text']}; }}
QPushButton:hover {{ background: {t['hover']}; }}
QPushButton:pressed {{ background: {t['border']}; }}
QPushButton:disabled {{ color: {t['disabled']}; background: {t['raised']}; border-color: {t['raised']}; }}
QPushButton::menu-indicator {{ width: 0px; }}
QPushButton#primary {{ background: {t['accent']}; color: {t['on_accent']}; border: 1px solid {t['accent']}; font-weight: 600; padding: 8px 22px; }}
QPushButton#primary:hover {{ background: {t['accent_hover']}; }}
QPushButton#primary:disabled {{ background: {t['raised']}; color: {t['disabled']}; border-color: {t['raised']}; }}
QPushButton#danger {{ color: {t['bad']}; }}
QPushButton#danger:disabled {{ color: {t['disabled']}; }}
QPushButton#flat {{ border: 1px solid transparent; background: transparent; padding: 7px 10px; }}
QPushButton#flat:hover {{ background: {t['hover']}; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ padding: 5px 8px; border: 1px solid {t['border']}; border-radius: 6px;
    background: {t['input']}; color: {t['text']}; selection-background-color: {t['accent']}; selection-color: {t['on_accent']}; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {t['accent']}; }}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{ color: {t['disabled']}; }}
QComboBox QAbstractItemView {{ background: {t['card']}; color: {t['text']}; border: 1px solid {t['border']}; outline: 0;
    selection-background-color: {t['accent']}; selection-color: {t['on_accent']}; }}
QMenu {{ background: {t['card']}; color: {t['text']}; border: 1px solid {t['border']}; padding: 6px; }}
QMenu::item {{ padding: 7px 24px 7px 14px; border-radius: 6px; }}
QMenu::item:selected {{ background: {t['accent']}; color: {t['on_accent']}; }}
QMenu::separator {{ height: 1px; background: {t['border']}; margin: 6px 8px; }}
QToolTip {{ background: {t['card']}; color: {t['text']}; border: 1px solid {t['border']}; padding: 6px; }}
QTableView, QTableWidget, QListView, QTreeView {{ border: none; background: {t['card']}; alternate-background-color: {t['alt']};
    color: {t['text']}; selection-background-color: {t['sel']}; selection-color: {t['text']}; outline: 0; }}
QTableView::item, QTableWidget::item {{ padding: 4px 8px; }}
QAbstractItemView {{ background: {t['card']}; color: {t['text']}; selection-background-color: {t['accent']};
    selection-color: {t['on_accent']}; }}
QPlainTextEdit, QTextEdit {{ padding: 4px 6px; border: 1px solid {t['border']}; border-radius: 6px; background: {t['input']};
    color: {t['text']}; selection-background-color: {t['accent']}; selection-color: {t['on_accent']}; }}
QPlainTextEdit:focus, QTextEdit:focus {{ border: 1px solid {t['accent']}; }}
QGroupBox {{ border: 1px solid {t['border']}; border-radius: 10px; margin-top: 18px; padding: 14px 10px 10px 10px;
    color: {t['text']}; font-weight: 600; background: transparent; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 12px; padding: 0 6px; color: {t['muted']}; }}
QGroupBox QLabel, QGroupBox QCheckBox {{ font-weight: 400; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QLabel#section {{ color: {t['muted']}; font-weight: 700; font-size: 12px; }}
QLabel#filename {{ font-weight: 600; font-size: 14px; }}
QLabel#image {{ background: {t['input']}; border: 1px solid {t['border']}; border-radius: 10px; }}
QFrame#chip {{ border-radius: 11px; }}
QFrame#chip QLabel {{ background: transparent; }}
QToolButton {{ border: none; background: transparent; color: {t['muted']}; padding: 0 2px; }}
QToolButton:hover {{ color: {t['bad']}; }}
QPushButton#suggest {{ padding: 3px 10px; border-radius: 11px; border: 1px dashed {t['border']}; background: transparent; }}
QPushButton#suggest:hover {{ background: {t['hover']}; }}
QPushButton#toggle:checked {{ background: {t['hover']}; border-color: {t['accent']}; }}
QHeaderView::section {{ background: {t['card']}; color: {t['muted']}; border: none; border-bottom: 1px solid {t['border']};
    padding: 8px; font-weight: 600; }}
QProgressBar {{ border: none; border-radius: 4px; background: {t['track']}; max-height: 8px; min-height: 8px; }}
QProgressBar::chunk {{ background: {t['accent']}; border-radius: 4px; }}
QTabWidget::pane {{ border: none; background: transparent; top: 6px; }}
QTabBar {{ background: transparent; }}
QTabBar::tab {{ background: {t['raised']}; color: {t['text']}; border: 1px solid {t['border']}; border-radius: 8px;
    padding: 7px 16px; margin: 0 6px 0 0; }}
QTabBar::tab:hover:!selected {{ background: {t['hover']}; }}
QTabBar::tab:selected {{ background: {t['accent']}; color: {t['on_accent']}; border-color: {t['accent']}; font-weight: 600; }}
QScrollBar:vertical {{ background: transparent; width: 12px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {t['handle']}; border-radius: 5px; min-height: 30px; margin: 1px; }}
QScrollBar::handle:horizontal {{ background: {t['handle']}; border-radius: 5px; min-width: 30px; margin: 1px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0px; height: 0px; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
QSplitter::handle {{ background: transparent; }}
"""
