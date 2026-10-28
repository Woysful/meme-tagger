"""Right-hand panel: the selected image large, its tags as chips, character candidates and editable text."""
from __future__ import annotations
from PySide6.QtCore import QSize, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QImageReader, QPixmap
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QScrollArea, QVBoxLayout,
                               QWidget)
from ..tags import clean_tag
from .common import hint, set_state
from .model import STATUS, Row
from .widgets import ElidedLabel, FlowLayout, ImageView, TagChip, TagLineEdit


def section(text: str) -> QLabel:
    lab = QLabel(text.upper()); lab.setObjectName("section")
    return lab


def load_preview(path, max_side: int = 1600) -> QPixmap:
    r = QImageReader(str(path)); r.setAutoTransform(True)
    s = r.size()
    if s.isValid() and max(s.width(), s.height()) > max_side:
        r.setScaledSize(s.scaled(QSize(max_side, max_side), Qt.AspectRatioMode.KeepAspectRatio))
    img = r.read()
    return QPixmap.fromImage(img) if not img.isNull() else QPixmap()


class PreviewPanel(QWidget):
    changed = Signal(int, str)          # row id, what: tags | text | description | kind
    add_to_selection = Signal(list)     # tags typed while several rows are selected

    def __init__(self, words_fn, kind_of, parent=None):
        super().__init__(parent)
        self.kind_of, self.row, self.selected = kind_of, None, 0
        self._loading = False
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(8)
        self.image = ImageView(); self.image.double_clicked.connect(self.open_file)
        self.image.setToolTip("Double-click to open the file")
        outer.addWidget(self.image, 5)

        body = QWidget(); lay = QVBoxLayout(body); lay.setContentsMargins(2, 4, 8, 4); lay.setSpacing(6)
        self.name = ElidedLabel(""); self.name.setObjectName("filename")
        self.meta = hint("", wrap=False)
        self.state = hint("")
        lay.addWidget(self.name); lay.addWidget(self.meta); lay.addWidget(self.state)

        lay.addWidget(section("Tags"))
        self.chips_box = QWidget(); self.chips = FlowLayout(self.chips_box, 6)
        lay.addWidget(self.chips_box)
        self.add_edit = TagLineEdit(words_fn); self.add_edit.setPlaceholderText("Add tags… (Enter)")
        self.add_edit.returnPressed.connect(self._add_typed)
        self.add_btn = QPushButton("Add"); self.add_btn.clicked.connect(self._add_typed)
        row = QHBoxLayout(); row.addWidget(self.add_edit, 1); row.addWidget(self.add_btn)
        lay.addLayout(row)
        self.multi_hint = hint("")
        lay.addWidget(self.multi_hint)

        self.cand_title = section("Character candidates (WD14)")
        self.cand_box = QWidget(); self.cands = FlowLayout(self.cand_box, 6)
        lay.addWidget(self.cand_title); lay.addWidget(self.cand_box)

        lay.addWidget(section("Text on the image"))
        self.text_edit = QLineEdit(); self.text_edit.setPlaceholderText("used by {text} in file names")
        self.text_edit.editingFinished.connect(self._text_done)
        lay.addWidget(self.text_edit)
        lay.addWidget(section("Scene description"))
        self.desc_edit = QPlainTextEdit(); self.desc_edit.setFixedHeight(64)
        self.desc_edit.textChanged.connect(self._desc_typed)
        lay.addWidget(self.desc_edit)
        lay.addWidget(section("New location"))
        self.dest = hint("")
        self.dest.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.dest)
        self.why_btn = QPushButton("Why these tags?"); self.why_btn.setObjectName("flat"); self.why_btn.setCheckable(True)
        self.why = hint(""); self.why.setVisible(False)
        self.why.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.why_btn.toggled.connect(self.why.setVisible)
        lay.addWidget(self.why_btn, 0, Qt.AlignmentFlag.AlignLeft); lay.addWidget(self.why)
        lay.addStretch(1)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(body)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(scroll, 6)
        self.show_row(None)

    # ---- display ----
    def show_row(self, row: Row | None, selected: int = 1):
        same = row is not None and self.row is not None and row.id == self.row.id
        self.row, self.selected = row, selected
        self._loading = True
        for w in (self.add_edit, self.add_btn, self.text_edit, self.desc_edit, self.why_btn):
            w.setEnabled(row is not None)
        if row is None:
            self.image.set_pixmap(None, "Select a file to see it here")
            self.name.setText(""); self.meta.setText(""); self.state.setText("")
            self.text_edit.clear(); self.desc_edit.clear(); self.dest.setText(""); self.why.setText("")
            self.chips.clear(); self.cands.clear(); self.cand_title.setVisible(False); self.multi_hint.setText("")
            self._loading = False
            return
        it = row.item
        if not same:
            pix = load_preview(it.src)
            self.image.set_pixmap(pix, "Cannot show this image")
            size = it.src.stat().st_size if it.src.exists() else 0
            dims = f"{pix.width()}×{pix.height()} · " if not pix.isNull() else ""
            self.meta.setText(f"{dims}{size / 1024:.0f} KB · {it.src.parent}")
        self.name.setText(it.src.name)
        label, _ = STATUS.get(row.status, ("", ""))
        src = {"vlm": "vision model", "classic": "classic engine"}.get(row.source, "")
        self.state.setText(f"{label}{' · ' + src if src else ''}" + (f"\n{row.error}" if row.error else ""))
        set_state(self.state, "warn" if row.status == "error" else "")
        self._render_chips()
        if self.text_edit.text() != it.text:
            self.text_edit.setText(it.text)
        if self.desc_edit.toPlainText() != it.description:
            self.desc_edit.setPlainText(it.description)
        self.dest.setText(row.dest or "—")
        notes = list(row.notes) + ([f"OCR: {row.ocr}"] if row.ocr else [])
        self.why.setText("\n".join(notes) or "No analysis notes for this file.")
        self.multi_hint.setText(f"Tags you add go to all {selected} selected files." if selected > 1 else "")
        self._loading = False

    def _render_chips(self):
        it = self.row.item
        self.chips.clear()
        for t in it.tags:
            chip = TagChip(t, it.kinds.get(t) or self.kind_of(t))
            chip.removed.connect(self._remove)
            chip.kind_changed.connect(self._set_kind)
            self.chips.addWidget(chip)
        if not it.tags:
            self.chips.addWidget(hint("No tags yet", wrap=False))
        self.cands.clear()
        cands = [s for s in self.row.suggestions if s[0] not in it.tags]
        self.cand_title.setVisible(bool(cands))
        for tag, fr, p in cands:
            b = QPushButton(f"+ {tag}" + (f" · {fr}" if fr else "") + f" · {p:.0%}")
            b.setObjectName("suggest"); b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setToolTip("WD14 thinks this character may be in the image. Click to add it" + (" with its franchise" if fr else ""))
            b.clicked.connect(lambda _=False, t=tag, f=fr: self._add_candidate(t, f))
            self.cands.addWidget(b)
        self.chips_box.updateGeometry(); self.cand_box.updateGeometry()

    # ---- edits ----
    def _emit(self, what):
        self.row.item.touched = True
        if self.row.status not in ("queued", "analyzing"):
            self.row.status = "edited"
        self.changed.emit(self.row.id, what)

    def _remove(self, tag):
        it = self.row.item
        if tag in it.tags:
            it.tags = [t for t in it.tags if t != tag]
            self._emit("tags"); self._render_chips()

    def _set_kind(self, tag, kind):
        self.row.item.kinds[tag] = kind
        self._emit("kind"); self._render_chips()

    def _add_typed(self):
        tags = [t for t in map(clean_tag, self.add_edit.tags()) if t]
        self.add_edit.clear()
        if not tags or self.row is None:
            return
        if self.selected > 1:
            self.add_to_selection.emit(tags); return
        it = self.row.item
        new = [t for t in tags if t not in it.tags]
        if new:
            it.tags = it.tags + new
            self._emit("tags"); self._render_chips()

    def _add_candidate(self, tag, franchise):
        it = self.row.item
        add = [t for t in ([franchise] if franchise else []) + [tag] if t and t not in it.tags]
        it.kinds[tag] = "character"
        if franchise:
            it.kinds[franchise] = "franchise"
        it.tags = it.tags + add
        self._emit("tags"); self._render_chips()

    def _text_done(self):
        if self._loading or self.row is None or self.text_edit.text().strip() == self.row.item.text:
            return
        self.row.item.text = self.text_edit.text().strip(); self._emit("text")

    def _desc_typed(self):
        if self._loading or self.row is None:
            return
        txt = self.desc_edit.toPlainText().strip()
        if txt != self.row.item.description:
            self.row.item.description = txt; self._emit("description")

    def open_file(self):
        if self.row is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.row.item.src)))
