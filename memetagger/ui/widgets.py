"""Reusable widgets: flow layout, tag chips, tag input with completion, tag-chip table delegate, scaled image."""
from __future__ import annotations
import re
from PySide6.QtCore import QPoint, QRect, QSize, QStringListModel, Qt, Signal
from PySide6.QtGui import QAction, QColor, QFontMetrics, QPainter, QPixmap
from PySide6.QtWidgets import (QCompleter, QFrame, QHBoxLayout, QLabel, QLayout, QLineEdit, QMenu, QSizePolicy,
                               QStyle, QStyledItemDelegate, QToolButton)
from ..tags import KIND_LABELS, KINDS
from .common import kind_color, theme

TAGS_ROLE = Qt.ItemDataRole.UserRole + 1          # [(tag, kind)] for the chip delegate


class FlowLayout(QLayout):
    """Lays widgets out left to right and wraps them (Qt's flow layout example)."""

    def __init__(self, parent=None, spacing: int = 6):
        super().__init__(parent)
        self._items, self._sp = [], spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect); self._do(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        s = QSize()
        for it in self._items:
            s = s.expandedTo(it.minimumSize())
        return s

    def clear(self):
        while (it := self.takeAt(0)) is not None:
            if it.widget():
                it.widget().deleteLater()

    def _do(self, rect, test):
        x, y, line = rect.x(), rect.y(), 0
        for it in self._items:
            hint = it.sizeHint()
            nx = x + hint.width() + self._sp
            if nx - self._sp > rect.right() and line > 0:
                x, y = rect.x(), y + line + self._sp
                nx, line = x + hint.width() + self._sp, 0
            if not test:
                it.setGeometry(QRect(QPoint(x, y), hint))
            x, line = nx, max(line, hint.height())
        return y + line - rect.y()


def _rgba(c: QColor, a: int) -> str:
    return f"rgba({c.red()}, {c.green()}, {c.blue()}, {a})"


class TagChip(QFrame):
    removed = Signal(str)
    kind_changed = Signal(str, str)

    def __init__(self, tag: str, kind: str | None, parent=None):
        super().__init__(parent)
        self.tag, self.kind = tag, kind or "other"
        self.setObjectName("chip")
        c = kind_color(self.kind)
        self.setStyleSheet(f"QFrame#chip {{ background: {_rgba(c, 48)}; border: 1px solid {_rgba(c, 150)}; }}")
        lay = QHBoxLayout(self); lay.setContentsMargins(8, 2, 4, 2); lay.setSpacing(4)
        dot = QLabel("●"); dot.setStyleSheet(f"color: {c.name()};")
        name = QLabel(tag)
        x = QToolButton(); x.setText("✕"); x.setToolTip("Remove this tag"); x.setCursor(Qt.CursorShape.PointingHandCursor)
        x.clicked.connect(lambda: self.removed.emit(self.tag))
        for w in (dot, name, x):
            lay.addWidget(w)
        self.setToolTip(f"{KIND_LABELS.get(self.kind, 'Other')} · right-click to change the kind")
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)

    def _menu(self, pos):
        m = QMenu(self)
        for k in KINDS:
            a = QAction(("✓ " if k == self.kind else "   ") + KIND_LABELS[k], m)
            a.triggered.connect(lambda _=False, k=k: self.kind_changed.emit(self.tag, k))
            m.addAction(a)
        m.addSeparator()
        rm = QAction("Remove", m); rm.triggered.connect(lambda: self.removed.emit(self.tag)); m.addAction(rm)
        m.exec(self.mapToGlobal(pos))


_SEP = re.compile(r"[\s,]+")


class TagLineEdit(QLineEdit):
    """Space-separated tags with completion of the word under the cursor from the known tag vocabulary."""

    def __init__(self, words_fn, parent=None):
        super().__init__(parent)
        self.words_fn, self._words = words_fn, None
        self.model = QStringListModel(self)
        self.comp = QCompleter(self.model, self)
        self.comp.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.comp.setFilterMode(Qt.MatchFlag.MatchContains)
        self.comp.setMaxVisibleItems(12)
        self.comp.setWidget(self)
        self.comp.activated[str].connect(self._insert)
        self.textEdited.connect(self._update)

    def _token(self):
        left = self.text()[:self.cursorPosition()]
        m = re.search(r"[^\s,]*$", left)
        return m.start(), left[m.start():]

    def _update(self, *_):
        start, tok = self._token()
        if len(tok) < 2:
            self.comp.popup().hide(); return
        words = self.words_fn()
        if words is not self._words:                      # the vocabulary list is rebuilt when tags change
            self._words = words; self.model.setStringList(words)
        self.comp.setCompletionPrefix(tok)
        if self.comp.completionCount():
            self.comp.complete()
        else:
            self.comp.popup().hide()

    def _insert(self, word):
        start, tok = self._token()
        t, pos = self.text(), self.cursorPosition()
        rest = t[pos:]
        rest = rest[re.match(r"[^\s,]*", rest).end():]
        new = t[:start] + word + " " + rest.lstrip()
        self.setText(new); self.setCursorPosition(start + len(word) + 1)

    def keyPressEvent(self, e):
        if self.comp.popup().isVisible() and e.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return, Qt.Key.Key_Tab,
                                                         Qt.Key.Key_Escape, Qt.Key.Key_Backtab):
            e.ignore(); return
        super().keyPressEvent(e)

    def tags(self) -> list[str]:
        return [t for t in _SEP.split(self.text()) if t]


class TagsDelegate(QStyledItemDelegate):
    """Paints tags as colored chips; edits them as text with completion."""

    def __init__(self, words_fn, parent=None):
        super().__init__(parent)
        self.words_fn = words_fn

    def paint(self, p: QPainter, opt, idx):
        self.initStyleOption(opt, idx)
        opt.text = ""
        style = opt.widget.style() if opt.widget else None
        if style:
            style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, p, opt.widget)
        tags = idx.data(TAGS_ROLE) or []
        t = theme()
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        fm = QFontMetrics(opt.font)
        h, pad = fm.height() + 6, 6
        r = opt.rect.adjusted(pad, pad, -pad, -pad)
        x, y = r.x(), r.y()
        if not tags:
            p.setPen(QColor(t["bad"]))
            p.drawText(r, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "no tags")
        for name, kind in tags:
            w = fm.horizontalAdvance(name) + 16
            if x + w > r.right() and x > r.x():
                x, y = r.x(), y + h + 4
            if y + h > r.bottom() + 2:
                p.setPen(QColor(t["muted"])); p.drawText(QRect(r.right() - 20, r.bottom() - h, 20, h), Qt.AlignmentFlag.AlignCenter, "…")
                break
            c = kind_color(kind)
            bg = QColor(c); bg.setAlpha(50)
            border = QColor(c); border.setAlpha(150)
            p.setPen(border); p.setBrush(bg)
            p.drawRoundedRect(QRect(x, y, min(w, r.right() - x), h), h / 2, h / 2)
            p.setPen(QColor(t["text"]))
            p.drawText(QRect(x + 8, y, w - 16, h), Qt.AlignmentFlag.AlignVCenter, fm.elidedText(name, Qt.TextElideMode.ElideRight, max(10, r.right() - x - 16)))
            x += w + 4
        p.restore()

    def createEditor(self, parent, opt, idx):
        return TagLineEdit(self.words_fn, parent)

    def setEditorData(self, editor, idx):
        editor.setText(" ".join(t for t, _ in (idx.data(TAGS_ROLE) or [])))

    def setModelData(self, editor, model, idx):
        model.setData(idx, editor.text(), Qt.ItemDataRole.EditRole)


class ImageView(QLabel):
    """Shows a pixmap scaled to fit, keeping the aspect ratio."""
    double_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("image")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(200, 160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pix = QPixmap()

    def set_pixmap(self, pix: QPixmap | None, placeholder: str = ""):
        self._pix = pix or QPixmap()
        if self._pix.isNull():
            self.clear(); self.setText(placeholder)
        else:
            self.setText(""); self._rescale()

    def resizeEvent(self, e):
        super().resizeEvent(e); self._rescale()

    def _rescale(self):
        if not self._pix.isNull():
            s = self.size() - QSize(12, 12)
            self.setPixmap(self._pix.scaled(s, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def mouseDoubleClickEvent(self, e):
        self.double_clicked.emit()

    def sizeHint(self):
        return QSize(320, 260)


class ElidedLabel(QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self._full = text
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setText(text)

    def setText(self, text):
        self._full = text or ""
        self._elide()

    def full_text(self):
        return self._full

    def resizeEvent(self, e):
        super().resizeEvent(e); self._elide()

    def _elide(self):
        super().setText(self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideMiddle, max(20, self.width())))
        self.setToolTip(self._full)
