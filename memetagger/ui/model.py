"""Table model of the files in the current session (one Row per image) and its filter proxy."""
from __future__ import annotations
import itertools
from dataclasses import dataclass, field
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt, Signal
from PySide6.QtGui import QColor, QPixmap
from ..organizer import Item
from ..tags import clean_tag
from .common import theme
from .widgets import TAGS_ROLE

ROW_ROLE = Qt.ItemDataRole.UserRole + 2

STATUS = {   # key -> (label, theme color)
    "queued": ("Queued", "muted"), "analyzing": ("Analyzing…", "accent"), "done": ("Done", "ok"),
    "skipped": ("Kept tags", "muted"), "error": ("Error", "bad"), "edited": ("Edited", "accent"),
    "applied": ("Applied", "ok"), "idle": ("Not analyzed", "muted"),
}
C_THUMB, C_STATUS, C_FILE, C_TAGS, C_TEXT, C_DESC, C_DEST = range(7)
HEADERS = ["", "Status", "File", "Tags", "Text on the image", "Scene description", "New location"]

_ids = itertools.count(1)


@dataclass
class Row:
    item: Item
    status: str = "queued"
    id: int = field(default_factory=lambda: next(_ids))
    error: str = ""
    source: str = ""
    notes: list[str] = field(default_factory=list)
    suggestions: list[tuple[str, str | None, float]] = field(default_factory=list)
    ocr: str = ""
    thumb: QPixmap | None = None
    dest: str = ""
    dest_changed: bool = False


class ItemsModel(QAbstractTableModel):
    edited = Signal(int)                 # row id, after a user edit in the table

    def __init__(self, kind_of, parent=None):
        super().__init__(parent)
        self.rows: list[Row] = []
        self.kind_of = kind_of
        self._index: dict[int, int] = {}
        self.disabled_cols: set[int] = set()

    # ---- structure ----
    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(HEADERS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return HEADERS[section] + ("  (off)" if section in self.disabled_cols else "")
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.ToolTipRole and section in self.disabled_cols:
            return "Generation of this column is off; existing values are kept and stay editable."
        return None

    def flags(self, idx):
        f = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if idx.column() in (C_TAGS, C_TEXT, C_DESC):
            f |= Qt.ItemFlag.ItemIsEditable
        return f

    def kinds_for(self, it: Item) -> list[tuple[str, str | None]]:
        return [(t, it.kinds.get(t) or self.kind_of(t)) for t in it.tags]

    def data(self, idx, role=Qt.ItemDataRole.DisplayRole):
        if not idx.isValid():
            return None
        row = self.rows[idx.row()]
        it, c = row.item, idx.column()
        R = Qt.ItemDataRole
        if role == ROW_ROLE:
            return row
        if c == C_THUMB:
            return row.thumb if role == R.DecorationRole else None
        if c == C_STATUS:
            label, color = STATUS.get(row.status, ("", "muted"))
            if role == R.DisplayRole:
                return "● " + label
            if role == R.ForegroundRole:
                return QColor(theme()[color])
            if role == R.ToolTipRole:
                return row.error or None
            return None
        if c == C_FILE:
            if role == R.DisplayRole:
                return it.src.name
            if role == R.ToolTipRole:
                return str(it.src)
            return None
        if c == C_TAGS:
            if role == TAGS_ROLE:
                return self.kinds_for(it)
            if role in (R.DisplayRole, R.EditRole):
                return " ".join(it.tags)
            return None
        if c in (C_TEXT, C_DESC):
            if role in (R.DisplayRole, R.EditRole, R.ToolTipRole):
                return it.text if c == C_TEXT else it.description
            return None
        if c == C_DEST:
            if role == R.DisplayRole:
                return row.dest
            if role == R.ForegroundRole and not row.dest_changed:
                return QColor(theme()["muted"])
            if role == R.ToolTipRole:
                return row.dest
        return None

    def setData(self, idx, value, role=Qt.ItemDataRole.EditRole):
        if role != Qt.ItemDataRole.EditRole or not idx.isValid():
            return False
        row, c = self.rows[idx.row()], idx.column()
        it = row.item
        if c == C_TAGS:
            new = []
            for t in map(clean_tag, str(value).replace(",", " ").split()):
                if t and t not in new:
                    new.append(t)
            if new == it.tags:
                return False
            it.tags = new
        elif c == C_TEXT:
            it.text = str(value).strip()
        elif c == C_DESC:
            it.description = str(value).strip()
        else:
            return False
        it.touched = True
        if row.status not in ("queued", "analyzing"):
            row.status = "edited"
        self.refresh(row.id)
        self.edited.emit(row.id)
        return True

    # ---- rows ----
    def _reindex(self):
        self._index = {r.id: i for i, r in enumerate(self.rows)}

    def add(self, rows: list[Row]):
        if not rows:
            return
        n = len(self.rows)
        self.beginInsertRows(QModelIndex(), n, n + len(rows) - 1)
        self.rows += rows
        self._reindex()
        self.endInsertRows()

    def remove(self, ids: set[int]):
        keep = [r for r in self.rows if r.id not in ids]
        if len(keep) == len(self.rows):
            return
        self.beginResetModel()
        self.rows = keep; self._reindex()
        self.endResetModel()

    def clear(self):
        self.beginResetModel()
        self.rows = []; self._index = {}
        self.endResetModel()

    def row(self, rid: int) -> Row | None:
        i = self._index.get(rid)
        return self.rows[i] if i is not None else None

    def refresh(self, rid: int | None = None, cols=None):
        if not self.rows:
            return
        if rid is None:
            self.dataChanged.emit(self.index(0, 0), self.index(len(self.rows) - 1, len(HEADERS) - 1))
            return
        i = self._index.get(rid)
        if i is not None:
            a, b = (min(cols), max(cols)) if cols else (0, len(HEADERS) - 1)
            self.dataChanged.emit(self.index(i, a), self.index(i, b))

    def refresh_column(self, col: int):
        if self.rows:
            self.dataChanged.emit(self.index(0, col), self.index(len(self.rows) - 1, col))

    def items(self) -> list[Item]:
        return [r.item for r in self.rows]


FILTERS = [("All files", "all"), ("Without tags", "untagged"), ("Errors", "error"), ("Edited by me", "edited"),
           ("Will move or rename", "moving")]


class RowFilter(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.query, self.mode = "", "all"

    def set(self, query: str | None = None, mode: str | None = None):
        if query is not None:
            self.query = query.strip().lower()
        if mode is not None:
            self.mode = mode
        self.invalidate()

    def filterAcceptsRow(self, src_row, parent):
        row: Row = self.sourceModel().rows[src_row]
        it = row.item
        if self.mode == "untagged" and it.tags:
            return False
        if self.mode == "error" and row.status != "error":
            return False
        if self.mode == "edited" and row.status != "edited":
            return False
        if self.mode == "moving" and not row.dest_changed:
            return False
        if self.query:
            hay = f"{it.src.name} {' '.join(it.tags)} {it.text} {it.description}".lower()
            return all(q in hay for q in self.query.split())
        return True
